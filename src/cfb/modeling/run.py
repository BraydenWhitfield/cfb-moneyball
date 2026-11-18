"""Fit every learner x feature set, score the test year once, write a report.

    python -m cfb.modeling.run [--boot 1000]

Writes data/processed/models/: report.txt, predictions.parquet.
Model names are "<learner>:<feature set>", e.g. "ridge:baseline_plus".
"""

import argparse

import pandas as pd

from cfb.modeling.data import FEATURE_SETS, TARGET, load_master, population, split, test_variants
from cfb.modeling.evaluate import evaluate, tier_lifts
from cfb.modeling.linear import RidgeModel
from cfb.processed import PROCESSED_DIR

OUT_DIR = PROCESSED_DIR / "models"
LEARNERS = {"ridge": RidgeModel}
SETS = ["baseline", "baseline_plus", "full"]


def fit_all(kind, train):
    models = {}
    for learner, cls in LEARNERS.items():
        for fs in SETS:
            models[f"{learner}:{fs}"] = cls(FEATURE_SETS[kind][fs]).fit(train)
    return models


def _fmt_table(table):
    t = table.assign(cell=lambda d: d.apply(
        lambda r: f"{r.value:6.3f} [{r.lo:6.3f}, {r.hi:6.3f}]", axis=1))
    return t.pivot(index="model", columns="metric", values="cell").to_string()


def _fmt_diffs(diffs):
    if diffs.empty:
        return "(none)"
    d = diffs.assign(cell=lambda x: x.apply(
        lambda r: f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] p={r.p_not_better:.2f}", axis=1))
    return d.pivot(index=["model", "vs"], columns="metric", values="cell").to_string()


def run(n_boot=1000):
    master = load_master()
    lines, all_preds = [], []
    for kind in ["recruit", "transfer"]:
        df = population(master, kind)
        train, test = split(df, kind)
        models = fit_all(kind, train)
        lines += [f"{'=' * 78}", f"{kind.upper()}S: train {len(train):,} "
                  f"({train.acq_season.min()}-{train.acq_season.max()}), test {len(test):,} "
                  f"({test.acq_season.min()})", ""]
        for name, model in models.items():
            if hasattr(model, "alpha"):
                lines.append(f"{name}: alpha {model.alpha:g}")
        if "ridge:full" in models:
            coefs = models["ridge:full"].coefficients().head(15)
            lines += ["", "ridge:full standardized coefficients (top 15 by size):",
                      coefs.round(2).to_string()]
        preds = {name: model.predict(test) for name, model in models.items()}
        all_preds.append(test[["acquisition_id", "acquisition_type", "person_id", "name",
                               "tier", "position", TARGET]].assign(
            **{name: p for name, p in preds.items()}))
        for variant, rows in test_variants(train, test, kind).items():
            idx = rows.index
            table, diffs = evaluate(rows, {n: p[idx] for n, p in preds.items()}, n_boot=n_boot)
            lines += ["", f"--- test: {variant} (n={len(rows):,}) ---", _fmt_table(table), "",
                      "paired differences (model minus reference):", _fmt_diffs(diffs)]
        shown = [n for n in preds if n.endswith((":full", ":baseline_plus"))]
        for name in shown:
            lines += ["", f"tier lifts, {name} (test: all)",
                      tier_lifts(test[TARGET], preds[name], test.tier).round(2).to_string()]
        lines.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(all_preds, ignore_index=True).to_parquet(OUT_DIR / "predictions.parquet")
    text = "\n".join(lines)
    (OUT_DIR / "report.txt").write_text(text + "\n", encoding="utf-8")
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    args = parser.parse_args(argv)
    print(run(args.boot))


if __name__ == "__main__":
    main()
