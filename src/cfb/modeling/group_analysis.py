"""Group effects in recruit residuals: persistence, FDR call-outs, coach movers.

    python -m cfb.modeling.group_analysis [--boot 1000] [--sims 300]

Writes data/processed/models/groups_report.txt and group_residuals.parquet.
See cfb.modeling.groups for the residual and shrinkage definitions.
"""

import argparse

import pandas as pd

from cfb import load
from cfb.modeling.data import load_master
from cfb.modeling.groups import (add_group_keys, benjamini_hochberg, eb_shrink, head_coaches,
                                 mean_diff, permutation_pvalues, persistence, power_curve,
                                 recruit_population, residuals, school_states)
from cfb.processed import PROCESSED_DIR

OUT_DIR = PROCESSED_DIR / "models"
FDR_Q = 0.10
GROUPINGS = {"school": "school (all positions) - main school test",
             "school_pos": "school x position",
             "coach": "head coach of the class's first season (FBS only)"}


def build_frame():
    df = residuals(recruit_population(load_master()))
    return add_group_keys(df, head_coaches(load.load_coaches()),
                          school_states(load.load_teams()))


def persistence_table(df, n_boot, sims):
    train, test = df[df.split.eq("train")], df[df.split.eq("test")]
    y21, y22 = train[train.acq_season.eq(2021)], train[train.acq_season.eq(2022)]
    rows, powers = [], {}
    for key in GROUPINGS:
        checks = [("2021-22 -> 2023", train, test, "resid_w"),
                  ("2021-22 -> 2023, raw residuals", train, test, "resid"),
                  ("2021 -> 2022", y21, y22, "resid_w")]
        if key == "coach":
            keep = ~df.transition
            checks.append(("2021-22 -> 2023, no transition classes",
                           train[keep[train.index]], test[keep[test.index]], "resid_w"))
        for label, tr, te, col in checks:
            rows.append({"grouping": key, "check": label,
                         **persistence(tr, te, key, col=col, n_boot=n_boot)})
        sigma2 = eb_shrink(train.resid_w, train[key]).attrs["sigma2"]
        powers[key] = power_curve(train, test, key, sigma2, n_sim=sims)
    return pd.DataFrame(rows), powers


def in_state_table(df):
    rows = []
    for label, part in [("2021-22", df[df.split.eq("train")]), ("2023", df[df.split.eq("test")]),
                        ("2021", df[df.acq_season.eq(2021)]),
                        ("2022", df[df.acq_season.eq(2022)])]:
        rows.append({"classes": label, **mean_diff(part, "in_state")})
    return pd.DataFrame(rows)


def callouts(df, n_perm):
    """Per family: permutation p-values on 2021-22, BH q-values, the shrunk
    effect, and the same group's 2023 (out-of-sample) mean residual."""
    train, test = df[df.split.eq("train")], df[df.split.eq("test")]
    out = {}
    for key in GROUPINGS:
        pv = permutation_pvalues(train, key, n_perm=n_perm)
        pv["q"] = benjamini_hochberg(pv.p)
        eff = eb_shrink(train.resid_w, train[key])
        later = test.groupby(key).resid_w.agg(n_2023="size", mean_2023="mean")
        out[key] = pv.join(eff[["shrunk"]]).join(later).sort_values("p")
    return out


def _fmt_callouts(tables):
    lines = []
    for key, t in tables.items():
        hits = t[t.q < FDR_Q]
        lines.append(f"  {key}: {len(t)} groups tested, {len(hits)} with q < {FDR_Q}")
        show = hits if len(hits) else t.head(3)
        label = "called out" if len(hits) else "smallest p (none survive FDR)"
        lines.append(f"    {label}:")
        cols = ["n", "mean", "shrunk", "p", "q", "n_2023", "mean_2023"]
        lines += ["    " + row for row in show[cols].round(3).to_string().splitlines()]
    return lines


def _fmt_persist(t):
    t = t.copy()
    # With tau ~ 0 every shrunk effect is ~0, so the slope divides by ~0 variance.
    t["slope"] = t.apply(lambda r: "n/a (tau ~ 0)" if r.tau < 0.5 else
                         f"{r.slope:+.2f} [{r.slope_lo:+.2f}, {r.slope_hi:+.2f}]", axis=1)
    t["corr"] = t.apply(lambda r: f"{r['corr']:+.3f} [{r.corr_lo:+.3f}, {r.corr_hi:+.3f}]",
                        axis=1)
    t["tau"] = t.tau.round(2)
    return t[["grouping", "check", "groups_train", "groups_both", "n_test_covered", "tau",
              "slope", "corr"]].to_string(index=False)


def report_lines(df, persist, powers, in_state, calls):
    lines = [__doc__.split("\n\n")[0].strip(), "",
             f"recruits: 2021-22 {int(df.split.eq('train').sum()):,} (cross-fitted residuals), "
             f"2023 {int(df.split.eq('test').sum()):,} (out of time); QB/RB/WR/TE",
             f"coach known (FBS signing school): {df.coach_id.notna().mean():.0%}; "
             f"transition classes: {int(df.transition.sum()):,}", "",
             "PERSISTENCE (shrunk group effects -> later residuals, player level).",
             "tau = estimated SD of true group effects in PAR (0 = none). slope ~1 means real",
             "and persistent, ~0 means none. 95% CIs: bootstrap over whole groups.",
             _fmt_persist(persist), "",
             "POWER: chance the 2021-22 -> 2023 correlation clears the no-effect 95th",
             "percentile if true group effects had SD tau (real group sizes):"]
    for key, pc in powers.items():
        lines.append(f"  {key:<10} " + "  ".join(
            f"tau={r.tau:g}: power {r.power:.0%} (corr {r.median_corr:+.2f})"
            for r in pc.itertuples()))
    lines += ["", "IN-STATE minus OUT-OF-STATE mean residual (PAR), bootstrap 95% CI:",
              in_state.round(2).to_string(index=False), "",
              "INDIVIDUAL GROUPS (2021-22): permutation p-values within class x position x",
              f"level strata, Benjamini-Hochberg FDR per family. mean / shrunk / mean_2023 in "
              f"PAR."] + _fmt_callouts(calls)
    return lines


def run(n_boot=1000, sims=300, n_perm=10000):
    df = build_frame()
    persist, powers = persistence_table(df, n_boot, sims)
    calls = callouts(df, n_perm)
    lines = report_lines(df, persist, powers, in_state_table(df), calls)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["acquisition_id", "person_id", "name", "acq_season", "split", "position", "tier",
            "school", "school_pos", "coach_id", "coach", "transition", "in_state",
            "par_window", "pred", "resid", "resid_w"]
    df[keep].to_parquet(OUT_DIR / "group_residuals.parquet")
    text = "\n".join(lines)
    (OUT_DIR / "groups_report.txt").write_text(text + "\n", encoding="utf-8")
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--sims", type=int, default=300)
    args = parser.parse_args(argv)
    print(run(args.boot, args.sims))


if __name__ == "__main__":
    main()
