"""Direct test: do transfers' prior plays predict value beyond the market price?

    python -m cfb.modeling.prior_plays [--boot 2000]

Within each position (QB, RB, WR), two OLS models are fit on the training years:
  controls  portal rating (star-band median when missing), stars,
            rating_missing, unrated, destination level
  + prior   controls + log(1 + prior plays)
and scored out of time. Reported per position: the standardized prior-plays
coefficient (bootstrap CI over training rows) and the test-year Spearman gain
of "+ prior" over "controls" (paired bootstrap CI over test rows).

Primary: 2-season target (par_window), train 2022-23, test 2024. 2021 is left
out because prior plays are unobservable then (no 2020 data).
Secondary (different window, label it so): 1-season target (PAR in the
transfer season, any team), train 2022-24, test 2025; plus 2024 with the same
1-season target, to separate the window change from the year change.

Test rows: "origin_matched" (primary) = identity found at the ORIGIN school, so
prior plays are real and the outcome isn't conditioned on (matching at the
destination requires having played there); "all" = every row, where unmatched
entries have prior 0 and outcome 0 by construction.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression

from cfb.modeling.data import add_features, load_master
from cfb.processed import PROCESSED_DIR, load_id_aliases, load_portal_links
from cfb.production import build_production

POSITIONS = ["QB", "RB", "WR"]
LEVELS = ["FBS", "FCS"]          # "other" is the reference level
PRIOR = "log_prior_plays"


# --- Data ---------------------------------------------------------------------------

def season_par(ps: pd.DataFrame) -> pd.Series:
    """FBS-equivalent PAR per (player_id, season), summed over teams."""
    return ps.groupby(["player_id", "season"]).par_fbs.sum()


def transfers(master: pd.DataFrame, links: pd.DataFrame, by_season: pd.Series) -> pd.DataFrame:
    """2022-2025 transfers with features, both targets, and origin_matched."""
    t = master[master.acquisition_type.eq("transfer") & master.acq_season.between(2022, 2025)]
    t = add_features(t).merge(links[["portal_key", "match_side"]], on="portal_key", how="left")
    t["origin_matched"] = t.match_side.isin(["origin", "both"])
    keys = pd.MultiIndex.from_arrays([t.player_id.astype(object), t.acq_season.astype(int)])
    t["par_season1"] = pd.Series(by_season.reindex(keys).to_numpy(), index=t.index).fillna(0.0)
    t[PRIOR] = np.log1p(t.prior_plays.clip(lower=0))
    return t.reset_index(drop=True)


def design(df: pd.DataFrame, band_median: dict, with_prior: bool) -> np.ndarray:
    rating = df.rating.fillna(df.stars.map(band_median)).fillna(0.0)
    cols = [rating, df.stars.fillna(0.0), df.rating_missing.astype(float),
            df.unrated.astype(float)]
    cols += [df.level.eq(lvl).fillna(False).astype(float) for lvl in LEVELS]
    if with_prior:
        cols.append(df[PRIOR].fillna(0.0))
    return np.column_stack(cols)


# --- Test ---------------------------------------------------------------------------

def _spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def fit_pair(train, target):
    band = train.groupby("stars").rating.median().to_dict()
    y = train[target].to_numpy(float)
    ctrl = LinearRegression().fit(design(train, band, False), y)
    full = LinearRegression().fit(design(train, band, True), y)
    return band, ctrl, full


def prior_coefficient(train, target):
    """Coefficient on log prior plays per training-SD, in PAR units."""
    _, _, full = fit_pair(train, target)
    return full.coef_[-1] * train[PRIOR].std()


def test_position(train, test, target, n_boot=2000, seed=0) -> dict:
    band, ctrl, full = fit_pair(train, target)
    y = test[target].to_numpy(float)
    p0 = ctrl.predict(design(test, band, False))
    p1 = full.predict(design(test, band, True))
    rng = np.random.default_rng(seed)
    diffs, coefs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(_spearman(y[i], p1[i]) - _spearman(y[i], p0[i]))
    for _ in range(max(200, n_boot // 5)):        # refits: fewer resamples
        coefs.append(prior_coefficient(train.sample(len(train), replace=True,
                                                    random_state=int(rng.integers(1e9))),
                                       target))
    diffs = np.array(diffs)
    return {"n_train": len(train), "n_test": len(test),
            "coef_per_sd": prior_coefficient(train, target),
            "coef_lo": np.nanpercentile(coefs, 2.5), "coef_hi": np.nanpercentile(coefs, 97.5),
            "spearman_controls": _spearman(y, p0), "spearman_plus_prior": _spearman(y, p1),
            "gain": _spearman(y, p1) - _spearman(y, p0),
            "gain_lo": np.nanpercentile(diffs, 2.5), "gain_hi": np.nanpercentile(diffs, 97.5),
            "p_not_better": float(np.nanmean(diffs <= 0))}


def run_test(df, train_years, test_year, target, n_boot=2000) -> pd.DataFrame:
    rows = []
    for variant in ["origin_matched", "all"]:
        for pos in POSITIONS:
            d = df[df.position.eq(pos)]
            train = d[d.acq_season.between(*train_years)]
            test = d[d.acq_season.eq(test_year)]
            if variant == "origin_matched":
                train, test = train[train.origin_matched], test[test.origin_matched]
            rows.append({"test_rows": variant, "position": pos,
                         **test_position(train, test, target, n_boot)})
    return pd.DataFrame(rows)


def _fmt(res):
    r = res.copy()
    r["coef_per_sd"] = r.apply(lambda x: f"{x.coef_per_sd:+.2f} [{x.coef_lo:+.2f}, "
                                         f"{x.coef_hi:+.2f}]", axis=1)
    r["gain"] = r.apply(lambda x: f"{x.gain:+.3f} [{x.gain_lo:+.3f}, {x.gain_hi:+.3f}] "
                                  f"p={x.p_not_better:.2f}", axis=1)
    for c in ["spearman_controls", "spearman_plus_prior"]:
        r[c] = r[c].round(3)
    return r[["test_rows", "position", "n_train", "n_test", "coef_per_sd", "spearman_controls",
              "spearman_plus_prior", "gain"]].to_string(index=False)


CHECKS = [
    ("PRIMARY: 2-season target (par_window), train 2022-23, test 2024",
     (2022, 2023), 2024, "par_window"),
    ("SECONDARY (1-season target, different window): train 2022-23, test 2024",
     (2022, 2023), 2024, "par_season1"),
    ("SECONDARY (1-season target, different window): train 2022-24, test 2025",
     (2022, 2024), 2025, "par_season1"),
]


def run(n_boot=2000):
    master = load_master()
    ps, _ = build_production(load_id_aliases())
    df = transfers(master, load_portal_links(), season_par(ps))
    lines = [__doc__.split("\n\n")[0].strip(), "",
             "coef_per_sd: PAR per 1 training-SD of log(1 + prior plays), controls held "
             "fixed (95% bootstrap CI over training rows).",
             "gain: test-year Spearman(+ prior) - Spearman(controls), paired bootstrap 95% "
             "CI; p = share of resamples with gain <= 0.", ""]
    results = {}
    for title, train_years, test_year, target in CHECKS:
        res = run_test(df, train_years, test_year, target, n_boot)
        results[title] = res
        lines += ["=" * 100, title, _fmt(res), ""]
    text = "\n".join(lines)
    out = PROCESSED_DIR / "models"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prior_plays_report.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat(results, names=["check"]).to_parquet(out / "prior_plays_results.parquet")
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boot", type=int, default=2000)
    print(run(parser.parse_args(argv).boot))


if __name__ == "__main__":
    main()
