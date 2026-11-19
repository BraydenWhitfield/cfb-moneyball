"""Do some groups of recruits consistently beat their rating?

Residual = par_window - baseline_plus (ridge) prediction, for recruits from the
2021-2023 classes with a school, QB/RB/WR/TE (TEs only here; their PPA misses
blocking). 2021-22 residuals are cross-fitted (GroupKFold on person_id), so no
player's own outcome is in his prediction; 2023 residuals come from a model
fit on 2021-22 (out of time). Residuals are winsorized at the 2021-22 1st/99th
percentiles (same caps applied to 2023): one star QB shouldn't make a school.

Group effects use empirical Bayes shrinkage toward 0:
    shrunk_g = B_g * mean_g,  B_g = tau2 / (tau2 + sigma2 / n_g)
sigma2 = pooled within-group residual variance; tau2 = between-group variance
of true effects (DerSimonian-Laird method of moments, floored at 0). tau2 = 0
means the data show no real group differences at all.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cfb.modeling.data import FEATURE_SETS, TARGET, add_features
from cfb.modeling.linear import RidgeModel

POSITIONS = ("QB", "RB", "WR", "TE")
TRAIN_YEARS, TEST_YEAR = (2021, 2022), 2023
WINSOR = (0.01, 0.99)
CROSS_FIT_FOLDS = 5


# --- Residuals -------------------------------------------------------------------

def recruit_population(master: pd.DataFrame) -> pd.DataFrame:
    df = master[master.acquisition_type.eq("recruit") & master.window_complete.astype(bool)
                & master.acq_season.between(TRAIN_YEARS[0], TEST_YEAR)
                & master.position_group.isin(POSITIONS) & master.acquiring_team.notna()]
    return add_features(df).reset_index(drop=True)


def residuals(df: pd.DataFrame, features=None, folds=CROSS_FIT_FOLDS) -> pd.DataFrame:
    """Adds pred, resid, resid_w (winsorized) and split ("train" / "test")."""
    features = features or FEATURE_SETS["recruit"]["baseline_plus"]
    df = df.copy()
    train = df.acq_season.between(*TRAIN_YEARS)
    test = df.acq_season.eq(TEST_YEAR)
    df["split"] = np.where(train, "train", np.where(test, "test", "other"))
    df["pred"] = np.nan
    tr = df[train]
    for fit_idx, hold_idx in GroupKFold(n_splits=folds).split(tr, groups=tr.person_id):
        model = RidgeModel(features).fit(tr.iloc[fit_idx])
        df.loc[tr.index[hold_idx], "pred"] = model.predict(tr.iloc[hold_idx])
    if test.any():
        df.loc[test, "pred"] = RidgeModel(features).fit(tr).predict(df[test])
    df["resid"] = df[TARGET] - df.pred
    lo, hi = df.loc[train, "resid"].quantile(list(WINSOR))
    df["resid_w"] = df.resid.clip(lo, hi)
    return df


# --- Empirical Bayes ------------------------------------------------------------------

def eb_shrink(resid: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Shrunk group effects toward 0. Returns one row per group:
    n, mean, shrunk, B (shrinkage factor), se_shrunk; attrs: tau2, sigma2."""
    d = pd.DataFrame({"r": np.asarray(resid, float), "g": np.asarray(groups, object)})
    d = d.dropna()
    g = d.groupby("g").r.agg(n="size", mean="mean", var="var")
    multi = g[g.n >= 2]
    sigma2 = float(((multi.n - 1) * multi["var"]).sum() / (multi.n - 1).sum()) \
        if len(multi) else float(d.r.var())
    w = g.n / sigma2
    mu = float((w * g["mean"]).sum() / w.sum())
    q = float((w * (g["mean"] - mu) ** 2).sum())
    denom = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (len(g) - 1)) / denom) if denom > 0 else 0.0
    b = tau2 / (tau2 + sigma2 / g.n)
    out = g.drop(columns="var").assign(B=b, shrunk=b * g["mean"],
                                       se_shrunk=np.sqrt(b * sigma2 / g.n))
    out.attrs.update(tau2=tau2, sigma2=sigma2)
    return out
