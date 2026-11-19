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


# --- Group keys --------------------------------------------------------------------

def head_coaches(coaches: pd.DataFrame) -> pd.DataFrame:
    """(school, season) -> the coach with the most games that season (interims
    and fired coaches lose), and whether he differs from the previous season."""
    from cfb.clean import norm_team
    c = coaches.assign(school=norm_team(coaches.school),
                       coach=coaches.firstName + " " + coaches.lastName)
    main = (c.sort_values(["school", "season", "games"], ascending=[True, True, False])
             .drop_duplicates(["school", "season"])[["school", "season", "coach_id", "coach"]])
    prev = main.assign(season=main.season + 1).rename(
        columns={"coach_id": "prev_coach_id", "coach": "prev_coach"})
    out = main.merge(prev, on=["school", "season"], how="left")
    out["transition"] = out.prev_coach_id.notna() & out.prev_coach_id.ne(out.coach_id)
    return out.drop(columns="prev_coach")


def school_states(teams: pd.DataFrame) -> pd.Series:
    from cfb.clean import norm_team
    t = teams.assign(school=norm_team(teams.school))
    return t.drop_duplicates("school").set_index("school")["location.state"]


def add_group_keys(df: pd.DataFrame, coaches_by_season: pd.DataFrame,
                   states: pd.Series) -> pd.DataFrame:
    """school, school_pos, coach (FBS signing schools only), transition, in_state."""
    df = df.copy()
    df["school"] = df.acquiring_team.astype(object)
    df["school_pos"] = df.school + "|" + df.position.astype(str)
    hc = coaches_by_season.rename(columns={"school": "acquiring_team", "season": "acq_season"})
    df = df.merge(hc[["acquiring_team", "acq_season", "coach_id", "coach", "transition"]],
                  on=["acquiring_team", "acq_season"], how="left")
    df["coach_id"] = df.coach_id.astype(object).where(df.coach_id.notna(), np.nan)
    df["transition"] = df.transition.fillna(False).astype(bool)
    state = df.acquiring_team.map(states)
    df["in_state"] = np.where(state.isna() | df.home_state.isna(), np.nan,
                              (state == df.home_state).astype(float))
    return df


# --- Persistence -------------------------------------------------------------------

def _slope_corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    c = np.cov(x, y)
    return c[0, 1] / c[0, 0], c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])


def _persist_once(train, test, key, col):
    eff = eb_shrink(train[col], train[key])
    t = test[test[key].isin(eff.index)]
    x = t[key].map(eff.shrunk).to_numpy(float)
    return _slope_corr(x, t[col].to_numpy(float)) + (eff.attrs["tau2"], len(t))


def persistence(train, test, key, col="resid_w", n_boot=1000, seed=0) -> dict:
    """Shrunk effects from `train` vs residuals in `test`, at the player level.

    slope ~1: effects real and persistent; ~0: none. CIs from a bootstrap that
    resamples whole groups (train and test rows together) and re-estimates the
    effects each time.
    """
    train, test = train.dropna(subset=[key]), test.dropna(subset=[key])
    slope, corr, tau2, n_test = _persist_once(train, test, key, col)
    groups = np.array(sorted(set(train[key]) & set(test[key])), dtype=object)
    tr_idx = train.groupby(key).indices
    te_idx = test.groupby(key).indices
    tr_r, te_r = train[col].to_numpy(float), test[col].to_numpy(float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        tr_rows = [tr_idx[groups[p]] for p in pick]
        te_rows = [te_idx[groups[p]] for p in pick]
        tr = pd.DataFrame({key: np.repeat(np.arange(len(pick)), [len(r) for r in tr_rows]),
                           col: tr_r[np.concatenate(tr_rows)]})
        te = pd.DataFrame({key: np.repeat(np.arange(len(pick)), [len(r) for r in te_rows]),
                           col: te_r[np.concatenate(te_rows)]})
        boots.append(_persist_once(tr, te, key, col)[:2])
    boots = np.array(boots, float)
    return {"groups_train": train[key].nunique(), "groups_both": len(groups),
            "n_test_covered": n_test, "tau2": tau2, "tau": np.sqrt(tau2),
            "slope": slope, "slope_lo": np.nanpercentile(boots[:, 0], 2.5),
            "slope_hi": np.nanpercentile(boots[:, 0], 97.5),
            "corr": corr, "corr_lo": np.nanpercentile(boots[:, 1], 2.5),
            "corr_hi": np.nanpercentile(boots[:, 1], 97.5)}


def power_curve(train, test, key, sigma2, taus=(1, 2, 3, 5, 8), n_sim=300, seed=0):
    """Power of the persistence correlation, using the real group sizes.

    For each true effect SD tau: draw effects ~ N(0, tau^2), add N(0, sigma2)
    noise per player in train and test, run the same EB + correlation. Power =
    share of simulations whose correlation beats the 95th percentile of the
    no-effect (tau = 0) simulations.
    """
    train, test = train.dropna(subset=[key]), test.dropna(subset=[key])
    groups = sorted(set(train[key]) & set(test[key]))
    n_tr = train[key].value_counts().reindex(groups).to_numpy()
    n_te = test[key].value_counts().reindex(groups).to_numpy()
    g_tr = np.repeat(np.arange(len(groups)), n_tr)
    g_te = np.repeat(np.arange(len(groups)), n_te)
    rng = np.random.default_rng(seed)
    sd = np.sqrt(sigma2)

    def sim(tau):
        out = []
        for _ in range(n_sim):
            theta = rng.normal(0, tau, len(groups)) if tau > 0 else np.zeros(len(groups))
            eff = eb_shrink(theta[g_tr] + rng.normal(0, sd, len(g_tr)), g_tr)
            x = eff.shrunk.reindex(g_te).to_numpy(float)
            out.append(_slope_corr(x, theta[g_te] + rng.normal(0, sd, len(g_te)))[1])
        return np.array(out, float)

    null95 = np.nanpercentile(sim(0), 95)
    rows = []
    for tau in taus:
        c = sim(tau)
        rows.append({"tau": tau, "median_corr": np.nanmedian(c),
                     "power": float(np.nanmean(c > null95))})
    return pd.DataFrame(rows)


def mean_diff(df, flag, col="resid_w", n_boot=2000, seed=0) -> dict:
    """Mean residual where flag == 1 minus where flag == 0, bootstrap CI."""
    d = df.dropna(subset=[flag])
    a, b = d.loc[d[flag].eq(1), col].to_numpy(), d.loc[d[flag].eq(0), col].to_numpy()
    rng = np.random.default_rng(seed)
    boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    return {"n_yes": len(a), "n_no": len(b), "diff": a.mean() - b.mean(),
            "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5)}


# --- Multiple comparisons --------------------------------------------------------------

STRATA = ["acq_season", "position", "level"]


def permutation_pvalues(df, key, col="resid_w", strata=STRATA, n_perm=10000, seed=0):
    """Two-sided permutation p-value for each group's mean residual.

    Residuals are shuffled within strata (class year x position x level), so a
    group isn't "significant" just for signing many QBs or FBS players.
    p = (1 + #{|perm mean| >= |observed mean|}) / (1 + n_perm).
    """
    d = df.dropna(subset=[key]).reset_index(drop=True)
    codes, groups = pd.factorize(d[key])
    counts = np.bincount(codes)
    r = d[col].to_numpy(float)
    observed = np.bincount(codes, weights=r) / counts
    strata_idx = list(d.groupby(strata, dropna=False).indices.values())
    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(groups))
    perm = r.copy()
    for _ in range(n_perm):
        for idx in strata_idx:
            perm[idx] = r[idx][rng.permutation(len(idx))]
        means = np.bincount(codes, weights=perm) / counts
        exceed += np.abs(means) >= np.abs(observed) - 1e-12
    return pd.DataFrame({"n": counts, "mean": observed,
                         "p": (1 + exceed) / (1 + n_perm)}, index=pd.Index(groups, name=key))


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """BH-adjusted q-values (monotone, capped at 1)."""
    order = p.sort_values().index
    m = len(p)
    ranked = p.loc[order].to_numpy() * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return pd.Series(q, index=order).reindex(p.index)
