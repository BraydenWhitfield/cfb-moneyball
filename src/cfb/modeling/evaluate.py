"""Out-of-time metrics with paired bootstrap CIs.

  spearman          rank correlation of prediction and outcome (pooled)
  spearman_by_pos   mean of within-position Spearman (QB, RB, WR), so position
                    scale alone can't drive the score
  tier_lift         within each star tier (2, 3, 4-5, unrated): mean outcome of
                    the top TOP_SHARE by prediction minus the tier mean, then the
                    size-weighted average over tiers. "Could the model find the
                    underpriced players at a given price?"
  tier_pos_lift     the same inside tier x position cells (cells with at least
                    MIN_CELL rows), so a model can't score by putting QBs (the
                    biggest PAR) at the top of a tier

Every bootstrap resample scores all models on the same rows, so differences
between models get paired CIs.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

TOP_SHARE = 0.2
MIN_CELL = 5
METRICS = ["spearman", "spearman_by_pos", "tier_lift", "tier_pos_lift"]
SET_ORDER = ["baseline", "baseline_plus", "full"]


def _rank_corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _weighted_lift(y, pred, groups, min_n=1, top_share=TOP_SHARE):
    """Size-weighted mean over groups of (top-share mean - group mean). Ties
    in pred keep row order (stable sort), as in tier_lifts."""
    total, weight = 0.0, 0
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) < min_n:
            continue
        k = max(1, int(np.ceil(top_share * len(idx))))
        top = idx[np.argsort(-pred[idx], kind="stable")[:k]]
        total += (y[top].mean() - y[idx].mean()) * len(idx)
        weight += len(idx)
    return total / weight if weight else np.nan


def tier_lifts(y, pred, tiers, top_share=TOP_SHARE) -> pd.DataFrame:
    """Per tier: n, tier mean, top-share mean, lift."""
    df = pd.DataFrame({"y": np.asarray(y, float), "pred": np.asarray(pred, float),
                       "tier": np.asarray(tiers)})
    rows = []
    for t, g in df.groupby("tier"):
        k = max(1, int(np.ceil(top_share * len(g))))
        # stable sort, so ties keep row order and every model breaks them the same way
        top = g.sort_values("pred", ascending=False, kind="mergesort").head(k)
        rows.append({"tier": t, "n": len(g), "tier_mean": g.y.mean(), "top_mean": top.y.mean(),
                     "lift": top.y.mean() - g.y.mean()})
    return pd.DataFrame(rows).set_index("tier")


def metrics(y, pred, tiers, positions) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    positions = np.asarray(positions)
    by_pos = [_rank_corr(y[positions == p], pred[positions == p]) for p in np.unique(positions)]
    tiers = np.asarray(tiers).astype(str)
    cells = np.char.add(np.char.add(tiers, "|"), positions.astype(str))
    return {"spearman": _rank_corr(y, pred), "spearman_by_pos": np.nanmean(by_pos),
            "tier_lift": _weighted_lift(y, pred, tiers),
            "tier_pos_lift": _weighted_lift(y, pred, cells, min_n=MIN_CELL)}


def evaluate(test: pd.DataFrame, preds: dict, target="par_window", n_boot=1000, seed=0,
             references=("baseline_plus", "baseline")):
    """Point estimates, bootstrap 95% CIs, and paired differences vs references.

    preds: {model name: predictions aligned with test}. Each model is compared
    with the smaller reference feature sets of the same learner, e.g.
    "ridge:full" vs "ridge:baseline_plus" and "ridge:baseline".
    Returns (table of metrics with CIs, table of paired differences).
    """
    y = test[target].to_numpy(float)
    tiers, pos = test.tier.to_numpy(), test.position.to_numpy()
    names = list(preds)
    point = {n: metrics(y, preds[n], tiers, pos) for n in names}

    rng = np.random.default_rng(seed)
    boots = {n: [] for n in names}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for n in names:
            boots[n].append(metrics(y[idx], np.asarray(preds[n])[idx], tiers[idx], pos[idx]))
    boots = {n: pd.DataFrame(b) for n, b in boots.items()}

    rows = []
    for n in names:
        for m in METRICS:
            lo, hi = np.nanpercentile(boots[n][m], [2.5, 97.5])
            rows.append({"model": n, "metric": m, "value": point[n][m], "lo": lo, "hi": hi})
    table = pd.DataFrame(rows)

    diffs = []
    for n in names:
        learner, _, fs = n.partition(":")
        for ref in references:
            other = f"{learner}:{ref}"
            if other not in preds or SET_ORDER.index(fs) <= SET_ORDER.index(ref):
                continue
            for m in METRICS:
                d = boots[n][m] - boots[other][m]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                diffs.append({"model": n, "vs": other, "metric": m,
                              "diff": point[n][m] - point[other][m], "lo": lo, "hi": hi,
                              "p_not_better": float((d <= 0).mean())})
    return table, pd.DataFrame(diffs)
