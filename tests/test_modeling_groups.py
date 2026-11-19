import numpy as np
import pandas as pd
import pytest

from cfb.modeling.groups import eb_shrink, residuals


def test_eb_shrink_finds_no_variance_when_groups_are_noise():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 10)
    out = eb_shrink(rng.normal(0, 5, 500), groups)
    assert out.attrs["tau2"] < 1.0 and out.shrunk.abs().max() < out["mean"].abs().max()


def test_eb_shrink_keeps_real_effects_and_shrinks_small_groups_more():
    rng = np.random.default_rng(1)
    effects = rng.normal(0, 3, 40)
    sizes = np.where(np.arange(40) < 20, 4, 40)
    groups = np.repeat(np.arange(40), sizes)
    r = effects[groups] + rng.normal(0, 3, len(groups))
    out = eb_shrink(r, groups)
    assert 3 < out.attrs["tau2"] < 20                     # true tau2 = 9
    assert out.loc[0:19, "B"].mean() < out.loc[20:39, "B"].mean()
    assert np.corrcoef(out.shrunk, effects)[0, 1] > 0.7


def test_residuals_are_cross_fitted_and_winsorized():
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame({
        "acq_season": rng.choice([2021, 2022, 2023], n), "person_id": [f"p{i}" for i in range(n)],
        "rating": rng.uniform(0.8, 0.95, n), "stars": 3.0, "unrated": False,
        "position": pd.Series(rng.choice(["QB", "WR"], n), dtype="string"),
        "level": pd.Series(["FBS"] * n, dtype="string")})
    df["par_window"] = 100 * (df.rating - 0.8) + rng.normal(0, 2, n)
    df.loc[0, "par_window"] = 1e4                            # outlier
    out = residuals(df, features=["rating", "unrated", "position", "level"], folds=3)
    assert out.pred.notna().all() and set(out.split) == {"train", "test"}
    assert out.resid_w.max() < out.resid.max()
    assert abs(out.loc[out.split.eq("test"), "resid"].median()) < 1
