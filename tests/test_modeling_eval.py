import numpy as np
import pandas as pd
import pytest

from cfb.modeling.evaluate import evaluate, metrics, tier_lifts
from cfb.modeling.linear import LinearDesign, RidgeModel


def test_tier_lifts_top_share_minus_tier_mean():
    y = [10, 0, 0, 0, 0, 5, 1]
    pred = [9, 1, 2, 3, 4, 1, 0]
    out = tier_lifts(y, pred, ["a"] * 5 + ["b"] * 2)
    assert out.loc["a", "lift"] == pytest.approx(10 - 2)        # top 1 of 5 (ceil 20%)
    assert out.loc["b", "lift"] == pytest.approx(5 - 3)


def test_tier_pos_lift_ignores_position_scale():
    # Inside each cell the model ranks perfectly backwards; pooled within the
    # tier it still "wins" by ranking every QB above every WR.
    y = np.array([100, 90, 80, 70, 60, 1, 2, 3, 4, 5], dtype=float)
    pred = np.array([10, 11, 12, 13, 14, 5, 4, 3, 2, 1], dtype=float)   # QBs on top
    m = metrics(y, pred, ["3"] * 10, ["QB"] * 5 + ["WR"] * 5)
    assert m["tier_lift"] > 0 and m["tier_pos_lift"] < 0


def test_evaluate_pairs_models_and_compares_only_with_smaller_sets():
    rng = np.random.default_rng(0)
    test = pd.DataFrame({"par_window": rng.normal(size=200),
                         "tier": rng.choice(["3", "4-5"], 200),
                         "position": rng.choice(["QB", "WR"], 200)})
    preds = {"ridge:baseline": rng.normal(size=200),
             "ridge:full": test.par_window.to_numpy() + rng.normal(scale=0.1, size=200)}
    table, diffs = evaluate(test, preds, n_boot=50)
    assert set(table.model) == set(preds)
    assert set(zip(diffs.model, diffs.vs)) == {("ridge:full", "ridge:baseline")}
    sp = diffs[diffs.metric.eq("spearman")].iloc[0]
    assert sp["diff"] > 0.5 and sp.lo > 0 and sp.p_not_better == 0


def _train(n=120, seed=0):
    rng = np.random.default_rng(seed)
    stars = rng.choice([2.0, 3.0, 4.0], n)
    df = pd.DataFrame({"stars": stars, "rating": 0.7 + 0.07 * stars + rng.normal(0, .01, n),
                       "unrated": False, "position": rng.choice(["QB", "WR"], n),
                       "height": rng.normal(72, 2, n), "person_id": [f"p{i}" for i in range(n)]})
    df.loc[:9, "rating"] = np.nan          # stars but no rating
    df["par_window"] = 50 * (df.rating.fillna(0.91) - 0.8) + rng.normal(0, 1, n)
    return df


def test_linear_design_fills_rating_from_star_band_and_adds_interactions():
    df = _train()
    d = LinearDesign(["rating", "unrated", "position", "height"]).fit(df)
    X = d.transform(df)
    band = df[df.stars.eq(df.stars[0])].rating.median()
    assert X.loc[0, "rating"] == pytest.approx(band)
    assert {"position=QB", "position=WR", "rating:position=QB"} <= set(X.columns)
    assert "height_na" not in X.columns             # nothing missing in training
    new = df.head(2).assign(position="RB")           # unseen level -> zeros
    assert d.transform(new)[["position=QB", "position=WR"]].sum().sum() == 0


def test_ridge_model_fits_and_predicts():
    df = _train()
    m = RidgeModel(["rating", "unrated", "position"]).fit(df)
    pred = m.predict(df)
    assert pred.shape == (len(df),) and np.corrcoef(pred, df.par_window)[0, 1] > 0.5
    assert m.alpha > 0 and "rating" in m.coefficients().index
