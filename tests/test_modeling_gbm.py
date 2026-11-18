import numpy as np
import pandas as pd
import pytest

import cfb.modeling.gbm as gbm
from cfb.modeling.gbm import LGBMModel


@pytest.fixture(autouse=True)
def small_grid(monkeypatch):
    monkeypatch.setattr(gbm, "GRID", {"num_leaves": [4], "min_data_in_leaf": [10],
                                      "lambda_l2": [0.0]})
    monkeypatch.setattr(gbm, "MAX_ROUNDS", 200)


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"rating": rng.uniform(0.7, 1.0, n),
                       "position": pd.Series(rng.choice(["QB", "WR"], n), dtype="string"),
                       "unrated": rng.random(n) < 0.1,
                       "person_id": [f"p{i}" for i in range(n)]})
    df.loc[df.unrated, "rating"] = np.nan
    df["par_window"] = (100 * (df.rating.fillna(0.8) - 0.8) * np.where(df.position == "QB", 3, 1)
                        + rng.normal(0, 1, n))
    return df


def test_lgbm_fits_handles_missing_and_unseen_levels():
    df = _data()
    m = LGBMModel(["rating", "position", "unrated"]).fit(df)
    pred = m.predict(df)
    assert np.corrcoef(pred, df.par_window)[0, 1] > 0.8
    assert m.rounds >= 1 and "rounds" in m.describe()
    new = df.head(3).assign(position=pd.Series(["RB"] * 3, dtype="string", index=df.index[:3]))
    assert np.isfinite(m.predict(new)).all()


def test_shap_values_sum_to_predictions():
    df = _data()
    m = LGBMModel(["rating", "position", "unrated"]).fit(df)
    shap = m.shap_values(df.head(20))
    assert list(shap.columns) == ["rating", "position", "unrated", "expected_value"]
    np.testing.assert_allclose(shap.sum(axis=1), m.predict(df.head(20)), rtol=1e-6, atol=1e-6)


def test_shap_summary_ranks_features_and_gives_direction():
    from cfb.modeling.explain import shap_summary
    df = _data()
    m = LGBMModel(["rating", "position", "unrated"]).fit(df)
    s = shap_summary(m, df).set_index("feature")
    assert s.share.sum() == pytest.approx(1.0)
    assert s.index[0] in {"rating", "position"}
    assert s.loc["rating", "direction"] > 0.5            # higher rating -> higher prediction
    assert "QB" in s.loc["position", "levels"]
