import numpy as np
import pandas as pd

from cfb.modeling.prior_plays import PRIOR, season_par, test_position as position_test


def test_season_par_sums_teams_within_a_season():
    ps = pd.DataFrame({"player_id": ["1", "1", "1"], "season": [2024, 2024, 2025],
                       "par_fbs": [2.0, 3.0, 7.0]})
    out = season_par(ps)
    assert out.loc[("1", 2024)] == 5.0 and out.loc[("1", 2025)] == 7.0


def _frame(n, seed, effect):
    rng = np.random.default_rng(seed)
    stars = rng.choice([2.0, 3.0, 4.0], n)
    df = pd.DataFrame({"stars": stars, "rating": 0.6 + 0.08 * stars,
                       "rating_missing": False, "unrated": False,
                       "level": pd.Series(rng.choice(["FBS", "FCS"], n), dtype="string")})
    df[PRIOR] = rng.uniform(0, 6, n)
    df["par_window"] = 10 * (stars - 3) + effect * df[PRIOR] + rng.normal(0, 3, n)
    return df


def test_position_test_detects_a_real_prior_effect_and_not_a_fake_one():
    real = position_test(_frame(400, 0, 5.0), _frame(300, 1, 5.0), "par_window", n_boot=200)
    assert real["coef_lo"] > 0 and real["gain_lo"] > 0 and real["p_not_better"] < 0.05
    none = position_test(_frame(400, 0, 0.0), _frame(300, 1, 0.0), "par_window", n_boot=200)
    assert none["coef_lo"] < 0 < none["coef_hi"] and none["gain_lo"] < 0
