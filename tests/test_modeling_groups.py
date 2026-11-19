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


def test_head_coaches_picks_most_games_and_flags_transitions():
    from cfb.modeling.groups import head_coaches
    c = pd.DataFrame({"school": ["A", "A", "A", "B", "B"], "season": [2021, 2022, 2022, 2021, 2022],
                      "games": [12, 3, 10, 12, 12], "coach_id": ["1", "1", "2", "3", "3"],
                      "firstName": ["X"] * 5, "lastName": ["Y"] * 5})
    hc = head_coaches(c).set_index(["school", "season"])
    assert hc.loc[("A", 2022), "coach_id"] == "2" and hc.loc[("A", 2022), "transition"]
    assert not hc.loc[("B", 2022), "transition"] and not hc.loc[("A", 2021), "transition"]


def _groups_data(tau, seed=0, n_groups=60, per=8):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, tau, n_groups) if tau else np.zeros(n_groups)
    rows = []
    for split in ["train", "test"]:
        g = np.repeat(np.arange(n_groups), per)
        rows.append(pd.DataFrame({"g": g.astype(str), "split": split,
                                  "resid_w": theta[g] + rng.normal(0, 5, len(g))}))
    d = pd.concat(rows, ignore_index=True)
    return d[d.split.eq("train")], d[d.split.eq("test")]


def test_persistence_recovers_real_effects_and_not_noise():
    from cfb.modeling.groups import persistence
    tr, te = _groups_data(tau=5)
    real = persistence(tr, te, "g", n_boot=100)
    assert real["corr_lo"] > 0 and 0.5 < real["slope"] < 1.5
    tr, te = _groups_data(tau=0, seed=3)
    null = persistence(tr, te, "g", n_boot=100)
    assert null["corr_lo"] < 0.1


def test_power_curve_rises_with_effect_size():
    from cfb.modeling.groups import power_curve
    tr, te = _groups_data(tau=0)
    pc = power_curve(tr, te, "g", sigma2=25.0, taus=(0.5, 5), n_sim=60)
    assert pc.power.iloc[1] > pc.power.iloc[0] and pc.power.iloc[1] > 0.8


def test_mean_diff():
    from cfb.modeling.groups import mean_diff
    d = pd.DataFrame({"f": [1.0] * 50 + [0.0] * 50 + [np.nan],
                      "resid_w": [2.0] * 50 + [0.0] * 51})
    out = mean_diff(d, "f", n_boot=100)
    assert out["diff"] == 2.0 and out["n_yes"] == 50 and out["n_no"] == 50


def test_benjamini_hochberg_known_values():
    from cfb.modeling.groups import benjamini_hochberg
    q = benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, 0.5], index=list("abcd")))
    assert q.round(4).to_dict() == {"a": 0.04, "b": 0.0533, "c": 0.0533, "d": 0.5}


def test_permutation_pvalues_flag_a_real_group_within_strata():
    from cfb.modeling.groups import permutation_pvalues
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"g": rng.choice([f"s{i}" for i in range(20)], n),
                       "acq_season": rng.choice([2021, 2022], n),
                       "position": rng.choice(["QB", "WR"], n), "level": "FBS",
                       "resid_w": rng.normal(0, 1, n)})
    df.loc[df.g.eq("s0"), "resid_w"] += 3
    out = permutation_pvalues(df, "g", n_perm=500)
    assert out.loc["s0", "p"] < 0.01
    assert (out.drop("s0").p > 0.01).mean() > 0.8
