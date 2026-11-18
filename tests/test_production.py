"""Tests for player-season production (src/cfb/production.py)."""

import numpy as np
import pandas as pd
import pytest

from cfb.production import (FALLBACK_FCS_DISCOUNT, add_fbs_equivalent, add_par, add_roles,
                            box_plays, build_player_seasons, estimate_exchange_rate,
                            implied_plays, replacement_levels, season_position, team_level)


def S(*values, dtype=None):
    return pd.Series(values, dtype=dtype)


def test_team_level_maps_conferences():
    out = team_level(S("SEC", "Ind", "MVFC", "INDAA", "", None))
    assert out.tolist() == ["FBS", "FBS", "FCS", "FCS", "other", "other"]


def test_implied_plays_only_when_the_rounded_average_pins_it_down():
    out = implied_plays(S(10.0, 0.0, 0.3, 5.0), S(0.5, np.nan, 0.01, -0.25))
    assert out.iloc[0] == pytest.approx(20) and out.iloc[1] == 0
    assert np.isnan(out.iloc[2])                      # |average| < 0.05
    assert np.isnan(out.iloc[3])                      # signs disagree: inconsistent row


def test_season_position_from_play_mix():
    listed = S("WR", "QB", "QB", "QB", "TE", "RB", dtype="string")
    att = S(250, 60, 0, 0, 30, 1)
    car = S(40, 200, 2, 25, 90, 150)
    rec = S(0, 0, 30, 1, 2, 20)
    out = season_position(listed, att, car, rec)
    # converted WR who played QB; option QB stays QB; QB -> WR; QB -> RB;
    # run-first QB listed TE; RB throwing one pass stays RB
    assert out.tolist() == ["QB", "QB", "WR", "RB", "QB", "RB"]


def test_box_plays_sums_attempts_carries_receptions():
    stats = pd.DataFrame({
        "id": ["1"] * 4, "season": [2023] * 4, "team": ["Iowa"] * 4,
        "category": ["passing", "rushing", "receiving", "passing"],
        "statType": ["ATT", "CAR", "REC", "YDS"], "stat": ["300", "40", "1", "2500"]})
    row = box_plays(stats).iloc[0]
    assert (row.ATT, row.CAR, row.REC, row.box_plays) == (300, 40, 1, 341)


def _ppa(rows):
    """rows: (id, season, team, position, conference, total, pass_total, pass_avg)."""
    df = pd.DataFrame(rows, columns=["id", "season", "team", "position", "conference",
                                     "totalPPA.all", "totalPPA.pass", "averagePPA.pass"])
    df["totalPPA.rush"] = 0.0
    df["averagePPA.rush"] = np.nan
    return df


def test_build_player_seasons_prefers_implied_plays_then_calibrated_box():
    ppa = _ppa([("1", 2023, "A", "WR", "SEC", 10.0, 10.0, 0.5),     # implied 20
                ("2", 2023, "A", "WR", "SEC", 0.2, 0.2, 0.01),     # not pinned down
                ("3", 2023, "A", "WR", "SEC", 1.0, 1.0, np.nan)])  # no average, no box
    stats = pd.DataFrame({"id": ["1", "2"], "season": [2023, 2023], "team": ["A", "A"],
                          "category": ["receiving"] * 2, "statType": ["REC"] * 2,
                          "stat": ["16", "8"]})
    ps, factor = build_player_seasons(ppa, stats)
    ps = ps.set_index("player_id")
    assert factor == {"WR": 1.25}                     # 20 implied / 16 receptions
    assert ps.loc["1", "plays"] == pytest.approx(20) and ps.loc["1", "plays_source"] == "implied"
    assert ps.loc["2", "plays"] == pytest.approx(10) and ps.loc["2", "plays_source"] == "box"


def _ps(rows):
    return pd.DataFrame(rows, columns=["player_id", "season", "team", "position_group", "level",
                                       "total_ppa", "plays"])


def test_par_is_ppa_above_level_specific_replacement():
    rows = [(str(i), 2023, "A", "RB", "FBS", 0.1 * i * 100, 100.0) for i in range(1, 11)]
    rows += [("x", 2023, "A", "RB", "other", 50.0, 100.0)]
    ps = _ps(rows)
    repl = replacement_levels(ps)
    expected = np.quantile([0.1 * i for i in range(1, 11)], 0.2)
    assert repl.loc[("RB", 2023, "FBS")] == pytest.approx(expected)
    out = add_par(ps, repl).set_index("player_id")
    assert out.loc["10", "par"] == pytest.approx(100 - expected * 100)
    assert out.loc["x", "par"] == 0                   # level "other" has no replacement


def test_exchange_rate_recovers_a_known_ratio():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(80):   # FCS season t, FBS season t+1 at half the rate above replacement
        r = rng.uniform(0.1, 0.5)
        rows.append((f"p{i}", 2022, "F", "WR", "FCS", 0, 40.0, r))
        rows.append((f"p{i}", 2023, "B", "WR", "FBS", 0, 40.0, r / 2))
    ps = pd.DataFrame(rows, columns=["player_id", "season", "team", "position_group", "level",
                                     "total_ppa", "plays", "rate"])
    ps["par"] = ps.rate * ps.plays
    out = estimate_exchange_rate(ps.drop(columns="rate"), n_boot=200)
    assert out["source"] == "movers" and out["k"] == pytest.approx(0.5)
    assert out["n_up"] == 80 and out["n_down"] == 0

    few = estimate_exchange_rate(ps[ps.player_id.isin(["p0", "p1"])].drop(columns="rate"),
                                 n_boot=50)
    assert few["source"] == "fallback" and few["k"] == FALLBACK_FCS_DISCOUNT


def test_fbs_equivalent_scales_fcs_and_zeroes_other():
    ps = pd.DataFrame({"level": ["FBS", "FCS", "other"], "par": [10.0, 10.0, 10.0]})
    assert add_fbs_equivalent(ps, 0.4).par_fbs.tolist() == [10.0, 4.0, 0.0]


def test_roles_rotation_and_starters_by_plays_rank():
    ps = _ps([("a", 2023, "A", "WR", "FBS", 0, 90.0), ("b", 2023, "A", "WR", "FBS", 0, 80.0),
              ("c", 2023, "A", "WR", "FBS", 0, 70.0), ("d", 2023, "A", "WR", "FBS", 0, 60.0),
              ("e", 2023, "A", "WR", "FBS", 0, 5.0), ("f", 2023, "A", "QB", "other", 0, 300.0)])
    roles = add_roles(ps).set_index("player_id").role
    assert roles.to_dict() == {"a": 2, "b": 2, "c": 2, "d": 1, "e": 0, "f": 0}
