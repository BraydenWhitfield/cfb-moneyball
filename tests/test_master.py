"""Tests for the master acquisition table (src/cfb/master.py)."""

import numpy as np
import pandas as pd
import pytest

from cfb.master import (build_master, draft_outcomes, level_lookup, level_of, prior_production,
                        window_outcomes)


def season(pid, year, team, par, level="FBS", plays=100.0, role=2):
    return {"player_id": pid, "season": year, "team": team, "level": level, "plays": plays,
            "par_fbs": par, "role": role}


def acq_row(aid, kind, pid, year, team):
    return {"acquisition_id": aid, "acquisition_type": kind, "player_id": pid,
            "acq_season": year, "acquiring_team": team}


PS = pd.DataFrame([
    season("1", 2021, "Oklahoma", 10.0), season("1", 2022, "USC", 20.0),
    season("1", 2023, "USC", 30.0), season("1", 2024, "USC", 99.0),     # outside a 3-year window
    season("2", 2024, "Miami", 5.0, level="FCS", role=1), season("2", 2025, "Miami", 7.0),
])


def test_window_outcomes_credit_any_team_and_track_the_acquiring_team():
    acq = pd.DataFrame([acq_row("r", "recruit", "1", 2021, "Oklahoma"),
                        acq_row("t", "transfer", "2", 2024, "Miami"),
                        acq_row("u", "transfer", None, 2025, "Miami")])     # unmatched
    out = window_outcomes(acq, PS, last_season=2025)
    assert out.loc["r", "par_window"] == 60 and out.loc["r", "par_window_at_team"] == 10
    assert out.loc["t", "par_window"] == 12 and out.loc["t", "tier_window"] == 2
    assert out.loc["t", "tier_window_fbs"] == 2
    assert out.loc["u", "par_window"] == 0 and out.loc["u", "seasons_played"] == 0
    assert out.window_complete.to_dict() == {"r": True, "t": True, "u": False}
    assert out.loc["u", "seasons_observed"] == 1


def test_prior_production_only_for_matched_transfers_after_2021():
    acq = pd.DataFrame([acq_row("t", "transfer", "1", 2022, "USC"),
                        acq_row("t21", "transfer", "1", 2021, "Oklahoma"),   # no 2020 data
                        acq_row("u", "transfer", None, 2023, "X"),
                        acq_row("r", "recruit", "1", 2021, "Oklahoma")])
    out = prior_production(acq, PS)
    assert out.loc["t", "prior_par"] == 10 and out.loc["t", "prior_seasons_played"] == 1
    assert out.loc[["t21", "u", "r"], "prior_par"].isna().all()
    assert out.loc["t", "prior_seasons_available"] == 1 and np.isnan(
        out.loc["r", "prior_seasons_available"])


def test_draft_outcomes_count_only_picks_after_the_acquisition():
    players = pd.DataFrame({"player_id": ["1", "2"], "drafted": [True, True],
                            "draft_year": [2024, 2022], "draft_round": [1, 3],
                            "draft_overall": [1, 70]})
    acq = pd.DataFrame([acq_row("a", "recruit", "1", 2021, "X"),
                        acq_row("b", "transfer", "2", 2022, "Y"),       # drafted before playing
                        acq_row("c", "recruit", "1", 2024, "Z")])
    out = draft_outcomes(acq, players, last_draft_year=2026)
    assert out.drafted_after.tolist() == [True, False, False]
    assert out.loc["a", "draft_round"] == 1 and np.isnan(out.loc["b", "draft_round"])
    assert out.draft_observable.tolist() == [True, True, False]   # 2024 + 3 > 2026


def test_level_of_falls_back_to_usual_level_then_other():
    lookup = level_lookup(PS)
    out = level_of(pd.Series(["Miami", "USC", "Kilgore College", None]),
                   pd.Series([2024, 2021, 2023, 2023]), lookup)
    assert out.tolist()[:3] == ["FCS", "FBS", "other"] and pd.isna(out.iloc[3])


def test_build_master_rows_filters_and_flags():
    players = pd.DataFrame({
        "person_id": ["1", "rec:r2", "rec:r3", "3"], "player_id": ["1", None, None, "3"],
        "recruit_id": ["r1", "r2", "r3", "r4"], "name": ["A", "B", "C", "D"],
        "year": [2021, 2022, 2020, 2021], "committedTo": ["Oklahoma", "USC", "USC", "USC"],
        "position_group": ["QB", "WR", "QB", "DL"], "qb_style": ["pro", None, None, None],
        "stars": [5.0, None, 3.0, 4.0], "rating": [0.99, None, 0.8, 0.9],
        "market_tier": ["5", "unrated", "3", "4"],
        "recruit_match_method": ["athlete_id", "unmatched", "unmatched", "athlete_id"],
        "in_recruits": [True] * 4, "drafted": [False] * 4, "draft_year": [np.nan] * 4,
        "draft_round": [np.nan] * 4, "draft_overall": [np.nan] * 4})
    links = pd.DataFrame({
        "portal_key": ["k1", "k2", "k3"], "person_id": ["1", "portal:k2", "portal:k3"],
        "player_id": ["1", None, None], "recruit_id": ["r1", None, None],
        "season": [2022, 2024, 2023], "firstName": ["A", "E", "F"], "lastName": ["", "", ""],
        "position_group": ["QB", "RB", "RB"], "origin": ["Oklahoma", "X", "X"],
        "destination": ["USC", "Miami", "Miami"], "withdrawn": [False, False, True],
        "went_pro": [False, False, False], "stars": [5.0, 3.0, 3.0], "rating": [1.0, 0.85, 0.85],
        "market_tier": ["5", "3", "3"], "eligibility": ["Immediate"] * 3,
        "match_method": ["name_team_season", "unmatched", "unmatched"],
        "nationwide_dependent": [False, False, False]})
    recruits_raw = pd.DataFrame({"id": ["r1", "r2", "r3", "r4"], "height": [74, 70, 75, 76],
                                 "weight": [215, 180, 220, 280], "stateProvince": ["CA"] * 4,
                                 "recruitType": ["HighSchool"] * 4})
    m = build_master(players, players, links, recruits_raw, PS).set_index("acquisition_id")
    # r3 (2020 class) and r4 (DL) are out of scope; k3 withdrew.
    assert sorted(m.index) == ["portal:k1", "portal:k2", "rec:r1", "rec:r2"]
    assert m.loc["rec:r2", "market_unrated"] and m.loc["rec:r2", "unmatched"]
    assert m.loc["rec:r2", "par_window"] == 0
    t = m.loc["portal:k1"]
    assert t.hs_stars == 5.0 and t.years_since_hs == 1 and t.qb_style == "pro"
    assert t.origin_level == "FBS" and t.par_window == 50 and t.prior_par == 10
    assert m.loc["portal:k2", "unmatched"] and m.loc["portal:k2", "acquiring_level"] == "FCS"
    assert m.loc["rec:r1", "height"] == 74
