import pandas as pd
import pytest

from cfb.clean import (clean_portal, clean_recruits, dedupe_player_seasons, market_tier,
                       norm_name, norm_team, position_group, qb_style)


def S(*values):
    return pd.Series(values, dtype="object")


def test_norm_name_handles_punctuation_suffixes_accents_and_spacing():
    out = norm_name(S("Marques Buford Jr.", "B.J. Smith", "José Pérez", "  D'Andre  Swift ",
                      "Smith-Marsette III", None))
    assert out.tolist() == ["marques buford", "bj smith", "jose perez", "dandre swift",
                            "smithmarsette", ""]


def test_norm_team_strips_whitespace_and_applies_only_known_aliases():
    out = norm_team(S("Garden City C.C. ", "Texas-Rio Grande Valley", "San José State",
                      "California (PA)", None))
    assert out.tolist()[:4] == ["Garden City C.C.", "UT Rio Grande Valley", "San José State",
                                "California (PA)"]  # D2 school must NOT become "California"
    assert pd.isna(out.iloc[4])


def test_position_group_maps_every_source_vocabulary():
    out = position_group(S("PRO", "DUAL", "QB", "Quarterback", "APB", "FB", "EDGE", "OLB",
                           "Safety", "PK", "ATH", "?"))
    assert out.tolist()[:11] == ["QB", "QB", "QB", "QB", "RB", "RB", "DL", "LB", "DB", "ST",
                                 "ATH"]
    assert pd.isna(out.iloc[11])  # unknown label


def test_qb_style_and_market_tier():
    assert qb_style(S("PRO", "DUAL", "QB")).fillna("-").tolist() == ["pro", "dual", "-"]
    assert market_tier(pd.Series([5.0, 3.0, None])).tolist() == ["5", "3", "unrated"]


def test_clean_recruits_flags_suspect_ranking():
    recruits = pd.DataFrame({
        "year": [2021] * 3, "name": ["A", "B", "C"], "committedTo": ["X"] * 3,
        "position": ["QB"] * 3, "stars": [5.0, 3.0, 3.0],
        "rating": [1.0] + [0.80] * 2,
        "ranking": [1.0, 2.0, 1.0],  # C is rated #2 (tied) but listed #1
    })
    out = clean_recruits(recruits)
    assert out.rank_from_rating.tolist() == [1, 2, 2]
    assert not out.ranking_suspect.any()  # off by 1, within tolerance

    recruits.loc[2, "ranking"] = 500.0
    assert clean_recruits(recruits).ranking_suspect.tolist() == [False, False, True]


def _portal(**overrides):
    row = {"season": 2025, "firstName": "Adrian", "lastName": "Wilson", "position": "S",
           "origin": "Washington State", "destination": "Arizona State",
           "transferDate": pd.Timestamp("2024-12-23", tz="UTC"),
           "rating": None, "stars": None, "eligibility": "Immediate"}
    row.update(overrides)
    return row


def test_clean_portal_dedupes_reentries_but_not_namesakes():
    portal = pd.DataFrame([
        _portal(),                                                 # person 1
        _portal(origin="Colorado", position="WR"),                 # namesake, different person
        _portal(origin="Yale", destination=None,                   # person 3, first entry
                transferDate=pd.Timestamp("2024-11-25", tz="UTC")),
        _portal(origin="Yale", destination="Northern Illinois",    # person 3, re-entry
                transferDate=pd.Timestamp("2024-11-26", tz="UTC")),
        _portal(origin="Ohio State", destination="Professional", firstName="Pro"),
    ])
    out = clean_portal(portal)
    assert len(out) == 4
    yale = out[out.origin == "Yale"]
    assert yale.destination.tolist() == ["Northern Illinois"]   # kept the later entry
    assert out.portal_key.is_unique
    pro = out[out.firstName == "Pro"].iloc[0]
    assert pro.went_pro and pd.isna(pro.destination)
    assert (out.market_tier == "unrated").all()                 # no stars -> unrated


def _ppa_row(**overrides):
    row = {"season": 2021, "id": "1", "name": "Reginald Todd", "position": "WR",
           "team": "Troy", "conference": "SBC", "totalPPA.all": 10.0}
    row.update(overrides)
    return row


def test_dedupe_player_seasons_drops_blank_conference_repeat_keeps_splits():
    df = pd.DataFrame([
        _ppa_row(),
        _ppa_row(conference=""),                               # exact repeat, blank conference
        _ppa_row(id="2", position="RB", **{"totalPPA.all": 0.9}),  # split: RB plays...
        _ppa_row(id="2", position="WR", **{"totalPPA.all": -0.7}), # ...and WR plays
    ])
    out = dedupe_player_seasons(df)
    assert len(out) == 3
    assert out[out.id == "1"].conference.tolist() == ["SBC"]
    assert sorted(out[out.id == "2"].position) == ["RB", "WR"]


def test_dedupe_player_seasons_refuses_to_drop_rows_with_different_numbers():
    df = pd.DataFrame([_ppa_row(), _ppa_row(conference="", **{"totalPPA.all": 99.0})])
    with pytest.raises(ValueError, match="different numbers"):
        dedupe_player_seasons(df)
