"""Spec for the step 2 crosswalk. Most of these fail with NotImplementedError
until the TODO functions in src/cfb/crosswalk.py are written."""

import pandas as pd
import pytest

from cfb.crosswalk import (MATCH_METHODS, build_players, build_stat_players,
                           match_recruits_by_id, match_recruits_by_name,
                           positions_compatible, resolve_college_position, resolve_conflicts)

# --- Tiny fixtures shaped like the real cleaned tables ----------------------


def recruit(rid, name, athlete_id=None, year=2021, team="Ohio State", position="QB",
            group=None, stars=4.0, rating=0.95):
    """One row of clean_recruits() output."""
    from cfb.clean import norm_name
    return {
        "id": rid, "athleteId": athlete_id, "name": name,
        "name_key": norm_name(pd.Series([name])).iloc[0],
        "year": year, "committedTo": team, "position": position,
        "position_group": group or {"PRO": "QB", "DUAL": "QB", "APB": "RB"}.get(position, position),
        "qb_style": {"PRO": "pro", "DUAL": "dual"}.get(position),
        "stars": stars, "rating": rating, "market_tier": str(int(stars)) if stars else "unrated",
        "ranking_suspect": False,
    }


def recruits_df(*rows):
    df = pd.DataFrame(rows)
    for col in ["id", "athleteId"]:
        df[col] = df[col].astype("string")
    return df


def stat_line(player_id, name, season, team="Ohio State", group="QB"):
    """One row of build_stat_players() output."""
    from cfb.clean import norm_name
    return {"player_id": player_id, "stat_season": season, "team": team, "player_name": name,
            "name_key": norm_name(pd.Series([name])).iloc[0], "stat_position": group,
            "stat_position_group": group}


def stats_df(*rows):
    df = pd.DataFrame(rows)
    df["player_id"] = df.player_id.astype("string")
    return df


def by_recruit(df):
    return df.set_index("recruit_id")


# --- build_stat_players (implemented) ---------------------------------------

def test_build_stat_players_one_row_per_player_season_team():
    stats = pd.DataFrame({
        "season": [2022, 2022, 2022], "playerId": ["7", "7", "8"],
        "player": ["Cam Ward", "Cam Ward", "Garden Guy"], "position": ["QB", "QB", "WR"],
        "team": ["Miami", "Miami", "Garden City C.C. "], "category": ["passing", "rushing", "x"],
        "statType": ["YDS", "YDS", "x"], "stat": ["1", "2", "3"],
    })
    ppa = pd.DataFrame({"season": [2022, 2023], "id": ["7", "7"], "name": ["Cam Ward"] * 2,
                        "position": ["QB", "QB"], "team": ["Miami", "Miami"]})
    out = build_stat_players(stats, ppa)
    assert len(out) == 3  # (7, 2022, Miami) once, (8, 2022), (7, 2023 from PPA only)
    assert set(out.columns) >= {"player_id", "stat_season", "team", "name_key",
                                "stat_position_group"}
    assert "Garden City C.C." in set(out.team)  # team names normalized


# --- positions_compatible -----------------------------------------------------

@pytest.mark.parametrize("a, b, expected", [
    ("QB", "QB", True),
    ("ATH", "DB", True),
    ("WR", "ATH", True),
    ("DL", "LB", True),
    ("LB", "DL", True),   # order doesn't matter
    ("RB", "WR", True),
    ("QB", "DL", False),
    ("OL", "WR", False),
    (None, "QB", True),   # unknown never rejects
    (pd.NA, "QB", True),
])
def test_positions_compatible(a, b, expected):
    assert positions_compatible(a, b) is expected


# --- match_recruits_by_id ------------------------------------------------------

def test_match_by_id_links_every_athlete_id_and_flags_missing_stats():
    recs = recruits_df(recruit("r1", "Quinn Ewers", athlete_id="100"),
                       recruit("r2", "Never Played", athlete_id="200"),
                       recruit("r3", "No Id Guy"))
    stats = stats_df(stat_line("100", "Quinn Ewers", 2022))
    out = by_recruit(match_recruits_by_id(recs, stats))

    assert set(out.index) == {"r1", "r2"}             # r3 has no athleteId
    assert out.loc["r1", "player_id"] == "100" and out.loc["r1", "has_stats"]
    assert out.loc["r2", "player_id"] == "200"        # identity known...
    assert not out.loc["r2", "has_stats"]             # ...but no production: a real bust signal
    assert (out.match_method == "athlete_id").all()


# --- match_recruits_by_name ----------------------------------------------------

def test_match_by_name_uses_name_team_and_season():
    recs = recruits_df(recruit("r1", "Marques Buford Jr.", team="Nebraska", position="CB",
                               group="DB"))
    stats = stats_df(stat_line("55", "Marques Buford", 2022, team="Nebraska", group="DB"))
    out = by_recruit(match_recruits_by_name(recs, stats))
    assert out.loc["r1", "player_id"] == "55"
    assert out.loc["r1", "match_method"] == "name_team_season"
    assert not out.loc["r1", "ambiguous"]


@pytest.mark.parametrize("stat_kwargs", [
    {"team": "Michigan"},           # wrong team
    {"season": 2020},               # before the recruit's class year (2021)
    {"group": "DL"},                # QB recruit vs same-named lineman
])
def test_match_by_name_rejects_wrong_team_season_or_position(stat_kwargs):
    recs = recruits_df(recruit("r1", "John Smith"))
    line = {"player_id": "9", "name": "John Smith", "season": 2022, **stat_kwargs}
    stats = stats_df(stat_line(**line))
    assert len(match_recruits_by_name(recs, stats)) == 0


def test_match_by_name_ignores_recruits_that_have_athlete_id():
    recs = recruits_df(recruit("r1", "John Smith", athlete_id="9"))
    stats = stats_df(stat_line("9", "John Smith", 2022))
    assert len(match_recruits_by_name(recs, stats)) == 0


def test_match_by_name_marks_ambiguous_instead_of_guessing():
    recs = recruits_df(recruit("r1", "Chris Johnson"))
    stats = stats_df(stat_line("1", "Chris Johnson", 2022), stat_line("2", "Chris Johnson", 2023))
    out = by_recruit(match_recruits_by_name(recs, stats))
    assert out.loc["r1", "ambiguous"]
    assert pd.isna(out.loc["r1", "player_id"])
    assert out.loc["r1", "match_method"] == "unmatched"


def test_match_by_name_same_player_multiple_seasons_is_not_ambiguous():
    recs = recruits_df(recruit("r1", "Chris Johnson"))
    stats = stats_df(stat_line("1", "Chris Johnson", 2022), stat_line("1", "Chris Johnson", 2023))
    out = by_recruit(match_recruits_by_name(recs, stats))
    assert out.loc["r1", "player_id"] == "1" and not out.loc["r1", "ambiguous"]


# --- resolve_conflicts -----------------------------------------------------------

def _id_match(rid, pid, has_stats=True):
    return {"recruit_id": rid, "player_id": pid, "match_method": "athlete_id",
            "has_stats": has_stats}


def _name_match(rid, pid, ambiguous=False):
    return {"recruit_id": rid, "player_id": pid,
            "match_method": "unmatched" if ambiguous else "name_team_season",
            "ambiguous": ambiguous}


def _frames(id_rows, name_rows):
    cols_id = ["recruit_id", "player_id", "match_method", "has_stats"]
    cols_nm = ["recruit_id", "player_id", "match_method", "ambiguous"]
    ids = pd.DataFrame(id_rows, columns=cols_id).astype({"player_id": "string"})
    names = pd.DataFrame(name_rows, columns=cols_nm).astype({"player_id": "string"})
    return ids, names


def test_resolve_conflicts_athlete_id_wins_over_name_match():
    ids, names = _frames([_id_match("r1", "100")], [_name_match("r2", "100")])
    out = by_recruit(resolve_conflicts(ids, names))
    assert out.loc["r1", "player_id"] == "100" and not out.loc["r1", "conflict"]
    assert out.loc["r2", "match_method"] == "unmatched" and out.loc["r2", "conflict"]
    assert pd.isna(out.loc["r2", "player_id"]) and not out.loc["r2", "has_stats"]


def test_resolve_conflicts_drops_all_name_matches_that_collide():
    ids, names = _frames([], [_name_match("r1", "5"), _name_match("r2", "5"),
                              _name_match("r3", "6")])
    out = by_recruit(resolve_conflicts(ids, names))
    assert out.loc[["r1", "r2"], "conflict"].all()
    assert (out.loc[["r1", "r2"], "match_method"] == "unmatched").all()
    assert out.loc["r3", "player_id"] == "6" and out.loc["r3", "has_stats"]


def test_resolve_conflicts_passes_ambiguous_through():
    ids, names = _frames([_id_match("r1", "100", has_stats=False)],
                         [_name_match("r2", pd.NA, ambiguous=True)])
    out = by_recruit(resolve_conflicts(ids, names))
    assert out.loc["r2", "ambiguous"] and not out.loc["r2", "conflict"]
    assert not out.loc["r1", "has_stats"] and not out.loc["r1", "ambiguous"]
    assert set(out.match_method) <= set(MATCH_METHODS)


# --- resolve_college_position ------------------------------------------------------

def test_resolve_college_position_most_seasons_then_most_recent():
    stats = stats_df(
        stat_line("1", "A", 2021, group="WR"), stat_line("1", "A", 2022, group="WR"),
        stat_line("1", "A", 2023, group="RB"),                       # WR in 2 of 3 seasons
        stat_line("2", "B", 2021, group="DB"), stat_line("2", "B", 2022, group="LB"),  # tie
    )
    out = resolve_college_position(pd.Series(["1", "2", "3"], dtype="string"), stats)
    assert out["1"] == "WR"
    assert out["2"] == "LB"          # tie -> most recent season
    assert "3" not in out.index      # no stat lines


# --- build_players -------------------------------------------------------------------

def test_build_players_one_row_per_recruit_with_person_ids_and_ath_resolution():
    recs = recruits_df(
        recruit("r1", "Quinn Ewers", athlete_id="100", position="PRO", group="QB"),
        recruit("r2", "Ath Lete", position="ATH", group="ATH"),          # name match, plays WR
        recruit("r3", "Ghost Player", position="ATH", group="ATH", stars=None),  # unmatched
    )
    stats = stats_df(stat_line("100", "Quinn Ewers", 2022),
                     stat_line("300", "Ath Lete", 2022, group="WR"))
    out = by_recruit(build_players(recs, stats))

    assert len(out) == 3 and out.person_id.is_unique
    assert out.loc["r1", "person_id"] == "100"
    assert out.loc["r1", "recruit_match_method"] == "athlete_id"
    assert out.loc["r1", "qb_style"] == "pro" and out.loc["r1", "position_signing"] == "PRO"

    assert out.loc["r2", "person_id"] == "300"
    assert out.loc["r2", "recruit_match_method"] == "name_team_season"
    assert out.loc["r2", "position_group"] == "WR"   # ATH resolved from college position

    assert out.loc["r3", "person_id"] == "rec:r3"
    assert out.loc["r3", "recruit_match_method"] == "unmatched"
    assert out.loc["r3", "position_group"] == "ATH"  # nothing to resolve from
    assert not out.loc["r3", "has_stats"]
    assert out.loc["r3", "market_tier"] == "unrated"
