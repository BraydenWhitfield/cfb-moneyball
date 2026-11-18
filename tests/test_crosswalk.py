"""Tests for the player crosswalk (src/cfb/crosswalk.py)."""

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
        "ranking_suspect": False, "athlete_id_placeholder": False,
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
])
def test_match_by_name_rejects_wrong_team_or_season(stat_kwargs):
    recs = recruits_df(recruit("r1", "John Smith"))
    line = {"player_id": "9", "name": "John Smith", "season": 2022, **stat_kwargs}
    stats = stats_df(stat_line(**line))
    assert len(match_recruits_by_name(recs, stats)) == 0


def test_match_by_name_accepts_a_lone_candidate_at_another_position():
    # QB recruit who became a WR (e.g. Shai Werts): position is only a tiebreaker.
    recs = recruits_df(recruit("r1", "John Smith"))
    stats = stats_df(stat_line("9", "John Smith", 2022, group="WR"))
    assert by_recruit(match_recruits_by_name(recs, stats)).loc["r1", "player_id"] == "9"


def test_match_by_name_uses_position_to_break_ties():
    recs = recruits_df(recruit("r1", "John Smith"))
    stats = stats_df(stat_line("9", "John Smith", 2022, group="DL"),
                     stat_line("10", "John Smith", 2022, group="QB"))
    out = by_recruit(match_recruits_by_name(recs, stats))
    assert out.loc["r1", "player_id"] == "10" and not out.loc["r1", "ambiguous"]

    stats = stats_df(stat_line("9", "John Smith", 2022, group="DL"),
                     stat_line("11", "John Smith", 2022, group="DB"))
    assert by_recruit(match_recruits_by_name(recs, stats)).loc["r1", "ambiguous"]


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


def test_match_by_name_also_tries_recruits_whose_athlete_id_never_plays():
    recs = recruits_df(recruit("r1", "Jamire Calvin", athlete_id="4277735", position="WR"))
    stats = stats_df(stat_line("4245147", "Jamire Calvin", 2021, group="WR"))  # different ID
    out = by_recruit(match_recruits_by_name(recs, stats))
    assert out.loc["r1", "player_id"] == "4245147"


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


def test_resolve_conflicts_upgrades_stale_athlete_id_to_clean_name_match():
    ids, names = _frames([_id_match("r1", "OLD", has_stats=False)], [_name_match("r1", "NEW")])
    out = by_recruit(resolve_conflicts(ids, names))
    assert len(out) == 1
    assert out.loc["r1", "player_id"] == "NEW"
    assert out.loc["r1", "match_method"] == "name_team_season"
    assert out.loc["r1", "replaced_athlete_id"] == "OLD"
    assert out.loc["r1", "has_stats"]


def test_resolve_conflicts_failed_upgrade_keeps_original_athlete_id():
    # r1's stale ID would upgrade to "5", but r2 owns "5" by athleteId.
    ids, names = _frames([_id_match("r1", "OLD", has_stats=False), _id_match("r2", "5")],
                         [_name_match("r1", "5")])
    out = by_recruit(resolve_conflicts(ids, names))
    assert out.loc["r1", "player_id"] == "OLD" and out.loc["r1", "match_method"] == "athlete_id"
    assert not out.loc["r1", "conflict"] and pd.isna(out.loc["r1", "replaced_athlete_id"])
    assert out.loc["r2", "player_id"] == "5"


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


# --- Portal (step 3) ------------------------------------------------------------

def portal_entry(first, last, season=2023, origin="Illinois", destination="Texas Tech",
                 position="RB", date="2022-12-15"):
    """One row of clean_portal() output."""
    from cfb.clean import clean_portal
    return clean_portal(pd.DataFrame([{
        "season": season, "firstName": first, "lastName": last, "position": position,
        "origin": origin, "destination": destination,
        "transferDate": pd.Timestamp(date, tz="UTC"), "rating": None, "stars": 3.0,
        "eligibility": "Immediate"}])).iloc[0].to_dict()


def portal_df(*rows):
    return pd.DataFrame(rows)


def test_match_portal_destination_origin_both_and_windows():
    from cfb.crosswalk import match_portal
    portal = portal_df(
        portal_entry("Kyron", "Cumby"),                          # found at both schools
        portal_entry("Dest", "Only"),                            # only plays after
        portal_entry("Too", "Late"),                             # dest line 3 seasons later
        portal_entry("Nobody", "Here"),
    )
    stats = stats_df(
        stat_line("1", "Kyron Cumby", 2022, team="Illinois", group="RB"),
        stat_line("1", "Kyron Cumby", 2023, team="Texas Tech", group="RB"),
        stat_line("2", "Dest Only", 2024, team="Texas Tech", group="RB"),   # season + 1: ok
        stat_line("3", "Too Late", 2026, team="Texas Tech", group="RB"),    # season + 3: no
    )
    out = match_portal(portal, stats).set_index("portal_key")
    keys = portal.portal_key.tolist()
    assert out.loc[keys[0], "player_id"] == "1" and out.loc[keys[0], "match_side"] == "both"
    assert out.loc[keys[1], "player_id"] == "2" and out.loc[keys[1], "match_side"] == "destination"
    for k in keys[2:]:
        assert out.loc[k, "match_method"] == "unmatched" and pd.isna(out.loc[k, "player_id"])
    assert len(out) == 4  # every entry present


def test_match_portal_sides_disagreeing_is_ambiguous():
    from cfb.crosswalk import match_portal
    portal = portal_df(portal_entry("Mark", "Williams"))
    stats = stats_df(stat_line("1", "Mark Williams", 2022, team="Illinois", group="RB"),
                     stat_line("2", "Mark Williams", 2023, team="Texas Tech", group="RB"))
    row = match_portal(portal, stats).iloc[0]
    assert row.ambiguous and row.match_method == "unmatched" and pd.isna(row.player_id)


def test_match_portal_lone_candidate_at_new_position_matches():
    # QB who became a punter at his destination (e.g. Jack Dawson).
    from cfb.crosswalk import match_portal
    portal = portal_df(portal_entry("Jack", "Dawson", position="QB"))
    stats = stats_df(stat_line("1", "Jack Dawson", 2023, team="Texas Tech", group="ST"))
    row = match_portal(portal, stats).iloc[0]
    assert row.player_id == "1" and row.match_side == "destination"


def test_link_recruits_to_portal_requires_origin_season_and_unique_recruit():
    from cfb.crosswalk import link_recruits_to_portal
    recs = recruits_df(
        recruit("r1", "Kyron Cumby", year=2021, team="Illinois", position="RB"),
        recruit("r2", "Late Signee", year=2024, team="Illinois", position="RB"),  # class after entry
        recruit("r3", "Twin Name", year=2020, team="Illinois", position="RB"),
        recruit("r4", "Twin Name", year=2021, team="Illinois", position="RB"),    # namesake
    )
    portal = portal_df(portal_entry("Kyron", "Cumby"), portal_entry("Late", "Signee"),
                       portal_entry("Twin", "Name"))
    out = link_recruits_to_portal(recs, portal)
    assert out.recruit_id.tolist() == ["r1"]


def test_build_crosswalk_transfer_path_portal_only_people_and_links():
    from cfb.crosswalk import build_crosswalk
    recs = recruits_df(
        recruit("r1", "Kyron Cumby", year=2021, team="Illinois", position="RB"),  # never played at Illinois
        recruit("r2", "Juco Guy", year=2021, team="Illinois", position="RB"),     # left for JUCO
    )
    portal = portal_df(
        portal_entry("Kyron", "Cumby"),                                            # plays at Texas Tech
        portal_entry("Juco", "Guy", destination="Garden City C.C."),               # no stats anywhere
        portal_entry("Walk", "On", origin="Delaware", destination="Towson"),       # no recruit record
    )
    stats = stats_df(stat_line("1", "Kyron Cumby", 2023, team="Texas Tech", group="RB"),
                     stat_line("9", "Walk On", 2023, team="Towson", group="RB"))
    players, links = build_crosswalk(recs, portal, stats)
    p = players.set_index("person_id")
    l = links.set_index("portal_key")
    k_cumby, k_juco, k_walk = portal.portal_key

    # r1 rescued through the portal: recruit -> portal entry -> player 1
    assert p.loc["1", "recruit_id"] == "r1" and p.loc["1", "via_portal"]
    assert p.loc["1", "recruit_match_method"] == "name_team_season"
    assert l.loc[k_cumby, "person_id"] == "1" and l.loc[k_cumby, "recruit_id"] == "r1"

    # r2 unmatched, but their portal entry still belongs to them
    assert l.loc[k_juco, "person_id"] == "rec:r2"
    assert p.loc["rec:r2", "n_portal_entries"] == 1 and not p.loc["rec:r2", "has_stats"]

    # portal-only person, keyed by player_id, not a recruit
    assert l.loc[k_walk, "person_id"] == "9"
    assert not p.loc["9", "in_recruits"] and p.loc["9", "has_stats"]

    assert players.person_id.is_unique
    assert links.person_id.isin(players.person_id).all()


# --- Draft + outputs (step 4) ------------------------------------------------------

def pick(year, overall, name, college_id=None, team="Miami", position="Quarterback"):
    return {"year": year, "overall": overall, "round": 1, "pick": overall, "name": name,
            "position": position, "collegeTeam": team, "nflTeam": "Tennessee",
            "collegeAthleteId": college_id}


def draft_df(*rows):
    df = pd.DataFrame(rows)
    df["collegeAthleteId"] = df.collegeAthleteId.astype("string")
    return df


def test_match_draft_by_id_then_name_and_drops_double_claims():
    from cfb.crosswalk import match_draft
    draft = draft_df(pick(2025, 1, "Cam Ward", college_id="7"),
                     pick(2025, 2, "No Id", team="Miami"),               # name fallback
                     pick(2024, 3, "Dup One", college_id="8"),
                     pick(2025, 4, "Dup Two", college_id="8"))            # same ID twice
    stats = stats_df(stat_line("7", "Cam Ward", 2024, team="Miami"),
                     stat_line("5", "No Id", 2024, team="Miami"))
    out = match_draft(draft, stats).set_index("draft_key")
    assert out.loc["2025-1", "player_id"] == "7"
    assert out.loc["2025-1", "draft_match_method"] == "athlete_id"
    assert out.loc["2025-2", "player_id"] == "5"
    assert out.loc["2025-2", "draft_match_method"] == "name_team_season"
    assert out.loc[["2024-3", "2025-4"], "draft_conflict"].all()
    assert (out.loc[["2024-3", "2025-4"], "draft_match_method"] == "unmatched").all()


def test_attach_draft_marks_existing_people_and_adds_new_ones():
    from cfb.crosswalk import attach_draft, build_crosswalk
    recs = recruits_df(recruit("r1", "Cam Ward", athlete_id="7", year=2021, team="Miami"))
    stats = stats_df(stat_line("7", "Cam Ward", 2024, team="Miami"))
    players, _ = build_crosswalk(recs, portal_df(portal_entry("X", "Y")), stats)
    out = attach_draft(players, draft_df(pick(2025, 1, "Cam Ward", college_id="7"),
                                         pick(2025, 9, "Unknown Kid")), stats).set_index("person_id")
    assert out.loc["7", "drafted"] and out.loc["7", "draft_overall"] == 1
    assert out.loc["draft:2025-9", "drafted"] and not out.loc["draft:2025-9", "in_recruits"]
    assert not out.drop(index=["7", "draft:2025-9"]).drafted.any()
    assert out.index.is_unique


def _write_raw(raw):
    """Minimal raw cache: one file per endpoint, shaped like the real API."""
    import json

    def put(subdir, name, records):
        path = raw / subdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records), encoding="utf-8")

    put("recruiting/players", "year=2021.json", [
        {"id": "r1", "athleteId": "7", "year": 2021, "ranking": 1, "name": "Cam Ward",
         "committedTo": "Miami", "position": "PRO", "stars": 5, "rating": 0.99},
        {"id": "r2", "athleteId": "-100", "year": 2021, "ranking": 2, "name": "Placeholder Pat",
         "committedTo": "Miami", "position": "WR", "stars": 3, "rating": 0.85}])
    put("player/portal", "year=2023.json", [
        {"season": 2023, "firstName": "Placeholder", "lastName": "Pat", "position": "WR",
         "origin": "Miami", "destination": "Texas Tech ", "transferDate": "2022-12-10T00:00:00Z",
         "rating": None, "stars": None, "eligibility": "Immediate"}])
    ppa = {"conference": "ACC", "averagePPA": {"all": 0.5}, "totalPPA": {"all": 50.0}}
    put("ppa/players/season", "year=2024.json", [
        {"season": 2024, "id": "7", "name": "Cam Ward", "position": "QB", "team": "Miami", **ppa}])
    put("stats/player/season", "year=2023.json", [
        {"season": 2023, "playerId": "8", "player": "Placeholder Pat", "position": "WR",
         "team": "Texas Tech", "conference": "B12", "category": "receiving", "statType": "YDS",
         "stat": "400"}])
    put("draft/picks", "year=2025.json", [
        {"collegeAthleteId": 7, "year": 2025, "overall": 1, "round": 1, "pick": 1,
         "name": "Cam Ward", "position": "Quarterback", "collegeTeam": "Miami",
         "nflTeam": "Tennessee"}])


def test_main_end_to_end_writes_outputs(tmp_path):
    from cfb.crosswalk import main
    raw, out = tmp_path / "raw", tmp_path / "processed"
    _write_raw(raw)
    main(out_dir=out, raw_dir=raw)

    players = pd.read_parquet(out / "players.parquet").set_index("person_id")
    links = pd.read_parquet(out / "portal_links.parquet")
    assert (out / "coverage.txt").read_text(encoding="utf-8").startswith("people:")

    assert players.loc["7", "drafted"] and players.loc["7", "qb_style"] == "pro"
    # Negative athleteId ignored -> rescued via portal (Miami -> Texas Tech, padded name)
    assert players.loc["8", "recruit_id"] == "r2" and players.loc["8", "via_portal"]
    assert players.loc["8", "athlete_id_placeholder"]
    assert links.person_id.tolist() == ["8"]


# --- merge_split_ids --------------------------------------------------------

def test_merge_split_ids_merges_one_person_split_across_seasons():
    from cfb.crosswalk import merge_split_ids
    stats = stats_df(stat_line("1", "Kevric Wiggins", 2022, team="Jackson State", group="WR"),
                     stat_line("2", "Kevric Wiggins", 2023, team="Jackson State", group="WR"),
                     stat_line("2", "Kevric Wiggins", 2024, team="Jackson State", group="WR"),
                     stat_line("9", "Someone Else", 2022, team="Jackson State"))
    out, aliases = merge_split_ids(stats)
    assert aliases[["player_id", "canonical_id"]].values.tolist() == [["1", "2"]]
    assert set(out.player_id) == {"2", "9"} and len(out) == 4


def test_merge_split_ids_keeps_namesakes_who_overlap_or_are_far_apart():
    from cfb.crosswalk import merge_split_ids
    stats = stats_df(
        # same season -> two people on one roster
        stat_line("1", "Aaron Young", 2024, team="Old Dominion"),
        stat_line("2", "Aaron Young", 2024, team="Old Dominion"),
        # two empty seasons between them -> treated as different people
        stat_line("3", "Jordan Brown", 2021, team="Fresno State"),
        stat_line("4", "Jordan Brown", 2024, team="Fresno State"),
        # same name, different teams -> not this rule
        stat_line("5", "Trey Adkins", 2021, team="South Carolina"),
        stat_line("6", "Trey Adkins", 2022, team="Mercer"))
    out, aliases = merge_split_ids(stats)
    assert aliases.empty and out.player_id.nunique() == 6


def test_apply_aliases_maps_only_retired_ids():
    from cfb.crosswalk import apply_aliases
    aliases = pd.DataFrame({"player_id": ["1"], "canonical_id": ["2"]}, dtype="string")
    assert apply_aliases(pd.Series(["1", "3"], dtype="string"), aliases).tolist() == ["2", "3"]


def test_apply_aliases_with_no_aliases_keeps_ids_as_strings():
    from cfb.crosswalk import apply_aliases
    empty = pd.DataFrame({"player_id": [], "canonical_id": []}, dtype="string")
    assert apply_aliases(pd.Series(["7"], dtype="string"), empty).tolist() == ["7"]


def test_merge_split_ids_keeps_apart_different_positions_and_two_recruits():
    from cfb.crosswalk import merge_split_ids
    stats = stats_df(
        stat_line("1", "Daniel Jackson", 2024, team="Minnesota", group="WR"),
        stat_line("2", "Daniel Jackson", 2025, team="Minnesota", group="ST"),
        stat_line("3", "Tyler Williams", 2023, team="Georgia", group="WR"),
        stat_line("4", "Tyler Williams", 2024, team="Georgia", group="WR"))
    out, aliases = merge_split_ids(stats, recruit_ids=["3", "4"])
    assert aliases.empty
    # Without the recruit records, the Williams pair would merge.
    assert len(merge_split_ids(stats)[1]) == 1
