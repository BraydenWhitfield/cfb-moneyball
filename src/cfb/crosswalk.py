"""Player crosswalk: link recruits (and later portal + draft) to on-field IDs.

Step 2 scope: recruits -> CFBD player IDs, and the `players` table.

Key fact: recruit `athleteId`, stats `playerId`, PPA `id`, and draft
`collegeAthleteId` are the SAME ID space (CFBD athlete IDs).

Matching never guesses: when a rule finds more than one candidate the row is
left unmatched and flagged (`ambiguous`), and one player_id never belongs to
two recruits (`conflict`).
"""

import pandas as pd

from cfb.clean import norm_name, norm_team, position_group

MATCH_METHODS = ("athlete_id", "name_team_season", "unmatched")

# Position groups that may legitimately be the same player across sources
# (recruiting services and stats disagree on hybrid roles). Used only to
# reject implausible NAME matches, e.g. a QB recruit vs a same-named DL.
# Same group is always compatible, and ATH is compatible with everything.
COMPATIBLE_GROUPS = {
    frozenset({"DL", "LB"}),   # edge rushers: DE vs OLB
    frozenset({"LB", "DB"}),   # safeties who move to LB
    frozenset({"RB", "WR"}),   # slot / all-purpose backs
    frozenset({"WR", "TE"}),
}


# --- Implemented: shared prep ----------------------------------------------

def build_stat_players(stats: pd.DataFrame, ppa: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (player_id, stat_season, team): every on-field appearance.

    Columns: player_id, stat_season, team, player_name, name_key,
             stat_position, stat_position_group.
    `stats` is the long box-score table; `ppa` (optional) adds PPA-only rows.
    """
    frames = [stats.rename(columns={"playerId": "player_id", "season": "stat_season",
                                    "player": "player_name", "position": "stat_position"})]
    if ppa is not None:
        frames.append(ppa.rename(columns={"id": "player_id", "season": "stat_season",
                                          "name": "player_name", "position": "stat_position"}))
    cols = ["player_id", "stat_season", "team", "player_name", "stat_position"]
    out = (pd.concat([f[cols] for f in frames], ignore_index=True)
             .drop_duplicates(["player_id", "stat_season", "team"])
             .reset_index(drop=True))
    out["player_id"] = out.player_id.astype("string")
    out["team"] = norm_team(out.team)
    out["name_key"] = norm_name(out.player_name)
    out["stat_position_group"] = position_group(out.stat_position)
    return out


# --- Recruits -> players ------------------------------------------------------

def positions_compatible(group_a, group_b) -> bool:
    """True if two position groups could be the same player.

    Same group, either side "ATH", or a pair in COMPATIBLE_GROUPS -> True.
    Unknown (None / NA) on either side -> True: don't reject on missing data.
    """
    if group_a is None or group_b is None or pd.isna(group_a) or pd.isna(group_b):
        return True
    if group_a == group_b or "ATH" in (group_a, group_b):
        return True
    return frozenset({group_a, group_b}) in COMPATIBLE_GROUPS


def _compatible_rows(df, col_a, col_b):
    """Boolean mask (a Series, so an empty frame stays empty) of compatible rows."""
    flags = [positions_compatible(a, b) for a, b in zip(df[col_a], df[col_b])]
    return pd.Series(flags, index=df.index, dtype=bool)


def match_recruits_by_id(recruits: pd.DataFrame, stat_players: pd.DataFrame) -> pd.DataFrame:
    """Every recruit with an athleteId -> that ID (match_method "athlete_id").

    has_stats says whether the ID ever appears on the field. An athleteId with
    no stats is still a confident identity: the missing production is real.
    """
    with_id = recruits[recruits.athleteId.notna()]
    on_field = set(stat_players.player_id.dropna())
    return pd.DataFrame({
        "recruit_id": with_id.id.astype("string").to_numpy(),
        "player_id": with_id.athleteId.astype("string").to_numpy(),
        "match_method": "athlete_id",
        "has_stats": with_id.athleteId.isin(on_field).to_numpy(),
    })


def _summarize_candidates(candidates, left_key, method):
    """Collapse candidate pairs to one row per left_key: a unique player_id, or
    ambiguous (player_id NA, match_method "unmatched") when there are several."""
    grouped = candidates.groupby(left_key).player_id.agg(["nunique", "first"])
    ambiguous = grouped["nunique"] > 1
    return pd.DataFrame({
        left_key: grouped.index.astype("string"),
        "player_id": grouped["first"].mask(ambiguous).astype("string").to_numpy(),
        "match_method": ["unmatched" if a else method for a in ambiguous],
        "ambiguous": ambiguous.to_numpy(),
    }).reset_index(drop=True)


def match_recruits_by_name(recruits: pd.DataFrame, stat_players: pd.DataFrame) -> pd.DataFrame:
    """Fallback for recruits without an athleteId: same normalized name, team ==
    committedTo, stat season >= class year, compatible position."""
    no_id = recruits[recruits.athleteId.isna()].rename(columns={"id": "recruit_id"})
    cand = no_id.merge(stat_players, left_on=["name_key", "committedTo"],
                       right_on=["name_key", "team"])
    cand = cand[cand.stat_season >= cand.year]
    cand = cand[_compatible_rows(cand, "position_group", "stat_position_group")]
    return _summarize_candidates(cand, "recruit_id", "name_team_season")


def resolve_conflicts(id_matches: pd.DataFrame, name_matches: pd.DataFrame) -> pd.DataFrame:
    """Merge ID and name matches so each player_id belongs to at most one recruit.

    athlete_id matches always win. A name match is dropped (-> unmatched,
    conflict=True) if its player_id is claimed by an athlete_id match or by
    another name match. Ambiguous rows pass through unchanged.
    """
    ids = id_matches.assign(ambiguous=False, conflict=False)
    names = name_matches.assign(has_stats=~name_matches.ambiguous.astype(bool), conflict=False)

    claimed_by_id = set(ids.player_id.dropna())
    has_pid = names.player_id.notna()
    claims = names.loc[has_pid, "player_id"].value_counts()
    clash = has_pid & (names.player_id.isin(claimed_by_id) | names.player_id.map(claims).gt(1))
    names.loc[clash, "player_id"] = pd.NA
    names.loc[clash, "match_method"] = "unmatched"
    names.loc[clash, "has_stats"] = False
    names.loc[clash, "conflict"] = True

    out = pd.concat([ids, names], ignore_index=True)
    for col in ["has_stats", "ambiguous", "conflict"]:
        out[col] = out[col].astype(bool)
    out["player_id"] = out.player_id.astype("string")
    return out[["recruit_id", "player_id", "match_method", "has_stats", "ambiguous", "conflict"]]


def resolve_college_position(player_ids: pd.Series, stat_players: pd.DataFrame) -> pd.Series:
    """College position group per player_id: the group seen in the most seasons,
    ties to the most recent season. Players without stat lines are absent.

    WARNING: post-signing information. Use it to decide which population an ATH
    belongs to - never as a model feature.
    """
    lines = stat_players[stat_players.player_id.isin(set(player_ids.dropna()))]
    lines = lines.dropna(subset=["stat_position_group"])
    per_group = (lines.groupby(["player_id", "stat_position_group"]).stat_season
                      .agg(seasons="nunique", latest="max").reset_index()
                      .sort_values(["player_id", "seasons", "latest"]))
    best = per_group.drop_duplicates("player_id", keep="last")
    return best.set_index("player_id").stat_position_group


PLAYER_COLUMNS = [
    "person_id", "player_id", "recruit_id", "name", "year", "committedTo",
    "position_signing", "position_group", "qb_style", "stars", "rating", "market_tier",
    "ranking_suspect", "recruit_match_method", "has_stats", "ambiguous", "conflict",
]


def build_players(recruits: pd.DataFrame, stat_players: pd.DataFrame) -> pd.DataFrame:
    """The `players` table: one row per recruit, linked to on-field IDs."""
    links = resolve_conflicts(match_recruits_by_id(recruits, stat_players),
                              match_recruits_by_name(recruits, stat_players))
    df = recruits.assign(recruit_id=recruits.id.astype("string")).merge(
        links, on="recruit_id", how="left")
    df["match_method"] = df.match_method.fillna("unmatched")
    for col in ["has_stats", "ambiguous", "conflict"]:
        df[col] = df[col].astype("boolean").fillna(False).astype(bool)
    df["person_id"] = df.player_id.fillna("rec:" + df.recruit_id)

    # ATH at signing -> college position, when we know it.
    college = resolve_college_position(df.player_id, stat_players)
    ath = df.position_group.eq("ATH")
    df.loc[ath, "position_group"] = df.loc[ath, "player_id"].map(college).fillna("ATH")

    df = df.rename(columns={"position": "position_signing",
                            "match_method": "recruit_match_method"})
    return df[PLAYER_COLUMNS].reset_index(drop=True)
