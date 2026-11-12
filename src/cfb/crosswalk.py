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
    """Name fallback: same normalized name, team == committedTo, stat season >=
    class year, compatible position.

    Applies to recruits with no athleteId AND to recruits whose athleteId never
    appears on the field (possibly a stale/duplicate ID - see resolve_conflicts).
    """
    on_field = set(stat_players.player_id.dropna())
    eligible = recruits.athleteId.isna() | ~recruits.athleteId.isin(on_field)
    no_id = recruits[eligible].rename(columns={"id": "recruit_id"})
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

    Stale-ID upgrade: a recruit whose athleteId never appears on the field but
    who has a clean (unique, unclaimed) name match switches to the matched ID;
    the old one is kept in replaced_athlete_id. A failed upgrade leaves the
    original athlete_id match untouched.
    """
    ids = id_matches.assign(ambiguous=False, conflict=False, replaced_athlete_id=pd.NA)
    names = name_matches.assign(has_stats=~name_matches.ambiguous.astype(bool), conflict=False,
                                replaced_athlete_id=pd.NA)

    claimed_by_id = set(ids.player_id.dropna())
    has_pid = names.player_id.notna()
    claims = names.loc[has_pid, "player_id"].value_counts()
    clash = has_pid & (names.player_id.isin(claimed_by_id) | names.player_id.map(claims).gt(1))

    upgrade = names.recruit_id.isin(ids.recruit_id)    # recruit also has an athleteId
    good_upgrade = upgrade & has_pid & ~clash
    old_id = ids.set_index("recruit_id").player_id
    names.loc[good_upgrade, "replaced_athlete_id"] = names.loc[good_upgrade, "recruit_id"].map(old_id)
    ids = ids[~ids.recruit_id.isin(names.loc[good_upgrade, "recruit_id"])]
    names = names[~upgrade | good_upgrade].copy()      # failed upgrades: keep the ID row
    clash = clash.loc[names.index]

    names.loc[clash, "player_id"] = pd.NA
    names.loc[clash, "match_method"] = "unmatched"
    names.loc[clash, "has_stats"] = False
    names.loc[clash, "conflict"] = True

    out = pd.concat([ids, names], ignore_index=True)
    for col in ["has_stats", "ambiguous", "conflict"]:
        out[col] = out[col].astype(bool)
    for col in ["player_id", "replaced_athlete_id"]:
        out[col] = out[col].astype("string")
    return out[["recruit_id", "player_id", "match_method", "has_stats", "ambiguous", "conflict",
                "replaced_athlete_id"]]


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
    "ranking_suspect", "athlete_id_placeholder", "recruit_match_method", "has_stats",
    "ambiguous", "conflict", "replaced_athlete_id",
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


# --- Portal -> players (step 3) -------------------------------------------------

# Stat seasons searched, relative to the portal `season` (= the season played at
# the destination). Chosen on real data: wider windows add <0.5% matches but
# more ambiguity. Where both sides find a player they agree 99% of the time.
DESTINATION_WINDOW = (0, 1)
ORIGIN_WINDOW = (-2, -1)


def _portal_candidates(portal, stat_players, team_col, window, side):
    cand = portal.merge(stat_players, left_on=["name_key", team_col],
                        right_on=["name_key", "team"])
    offset = cand.stat_season - cand.season
    cand = cand[(offset >= window[0]) & (offset <= window[1])]
    cand = cand[_compatible_rows(cand, "position_group", "stat_position_group")]
    return cand[["portal_key", "player_id"]].assign(side=side)


def match_portal(portal: pd.DataFrame, stat_players: pd.DataFrame) -> pd.DataFrame:
    """One row per portal entry (clean_portal output) -> player_id.

    Searches the destination (stat season within DESTINATION_WINDOW) and the
    origin (ORIGIN_WINDOW) by name + team + compatible position. One distinct
    player across both sides -> matched; several -> ambiguous (unmatched).
    Portal rows carry no IDs, so match_method is "name_team_season" or "unmatched".
    match_side: "destination", "origin", "both", or <NA>.
    """
    cand = pd.concat([
        _portal_candidates(portal, stat_players, "destination", DESTINATION_WINDOW,
                           "destination"),
        _portal_candidates(portal, stat_players, "origin", ORIGIN_WINDOW, "origin"),
    ], ignore_index=True)
    summary = _summarize_candidates(cand, "portal_key", "name_team_season")
    sides = (cand.drop_duplicates(["portal_key", "side"]).groupby("portal_key").side
                 .agg(lambda s: "both" if len(s) == 2 else s.iloc[0]))
    summary["match_side"] = (summary.portal_key.map(sides)
                             .where(~summary.ambiguous).astype("string"))

    out = portal[["portal_key"]].merge(summary, on="portal_key", how="left")
    out["match_method"] = out.match_method.fillna("unmatched")
    out["ambiguous"] = out.ambiguous.astype("boolean").fillna(False).astype(bool)
    out["player_id"] = out.player_id.astype("string")
    return out


def link_recruits_to_portal(recruits: pd.DataFrame, portal: pd.DataFrame) -> pd.DataFrame:
    """Recruit <-> portal entry pairs: same name, portal origin == committedTo,
    portal season >= class year, compatible position.

    Each portal entry links to at most one recruit (two same-named recruits at
    one school -> neither, never a guess). A recruit may have several entries
    (e.g. withdrew, then re-entered the next year).
    """
    pairs = recruits.rename(columns={"id": "recruit_id"}).merge(
        portal, left_on=["name_key", "committedTo"], right_on=["name_key", "origin"],
        suffixes=("_rec", "_portal"))
    pairs = pairs[pairs.season >= pairs.year]
    pairs = pairs[_compatible_rows(pairs, "position_group_rec", "position_group_portal")]
    one_recruit = pairs.groupby("portal_key").recruit_id.transform("nunique").eq(1)
    return (pairs.loc[one_recruit, ["recruit_id", "portal_key"]]
                 .astype("string").drop_duplicates().reset_index(drop=True))


def apply_transfer_path(players, portal_matches, recruit_portal):
    """Unmatched recruits inherit the player_id of their matched portal entry.

    Catches recruits who never recorded a stat at the school they signed with,
    then played after transferring. Skipped when the recruit's entries point to
    several players, or the player_id is already claimed (conflict rule).
    Adds a `via_portal` column.
    """
    players = players.assign(via_portal=False)
    open_ = (players.recruit_match_method.eq("unmatched")
             & ~players.ambiguous & ~players.conflict)
    cand = (recruit_portal.merge(portal_matches[["portal_key", "player_id"]], on="portal_key")
                          .dropna(subset=["player_id"]))
    cand = cand[cand.recruit_id.isin(players.loc[open_, "recruit_id"])]
    per_recruit = cand.groupby("recruit_id").player_id.agg(["nunique", "first"])
    per_recruit = per_recruit.loc[per_recruit["nunique"] == 1, "first"]

    claimed = set(players.player_id.dropna())
    claims = per_recruit.value_counts()
    new_ids = per_recruit[~per_recruit.isin(claimed) & per_recruit.map(claims).eq(1)]

    hit = players.recruit_id.isin(new_ids.index)
    players.loc[hit, "player_id"] = players.loc[hit, "recruit_id"].map(new_ids)
    players.loc[hit, "person_id"] = players.loc[hit, "player_id"]
    players.loc[hit, "recruit_match_method"] = "name_team_season"
    players.loc[hit, "has_stats"] = True
    players.loc[hit, "via_portal"] = True
    return players


def _new_people(rows: pd.DataFrame, stat_players: pd.DataFrame) -> pd.DataFrame:
    """players-table rows for people with no recruit record."""
    out = rows.reindex(columns=PLAYER_COLUMNS)
    out["has_stats"] = out.player_id.isin(set(stat_players.player_id.dropna()))
    out["ambiguous"] = False
    out["conflict"] = False
    out["via_portal"] = False
    out["in_recruits"] = False
    return out


PORTAL_LINK_COLUMNS = [
    "portal_key", "person_id", "player_id", "recruit_id", "match_method", "match_side",
    "ambiguous", "link_conflict", "season", "firstName", "lastName", "position",
    "position_group", "origin", "destination", "transferDate", "eligibility", "withdrawn",
    "went_pro", "stars", "rating", "market_tier", "date_season_mismatch",
]


def build_crosswalk(recruits: pd.DataFrame, portal: pd.DataFrame,
                    stat_players: pd.DataFrame):
    """Recruits + portal -> (players, portal_links).

    players: one row per person - recruits, plus people seen only in the
             portal (in_recruits=False). Adds via_portal, n_portal_entries,
             first_portal_season.
    portal_links: one row per portal entry with its person_id.
    """
    portal_matches = match_portal(portal, stat_players)
    recruit_portal = link_recruits_to_portal(recruits, portal)
    players = apply_transfer_path(build_players(recruits, stat_players),
                                  portal_matches, recruit_portal)
    players["in_recruits"] = True

    # Each portal entry's person. A matched entry belongs to whichever recruit
    # owns that player_id; if the name link pointed at a different recruit, flag it.
    links = (portal.merge(portal_matches, on="portal_key", how="left")
                   .merge(recruit_portal, on="portal_key", how="left"))
    owner = players.dropna(subset=["player_id"]).set_index("player_id").recruit_id
    owner_of_match = links.player_id.map(owner)
    links["link_conflict"] = (links.player_id.notna() & links.recruit_id.notna()
                              & owner_of_match.fillna("").ne(links.recruit_id)).astype(bool)
    links["recruit_id"] = owner_of_match.where(links.player_id.notna(), links.recruit_id)
    links.loc[links.link_conflict & owner_of_match.isna(), "recruit_id"] = pd.NA
    rec_person = players.set_index("recruit_id").person_id
    links["person_id"] = (links.player_id
                          .fillna(links.recruit_id.map(rec_person))
                          .fillna("portal:" + links.portal_key)).astype("string")
    links["recruit_id"] = links.recruit_id.astype("string")

    # People who appear only in the portal (latest entry supplies name/position).
    extra = (links[~links.person_id.isin(set(players.person_id))]
             .sort_values("transferDate").drop_duplicates("person_id", keep="last"))
    college = resolve_college_position(extra.player_id, stat_players)
    group = extra.position_group.where(extra.position_group.ne("ATH"),
                                       extra.player_id.map(college)).fillna("ATH")
    players = pd.concat([players, _new_people(pd.DataFrame({
        "person_id": extra.person_id, "player_id": extra.player_id,
        "name": extra.firstName + " " + extra.lastName, "position_group": group,
    }), stat_players)], ignore_index=True)

    counts = links.groupby("person_id").agg(n_portal_entries=("portal_key", "size"),
                                            first_portal_season=("season", "min"))
    players = players.join(counts, on="person_id")
    players["n_portal_entries"] = players.n_portal_entries.fillna(0).astype(int)
    players["first_portal_season"] = players.first_portal_season.astype("Int64")
    return players.reset_index(drop=True), links[PORTAL_LINK_COLUMNS].reset_index(drop=True)
