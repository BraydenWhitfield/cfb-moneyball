"""Cleaning helpers shared by the crosswalk: names, teams, positions, dedupes.

Findings behind each rule are in notebooks/01_explore.ipynb (section 6).
"""

import unicodedata

import pandas as pd

# --- Names -----------------------------------------------------------------

_SUFFIXES = r"\b(?:jr|sr|ii|iii|iv|v)\b"


def _strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def norm_name(names: pd.Series) -> pd.Series:
    """Normalize person names for matching across sources.

    "Marques Buford Jr." -> "marques buford", "B.J. Smith" -> "bj smith",
    "José Pérez" -> "jose perez". Nicknames (Cam/Cameron) are NOT handled.
    """
    s = names.astype("string").fillna("").map(_strip_accents).str.lower()
    s = s.str.replace(r"[.'’`\-]", "", regex=True)
    s = s.str.replace(_SUFFIXES, "", regex=True)
    return s.str.split().str.join(" ").astype("string")


# --- Teams -----------------------------------------------------------------

# Hand-verified spelling variants -> the name used by the stats/PPA endpoints.
# Deliberately NOT fuzzy: e.g. "California (PA)" is a D2 school, not California,
# and "Jacksonville" is not Jacksonville State.
TEAM_ALIASES = {
    "Albany": "UAlbany",
    "Saint Francis (PA)": "Saint Francis",
    "St. Thomas": "St. Thomas (MN)",
    "St. Thomas University": "St. Thomas University (Fl)",
    "Southeastern Louisiana": "SE Louisiana",
    "Texas-Permian Basin": "UT Permian Basin",
    "Texas-Rio Grande Valley": "UT Rio Grande Valley",
    "UT-Rio Grande Valley": "UT Rio Grande Valley",
    "UT-Rio Grande Valley (Do not use)": "UT Rio Grande Valley",
}

# Portal "destinations" that are not schools.
NON_TEAM_DESTINATIONS = {"Professional"}


def norm_team(teams: pd.Series) -> pd.Series:
    """Strip stray whitespace and apply TEAM_ALIASES. Accents are kept:
    every source spells "San José State" and "Hawai'i" the same way."""
    s = teams.astype("string").str.strip()
    return s.replace(TEAM_ALIASES)


# --- Positions -------------------------------------------------------------

# Every label seen in recruits, portal, PPA/usage, stats, and draft -> group.
POSITION_GROUPS = {
    # QB: pre-2021 recruit classes label QBs PRO (pro-style) / DUAL (dual-threat)
    "QB": "QB", "PRO": "QB", "DUAL": "QB", "Quarterback": "QB",
    # RB: APB = all-purpose back; fullbacks grouped with backs
    "RB": "RB", "APB": "RB", "FB": "RB", "Running Back": "RB", "Fullback": "RB",
    "WR": "WR", "Wide Receiver": "WR",
    "TE": "TE", "Tight End": "TE",
    "OT": "OL", "IOL": "OL", "OG": "OL", "OC": "OL", "OL": "OL", "C": "OL", "G": "OL",
    "Offensive Tackle": "OL", "Offensive Guard": "OL", "Center": "OL",
    "DL": "DL", "DT": "DL", "SDE": "DL", "WDE": "DL", "EDGE": "DL", "DE": "DL", "NT": "DL",
    "Defensive Tackle": "DL", "Defensive End": "DL", "Defensive Edge": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "Linebacker": "LB", "Inside Linebacker": "LB", "Outside Linebacker": "LB",
    "CB": "DB", "S": "DB", "DB": "DB", "Cornerback": "DB", "Safety": "DB",
    "K": "ST", "P": "ST", "LS": "ST", "PK": "ST", "KR": "ST", "PR": "ST", "RET": "ST",
    "Place Kicker": "ST", "Punter": "ST", "Long Snapper": "ST",
    # ATH = undecided at signing; resolved later from the college position.
    "ATH": "ATH",
}


def position_group(positions: pd.Series) -> pd.Series:
    """Map raw position labels to groups (QB, RB, WR, TE, OL, DL, LB, DB, ST, ATH).
    Unknown labels (e.g. "?") become <NA>."""
    s = positions.astype("string").str.strip()
    return s.map(POSITION_GROUPS).astype("string")


def qb_style(positions: pd.Series) -> pd.Series:
    """"pro" / "dual" for PRO/DUAL recruits, else <NA>. Known at signing, so it
    is a legitimate model feature."""
    s = positions.astype("string").str.strip()
    return s.map({"PRO": "pro", "DUAL": "dual"}).astype("string")


# --- Market price ----------------------------------------------------------

def market_tier(stars: pd.Series) -> pd.Series:
    """Star rating as a category; missing stars become "unrated" (never imputed)."""
    tier = stars.astype("Int64").astype("string")
    return tier.fillna("unrated")


# --- Dataset cleaners ------------------------------------------------------

def clean_recruits(recruits: pd.DataFrame) -> pd.DataFrame:
    df = recruits.copy()
    df["name_key"] = norm_name(df.name)
    df["committedTo"] = norm_team(df.committedTo)
    df["position_group"] = position_group(df.position)
    df["qb_style"] = qb_style(df.position)
    df["market_tier"] = market_tier(df.stars)
    # `ranking` is unreliable (~91 rows hold a rank from some other list), so
    # recompute the national rank from rating within each class. Ties share a rank.
    df["rank_from_rating"] = (df.groupby("year").rating
                              .rank(ascending=False, method="min").astype("Int64"))
    df["ranking_suspect"] = (df.ranking - df.rank_from_rating).abs() > 100
    return df


def clean_portal(portal: pd.DataFrame) -> pd.DataFrame:
    """Normalize names/teams, flag special cases, and drop re-entries.

    `season` = the season the player will play at the destination (entries run
    from about December of season-1 through the spring of season).
    """
    df = portal.copy()
    df["name_key"] = norm_name(df.firstName + " " + df.lastName)
    df["origin"] = norm_team(df.origin)
    df["destination"] = norm_team(df.destination)
    df["went_pro"] = df.destination.isin(NON_TEAM_DESTINATIONS)
    df.loc[df.went_pro, "destination"] = pd.NA
    df["withdrawn"] = df.eligibility.eq("Withdrawn")
    df["position_group"] = position_group(df.position)
    df["market_tier"] = market_tier(df.stars)
    # A handful of rows have a transferDate two or more years before their season.
    df["date_season_mismatch"] = df.season - df.transferDate.dt.year > 1

    # Re-entries: same person (name + origin) entering twice in one season.
    # Name alone is NOT enough: many different players share names.
    # Keep the latest entry; on a tie prefer the one with a destination.
    df = (df.assign(_has_dest=df.destination.notna())
            .sort_values(["transferDate", "_has_dest"])
            .drop_duplicates(["name_key", "season", "origin"], keep="last")
            .drop(columns="_has_dest")
            .sort_index())
    df["portal_key"] = (df.season.astype("string") + ":" + df.name_key + ":"
                        + df.origin.fillna("unknown"))
    return df.reset_index(drop=True)


def dedupe_player_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact repeats in PPA / usage tables.

    The API returns some (id, season, team, position) rows twice: identical
    numbers, but one copy has a blank conference. Keep the copy that has one.
    Rows for the same player-season under two *different* positions are not
    repeats (they are position splits with different numbers) and are kept;
    combine_ppa_splits() / combine_usage_splits() merge those afterwards.
    """
    key = ["id", "season", "team", "position"]
    stat_cols = [c for c in df.columns if c not in key + ["name", "conference"]]
    repeats = df[df.duplicated(key, keep=False)]
    differs = repeats.groupby(key)[stat_cols].nunique(dropna=False).gt(1).any(axis=1)
    if differs.any():
        raise ValueError(f"{differs.sum()} repeated {key} groups have different numbers; "
                         "refusing to drop them. Inspect before deduplicating.")

    blank_conf = df.conference.fillna("").eq("")
    return (df.assign(_blank=blank_conf)
              .sort_values("_blank")
              .drop_duplicates(key, keep="first")
              .drop(columns="_blank")
              .sort_index()
              .reset_index(drop=True))


SPLIT_KEY = ["id", "season", "team"]


def _implied_plays(total, average):
    """Plays behind an average: total / average. No plays when average is NaN and
    total is 0 (e.g. a WR's rushing). Unknown (NaN) when average is exactly 0.
    Not rounded: the API rounds averages to 3 decimals, so this is an estimate."""
    plays = total / average.where(average != 0)
    no_plays = average.isna() & total.fillna(0).eq(0)
    return plays.mask(no_plays, 0.0)


def _combine_splits(df, metric_prefixes, weight_fn):
    """Collapse rows sharing SPLIT_KEY (one row per position) into one row.

    Additive columns (metric_prefixes, e.g. totalPPA.* or usage.*) are summed.
    The row takes its position, name, and conference from the split with the
    largest weight_fn(group). Requires dedupe_player_seasons() first.
    """
    is_split = df.duplicated(SPLIT_KEY, keep=False)
    singles = df[~is_split].assign(position_split=False, split_positions=pd.NA)
    sum_cols = [c for c in df.columns if c.startswith(metric_prefixes)]

    combined = []
    for _, group in df[is_split].groupby(SPLIT_KEY):
        row = group.loc[weight_fn(group).idxmax()].copy()
        row[sum_cols] = group[sum_cols].sum(min_count=1)
        conferences = group.conference.replace("", pd.NA).dropna()
        row["conference"] = conferences.iloc[0] if len(conferences) else row.conference
        row["position_split"] = True
        row["split_positions"] = "|".join(sorted(group.position))
        combined.append((group, row))
    return singles, combined


def combine_ppa_splits(ppa: pd.DataFrame) -> pd.DataFrame:
    """One row per (id, season, team) in the PPA table.

    For ~70 player-seasons the API returns one row per position (e.g. 25 plays
    as RB, 2 as WR). totalPPA.* is summed exactly; averagePPA.* is recomputed as
    sum(total) / sum(implied plays), or NaN when the plays can't be inferred.
    Position = the split with the most plays.
    """
    categories = [c.removeprefix("totalPPA.") for c in ppa.columns if c.startswith("totalPPA.")]
    singles, combined = _combine_splits(
        ppa, ("totalPPA.",),
        weight_fn=lambda g: _implied_plays(g["totalPPA.all"], g["averagePPA.all"]).fillna(0))

    rows = []
    for group, row in combined:
        for cat in categories:
            plays = _implied_plays(group[f"totalPPA.{cat}"], group[f"averagePPA.{cat}"])
            total_plays = plays.sum(min_count=len(plays))  # NaN if any split unknown
            row[f"averagePPA.{cat}"] = (row[f"totalPPA.{cat}"] / total_plays
                                        if total_plays and total_plays > 0 else float("nan"))
        rows.append(row)
    return _reassemble(singles, rows)


def combine_usage_splits(usage: pd.DataFrame, ppa_combined: pd.DataFrame) -> pd.DataFrame:
    """One row per (id, season, team) in the usage table.

    Split rows' usage.* values are NOT additive: their sum is ~3x what the
    player's play counts imply (each split seems to use its own denominator),
    and it can't be recovered. So merged rows get usage.* = NaN, flagged by
    position_split. Position is taken from combine_ppa_splits() output so the
    two tables agree on each player's main position.
    """
    singles, combined = _combine_splits(usage, ("usage.",),
                                        weight_fn=lambda g: g["usage.overall"])
    primary = ppa_combined.set_index(SPLIT_KEY).position
    usage_cols = [c for c in usage.columns if c.startswith("usage.")]
    rows = []
    for group, row in combined:
        key = tuple(row[SPLIT_KEY])
        if key in primary.index:
            row["position"] = primary.loc[key]
        row[usage_cols] = float("nan")
        rows.append(row)
    return _reassemble(singles, rows)


def _reassemble(singles, rows):
    out = pd.concat([singles, pd.DataFrame(rows)], ignore_index=True)
    out["split_positions"] = out.split_positions.astype("string")
    out["position_split"] = out.position_split.astype(bool)
    return out.sort_values(["season", "id"]).reset_index(drop=True)
