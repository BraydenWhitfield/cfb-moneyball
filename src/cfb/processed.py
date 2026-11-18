"""Load data/processed/ crosswalk outputs for analysis, applying the sample rules.

The main analysis excludes recruit links made by "nationwide_name" (a name
unique among all stat lines, at any team). Those matches are probably right
but only spot-checked, so they are a robustness check:

    players = load_players()                          # main analysis
    players = load_players(include_nationwide=True)   # robustness check

Excluded recruits are DROPPED, not turned back into unmatched rows: most are
likely real players, so counting them as no-stat busts would add errors.
"""

from pathlib import Path

import pandas as pd

from cfb.client import PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NATIONWIDE = "nationwide_name"


def _dir(processed_dir):
    return PROCESSED_DIR if processed_dir is None else Path(processed_dir)


def nationwide_recruit_ids(players: pd.DataFrame) -> set:
    return set(players.loc[players.recruit_match_method.eq(NATIONWIDE), "recruit_id"].dropna())


def load_players(processed_dir=None, include_nationwide=False) -> pd.DataFrame:
    """players.parquet; by default without nationwide_name recruits."""
    players = pd.read_parquet(_dir(processed_dir) / "players.parquet")
    if include_nationwide:
        return players
    # recruit_match_method is NA for portal/draft-only people: keep them.
    keep = players.recruit_match_method.fillna("").ne(NATIONWIDE)
    return players[keep].reset_index(drop=True)


def load_portal_links(processed_dir=None, include_nationwide=False) -> pd.DataFrame:
    """portal_links.parquet, with a `nationwide_dependent` column.

    A link is nationwide-dependent when its recruit was matched by
    nationwide_name. By default such links lose that recruit (recruit_id -> NA)
    and fall back to their own identity: their portal stat match if they have
    one, else "portal:<portal_key>".
    """
    d = _dir(processed_dir)
    links = pd.read_parquet(d / "portal_links.parquet")
    players = pd.read_parquet(d / "players.parquet", columns=["recruit_id",
                                                               "recruit_match_method"])
    dependent = links.recruit_id.isin(nationwide_recruit_ids(players))
    links["nationwide_dependent"] = dependent.astype(bool)
    if include_nationwide:
        return links
    links.loc[dependent, "recruit_id"] = pd.NA
    links.loc[dependent, "person_id"] = links.loc[dependent, "player_id"].fillna(
        "portal:" + links.loc[dependent, "portal_key"])
    return links


def load_id_aliases(processed_dir=None) -> pd.DataFrame:
    """id_aliases.parquet: retired player_id -> canonical_id (see merge_split_ids)."""
    return pd.read_parquet(_dir(processed_dir) / "id_aliases.parquet")
