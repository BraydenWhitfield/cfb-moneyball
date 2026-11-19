"""Load cached raw CFBD JSON (data/raw/) into flat DataFrames.

No API calls: run `python -m cfb.fetch --run` first to populate the cache.
ID columns are cast to pandas' nullable string type so joins across sources
compare like with like (the draft endpoint returns IDs as ints, others as str).
"""

import json
from pathlib import Path

import pandas as pd

from cfb.client import DEFAULT_CACHE_DIR


def load_raw(subdir, pattern="year=*.json", raw_dir=DEFAULT_CACHE_DIR):
    """Concatenate every cached JSON file in raw_dir/subdir matching pattern."""
    files = sorted((Path(raw_dir) / subdir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No cached files for {subdir}/{pattern} in {raw_dir}")
    frames = [pd.json_normalize(json.loads(f.read_text(encoding="utf-8"))) for f in files]
    return pd.concat(frames, ignore_index=True)


def _as_id(df, *cols):
    """Cast ID columns to nullable strings ("4431274", never 4431274.0 or "nan")."""
    for col in cols:
        values = df[col]
        if pd.api.types.is_numeric_dtype(values):
            values = values.astype("Int64")
        df[col] = values.astype("string")
    return df


def load_recruits(raw_dir=DEFAULT_CACHE_DIR):
    return _as_id(load_raw("recruiting/players", raw_dir=raw_dir), "id", "athleteId")


def load_portal(raw_dir=DEFAULT_CACHE_DIR):
    df = load_raw("player/portal", raw_dir=raw_dir)
    df["transferDate"] = pd.to_datetime(df.transferDate, utc=True)
    return df


def load_ppa(raw_dir=DEFAULT_CACHE_DIR):
    return _as_id(load_raw("ppa/players/season", raw_dir=raw_dir), "id")


def load_usage(exclude_garbage_time=False, raw_dir=DEFAULT_CACHE_DIR):
    pattern = "excludeGarbageTime=true_year=*.json" if exclude_garbage_time else "year=*.json"
    return _as_id(load_raw("player/usage", pattern, raw_dir=raw_dir), "id")


def load_stats(raw_dir=DEFAULT_CACHE_DIR):
    """Season box-score stats in long format: one row per player x stat."""
    return _as_id(load_raw("stats/player/season", raw_dir=raw_dir), "playerId")


def load_draft(raw_dir=DEFAULT_CACHE_DIR):
    return _as_id(load_raw("draft/picks", raw_dir=raw_dir), "collegeAthleteId")


def load_coaches(raw_dir=DEFAULT_CACHE_DIR):
    """One row per (coach, school, season) with games coached; coach_id as string."""
    files = sorted((Path(raw_dir) / "coaches").glob("year=*.json"))
    if not files:
        raise FileNotFoundError(f"No cached coaches files in {raw_dir}")
    frames = [pd.json_normalize(json.loads(f.read_text(encoding="utf-8")), "seasons",
                                ["id", "firstName", "lastName"]) for f in files]
    df = pd.concat(frames, ignore_index=True).rename(columns={"id": "coach_id", "year": "season"})
    return _as_id(df, "coach_id")


def load_teams(raw_dir=DEFAULT_CACHE_DIR):
    """Team list with location columns (location.state etc.)."""
    return load_raw("teams", raw_dir=raw_dir)
