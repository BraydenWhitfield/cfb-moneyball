"""Model datasets: populations, signing-time features, and out-of-time splits.

Populations (complete outcome windows only, from data/processed/master.parquet):
  recruit   2021-2023 classes WITH a committedTo. A recruit with no school was
            not acquired by anyone; those rows are also why unrated recruits
            looked better than 3-stars (unrated records exist only for signees,
            see CLAUDE.md). Train 2021-2022, test 2023.
  transfer  2021-2024 portal entries. Train 2021-2023, test 2024.

Feature sets (all known at signing; class year is never a feature because the
test year lies outside the training range):
  baseline       market price + position: rating, unrated flag, position.
                 Transfers also get stars and a rating_missing flag: the portal
                 gives stars every season but a rating for only ~60% of rated
                 entries (ratings sit inside their star band), so stars are part
                 of the price.
  baseline_plus  + level of the acquiring school (a P4 offer is a market signal).
                 Beating baseline_plus out of time is the main test.
  full           + everything else known at signing
"""

import numpy as np
import pandas as pd

from cfb.processed import PROCESSED_DIR

TARGET = "par_window"
SPLITS = {"recruit": ((2021, 2022), 2023), "transfer": ((2021, 2023), 2024)}

# Recruiting hotbeds stand alone; other states fall into census regions.
HOTBEDS = {"TX", "FL", "CA", "GA"}
REGIONS = {
    "Northeast": {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"},
    "Midwest": {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"},
    "South": {"DE", "DC", "MD", "VA", "WV", "NC", "SC", "AL", "KY", "MS", "TN", "AR", "LA",
              "OK"},
    "West": {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "HI", "OR", "WA"},
}
LEVEL_ORDER = {"other": 0, "FCS": 1, "FBS": 2}

CATEGORICAL = ["position", "level", "home_area", "eligibility", "origin_level", "move",
               "qb_style"]

BASELINE = {"recruit": ["rating", "unrated", "position"],
            "transfer": ["rating", "stars", "rating_missing", "unrated", "position"]}
FEATURE_SETS = {
    "recruit": {
        "baseline": BASELINE["recruit"],
        "baseline_plus": BASELINE["recruit"] + ["level"],
        "full": BASELINE["recruit"] + ["level", "height", "weight", "home_area", "unrated_fcs"],
    },
    "transfer": {
        "baseline": BASELINE["transfer"],
        "baseline_plus": BASELINE["transfer"] + ["level"],
        "full": BASELINE["transfer"] + ["level", "origin_level", "move", "eligibility", "hs_rating",
                            "hs_unrated", "hs_missing", "years_since_hs", "qb_style",
                            "height", "weight", "prior_par", "prior_plays", "prior_best_tier",
                            "prior_seasons_available", "prior_missing"],
    },
}


def home_area(states: pd.Series) -> pd.Series:
    def area(state):
        if pd.isna(state):
            return pd.NA
        if state in HOTBEDS:
            return state
        return next((name for name, members in REGIONS.items() if state in members), "other")
    return states.map(area).astype("string")


def tier(market_tier: pd.Series) -> pd.Series:
    """Star tier for the within-tier metric, with 4 and 5 stars merged."""
    return market_tier.replace({"4": "4-5", "5": "4-5"}).astype("string")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model columns derived from master-table columns (all signing-time)."""
    out = df.copy()
    out["rating"] = out.market_rating.astype(float)
    out["unrated"] = out.market_unrated.astype(bool)
    out["stars"] = pd.to_numeric(out.market_stars, errors="coerce").astype(float)
    out["rating_missing"] = out.rating.isna() & ~out.unrated
    out["position"] = out.position_group.astype("string")
    out["level"] = out.acquiring_level.astype("string")
    out["unrated_fcs"] = out.unrated & out.level.eq("FCS")
    out["home_area"] = home_area(out.home_state)
    out["hs_missing"] = out.recruit_id.isna()
    out["hs_unrated"] = ~out.hs_missing & out.hs_stars.isna()
    out["prior_missing"] = out.prior_par.isna()
    origin = out.origin_level.map(LEVEL_ORDER).astype(float).to_numpy()
    dest = out.level.map(LEVEL_ORDER).astype(float).to_numpy()
    out["move"] = pd.Series(np.select([np.isnan(origin) | np.isnan(dest), dest > origin,
                                       dest < origin],
                                      ["unknown", "up", "down"], "same"),
                            index=out.index, dtype="string")
    out["tier"] = tier(out.market_tier)
    for col in ["height", "weight", "hs_rating", "years_since_hs", "prior_par", "prior_plays",
                "prior_best_tier", "prior_seasons_available"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in ["eligibility", "qb_style", "origin_level"]:
        out[col] = out[col].astype("string")
    return out


def population(master: pd.DataFrame, kind: str) -> pd.DataFrame:
    (first, last_train), test_year = SPLITS[kind]
    df = master[master.acquisition_type.eq(kind) & master.window_complete.astype(bool)
                & master.acq_season.between(first, test_year)]
    if kind == "recruit":
        df = df[df.acquiring_team.notna()]
    return add_features(df).reset_index(drop=True)


def split(df: pd.DataFrame, kind: str):
    """(train, test) by acquisition season."""
    (first, last_train), test_year = SPLITS[kind]
    train = df[df.acq_season.between(first, last_train)].reset_index(drop=True)
    test = df[df.acq_season.eq(test_year)].reset_index(drop=True)
    return train, test


def test_variants(train: pd.DataFrame, test: pd.DataFrame, kind: str) -> dict:
    """Test sets to report. Transfers: all rows; only players not seen in
    training (a tree could recognize a player's fixed traits); only matched
    entries (unmatched transfers are scored 0 but ~30% may have played)."""
    if kind == "recruit":
        return {"all": test}
    seen = test.person_id.isin(set(train.person_id))
    return {"all": test, "new_players": test[~seen], "matched_only": test[~test.unmatched]}


def load_master(path=None) -> pd.DataFrame:
    return pd.read_parquet(path or PROCESSED_DIR / "master.parquet")
