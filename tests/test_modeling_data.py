import pandas as pd

from cfb.modeling.data import (FEATURE_SETS, add_features, home_area, population, split,
                               tier)
from cfb.modeling.data import test_variants as variants_of


def master_row(aid, kind, season, **kw):
    row = {"acquisition_id": aid, "acquisition_type": kind, "acq_season": season,
           "window_complete": True, "acquiring_team": "Iowa", "acquiring_level": "FBS",
           "origin_level": "FCS" if kind == "transfer" else pd.NA, "person_id": aid,
           "recruit_id": "r" + aid, "market_rating": 0.85, "market_stars": 3.0,
           "market_unrated": False, "market_tier": "3", "position_group": "WR",
           "home_state": "OH", "hs_stars": 3.0, "hs_rating": 0.85, "years_since_hs": 2,
           "prior_par": 5.0, "prior_plays": 50.0, "prior_best_tier": 1.0,
           "prior_seasons_available": 1.0, "eligibility": "Immediate", "qb_style": pd.NA,
           "height": 72.0, "weight": 190.0, "unmatched": False, "par_window": 1.0}
    row.update(kw)
    return row


def test_no_class_year_or_outcome_columns_in_any_feature_set():
    banned = {"acq_season", "hs_class_year", "prior_seasons_available", "par_window",
              "tier_window", "unmatched", "match_method", "drafted_after"}
    for sets in FEATURE_SETS.values():
        for cols in sets.values():
            assert not banned & set(cols)
    for kind, sets in FEATURE_SETS.items():   # nested: baseline in plus in full
        assert set(sets["baseline"]) <= set(sets["baseline_plus"]) <= set(sets["full"])


def test_recruit_population_needs_a_school_and_a_complete_window():
    m = pd.DataFrame([master_row("a", "recruit", 2021),
                      master_row("b", "recruit", 2022, acquiring_team=None),
                      master_row("c", "recruit", 2023, window_complete=False),
                      master_row("d", "recruit", 2024),
                      master_row("e", "transfer", 2022)])
    assert population(m, "recruit").acquisition_id.tolist() == ["a"]


def test_split_and_transfer_test_variants():
    m = pd.DataFrame([master_row("a", "transfer", 2022, person_id="p1"),
                      master_row("b", "transfer", 2024, person_id="p1"),     # seen in train
                      master_row("c", "transfer", 2024, person_id="p2", unmatched=True),
                      master_row("d", "transfer", 2024, person_id="p3")])
    train, test = split(population(m, "transfer"), "transfer")
    assert train.acquisition_id.tolist() == ["a"] and len(test) == 3
    v = variants_of(train, test, "transfer")
    assert v["new_players"].acquisition_id.tolist() == ["c", "d"]
    assert v["matched_only"].acquisition_id.tolist() == ["b", "d"]


def test_features_flags_and_moves():
    m = pd.DataFrame([master_row("a", "transfer", 2022),                     # FCS -> FBS
                      master_row("b", "transfer", 2022, market_rating=None),  # stars only
                      master_row("c", "transfer", 2022, market_rating=None, market_stars=None,
                                 market_unrated=True, market_tier="unrated",
                                 acquiring_level="FCS", recruit_id=None, hs_stars=None,
                                 prior_par=None)])
    f = add_features(m).set_index("acquisition_id")
    assert f.move.tolist() == ["up", "up", "same"]
    assert f.rating_missing.tolist() == [False, True, False]
    assert f.loc["c", "unrated_fcs"] and f.loc["c", "hs_missing"]
    assert not f.loc["c", "prior_missing"]   # only 2021 transfers are flagged
    assert not f.loc["c", "hs_unrated"]


def test_home_area_and_tier():
    assert home_area(pd.Series(["TX", "OH", "ON", None])).tolist()[:3] == ["TX", "Midwest",
                                                                          "other"]
    assert tier(pd.Series(["4", "5", "3", "unrated"])).tolist() == ["4-5", "4-5", "3", "unrated"]


def test_unmatched_transfers_get_no_linked_history_not_a_missing_flag():
    m = pd.DataFrame([master_row("a", "transfer", 2023, unmatched=True, prior_par=None,
                                 prior_plays=None, prior_best_tier=None),
                      master_row("b", "transfer", 2021, prior_par=None, prior_plays=None,
                                 prior_best_tier=None)])
    f = add_features(m).set_index("acquisition_id")
    assert f.loc["a", "prior_par"] == 0 and not f.loc["a", "prior_missing"]
    assert pd.isna(f.loc["b", "prior_par"]) and f.loc["b", "prior_missing"]


def test_model_populations_exclude_tight_ends():
    m = pd.DataFrame([master_row("a", "recruit", 2021),
                      master_row("b", "recruit", 2021, position_group="TE")])
    assert population(m, "recruit").acquisition_id.tolist() == ["a"]
