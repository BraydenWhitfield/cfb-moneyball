import pandas as pd
import pytest

from cfb.handcheck import (allocate, closest_candidates, draw_sample, main, stratified_rate,
                           summarize, unmatched_population, wilson)


def players_df():
    base = {"in_recruits": True, "recruit_match_method": "unmatched", "position_group": "WR",
            "year": 2022, "committedTo": "Texas Tech", "market_tier": "3",
            "ambiguous": False, "conflict": False}
    rows = [{**base, "recruit_id": f"r{i}", "person_id": f"rec:r{i}", "name": f"Player {i}"}
            for i in range(30)]
    rows += [
        {**base, "recruit_id": "m", "person_id": "1", "name": "Matched",
         "recruit_match_method": "athlete_id"},                          # matched: excluded
        {**base, "recruit_id": "ol", "person_id": "rec:ol", "name": "Lineman",
         "position_group": "OL"},                                        # out of scope
        {**base, "recruit_id": "y", "person_id": "rec:y", "name": "Too New", "year": 2025},
        {**base, "recruit_id": "s", "person_id": "rec:s", "name": "Star Guy", "market_tier": "5"},
    ]
    return pd.DataFrame(rows)


def links_df():
    base = {"match_method": "unmatched", "destination": "Towson", "withdrawn": False,
            "went_pro": False, "position_group": "RB", "season": 2023, "origin": "Delaware",
            "market_tier": "unrated", "ambiguous": False, "firstName": "Portal"}
    rows = [{**base, "portal_key": f"p{i}", "person_id": f"portal:p{i}", "lastName": str(i)}
            for i in range(10)]
    rows += [{**base, "portal_key": "w", "person_id": "portal:w", "lastName": "W",
              "withdrawn": True},                                         # withdrew: excluded
             {**base, "portal_key": "nd", "person_id": "portal:nd", "lastName": "ND",
              "destination": None}]                                       # no destination
    return pd.DataFrame(rows)


def stat_players_df():
    return pd.DataFrame({
        "player_id": ["10", "11"], "stat_season": [2023, 2024], "team": ["Texas Tech"] * 2,
        "player_name": ["Cameron Smith", "Zed Jones"], "name_key": ["cameron smith", "zed jones"],
        "stat_position": ["WR", "WR"], "stat_position_group": ["WR", "WR"]})


def test_population_applies_scope_and_strata():
    pop = unmatched_population(players_df(), links_df())
    assert set(pop[pop.source == "recruit"].key) == {f"r{i}" for i in range(30)} | {"s"}
    assert set(pop[pop.source == "portal"].key) == {f"p{i}" for i in range(10)}
    assert set(pop.stratum) == {"recruit|WR|3", "recruit|WR|4-5", "portal|RB|2/unrated"}


def test_allocate_spreads_evenly_and_respects_stratum_size():
    take = allocate(pd.Series({"a": 100, "b": 3, "c": 50}), 12)
    assert take.to_dict() == {"a": 5, "b": 3, "c": 4}
    assert allocate(pd.Series({"a": 2}), 10).to_dict() == {"a": 2}   # can't exceed population


def test_draw_sample_is_reproducible_and_records_weights():
    pop = unmatched_population(players_df(), links_df())
    a = draw_sample(pop, stat_players_df(), n=10, seed=1)
    b = draw_sample(pop, stat_players_df(), n=10, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert (a.source == "recruit").sum() == 5 and (a.source == "portal").sum() == 5
    assert a.set_index("stratum").stratum_population.to_dict()["recruit|WR|3"] == 30
    assert (a.verdict == "").all() and a.sample_id.is_unique


def test_closest_candidates_ranks_by_name_similarity():
    out = closest_candidates("Cam Smith", ["Texas Tech"], stat_players_df())
    assert out[0].startswith("Cameron Smith | Texas Tech 2023-2023 WR | id 10")
    assert "no stat lines" in closest_candidates("X", ["Garden City C.C."], stat_players_df())[0]


def test_wilson_interval_known_values():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.237, abs=1e-3) and hi == pytest.approx(0.763, abs=1e-3)
    lo, hi = wilson(0, 10)
    assert lo == 0 and hi == pytest.approx(0.278, abs=1e-3)


def test_stratified_rate_weights_by_population():
    # Stratum A (900 people): 0/10 failed. Stratum B (100 people): 10/10 failed.
    labeled = pd.DataFrame({
        "stratum": ["A"] * 10 + ["B"] * 10,
        "stratum_population": [900] * 10 + [100] * 10,
        "failed": [False] * 10 + [True] * 10,
    })
    p, lo, hi = stratified_rate(labeled)
    assert p == pytest.approx(0.10)       # not the unweighted 0.50
    assert lo < 0.10 < hi


def test_summarize_reads_verdicts_and_rejects_typos(tmp_path):
    path = tmp_path / "hc.csv"
    rows = pd.DataFrame({
        "sample_id": range(1, 5), "source": ["recruit"] * 4, "position_group": ["WR"] * 4,
        "stratum": ["recruit|WR|3"] * 4, "stratum_population": [40] * 4,
        "verdict": ["bust", "Failed_Join ", "unsure", ""]})
    rows.to_csv(path, index=False, encoding="utf-8-sig")
    text = summarize(path)
    assert "3 labeled (1 unsure" in text and "failed-join rate 50%" in text

    rows.loc[3, "verdict"] = "bust?"
    rows.to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Unknown verdicts"):
        summarize(path)


def test_sample_command_refuses_to_overwrite_labels(tmp_path):
    path = tmp_path / "hc.csv"
    path.write_text("my precious labels", encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        main(["sample", "--out", str(path)])
    assert path.read_text(encoding="utf-8") == "my precious labels"


def test_summarize_drops_excluded_records_from_sample_and_population(tmp_path):
    path = tmp_path / "hc.csv"
    pd.DataFrame({
        "sample_id": [1, 2, 3], "source": ["recruit"] * 3, "key": ["216935", "5", "6"],
        "position_group": ["RB"] * 3, "stratum": ["recruit|RB|4-5"] * 3,
        "stratum_population": [10] * 3, "verdict": ["bust", "bust", "failed_join"],
    }).to_csv(path, index=False, encoding="utf-8-sig")
    text = summarize(path)
    assert "2 rows, 2 labeled" in text and "1 excluded record(s) dropped" in text
    assert "1/2 failed joins" in text
