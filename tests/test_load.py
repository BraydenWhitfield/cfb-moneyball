import json

import pytest

from cfb.load import load_draft, load_raw


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def test_load_raw_concatenates_years_and_flattens_nested_fields(tmp_path):
    write(tmp_path / "ppa/players/season/year=2021.json",
          [{"id": "1", "season": 2021, "totalPPA": {"all": 1.5}}])
    write(tmp_path / "ppa/players/season/year=2022.json",
          [{"id": "1", "season": 2022, "totalPPA": {"all": 2.0}}])
    df = load_raw("ppa/players/season", raw_dir=tmp_path)
    assert df.season.tolist() == [2021, 2022]
    assert df["totalPPA.all"].tolist() == [1.5, 2.0]


def test_load_draft_casts_int_ids_to_strings_matching_other_sources(tmp_path):
    write(tmp_path / "draft/picks/year=2025.json",
          [{"collegeAthleteId": 4688380, "name": "A"}, {"collegeAthleteId": None, "name": "B"}])
    ids = load_draft(raw_dir=tmp_path).collegeAthleteId
    assert ids.iloc[0] == "4688380"   # not 4688380 or "4688380.0"
    assert ids.isna().iloc[1]


def test_load_raw_errors_clearly_when_cache_is_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="No cached files"):
        load_raw("player/portal", raw_dir=tmp_path)
