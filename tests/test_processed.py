import pandas as pd

from cfb.processed import load_players, load_portal_links


def _write(tmp_path):
    pd.DataFrame({
        "person_id": ["1", "2", "rec:r3", "draft:2025-1"], "player_id": ["1", "2", None, None],
        "recruit_id": ["r1", "r2", "r3", None],   # last: a draft-only person, no recruit
        "recruit_match_method": ["athlete_id", "nationwide_name", "unmatched", None],
    }, dtype="string").to_parquet(tmp_path / "players.parquet")
    pd.DataFrame({
        "portal_key": ["k1", "k2", "k3"],
        "person_id": ["1", "2", "2"],
        "player_id": ["1", "2", None],       # k3's identity came only via recruit r2
        "recruit_id": ["r1", "r2", "r2"],
    }).to_parquet(tmp_path / "portal_links.parquet")


def test_load_players_drops_nationwide_unless_asked(tmp_path):
    _write(tmp_path)
    assert load_players(tmp_path).person_id.tolist() == ["1", "rec:r3", "draft:2025-1"]
    assert len(load_players(tmp_path, include_nationwide=True)) == 4


def test_load_portal_links_detaches_nationwide_recruits(tmp_path):
    _write(tmp_path)
    links = load_portal_links(tmp_path).set_index("portal_key")
    assert links.nationwide_dependent.tolist() == [False, True, True]
    assert links.loc["k1", "recruit_id"] == "r1" and links.loc["k1", "person_id"] == "1"
    assert pd.isna(links.loc["k2", "recruit_id"]) and links.loc["k2", "person_id"] == "2"
    assert pd.isna(links.loc["k3", "recruit_id"]) and links.loc["k3", "person_id"] == "portal:k3"

    full = load_portal_links(tmp_path, include_nationwide=True).set_index("portal_key")
    assert full.loc["k3", "recruit_id"] == "r2" and full.loc["k3", "person_id"] == "2"
