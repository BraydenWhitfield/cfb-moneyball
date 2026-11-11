from cfb.client import CFBDClient


class FakeResponse:
    def __init__(self, data):
        self._data = data
        self.headers = {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class FakeSession:
    """Stands in for requests.Session: records calls, never touches the network."""

    def __init__(self, data):
        self.data = data
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.data)


def test_get_caches_response_and_sends_auth(tmp_path):
    payload = [{"name": "Test Player", "stars": 4}]
    session = FakeSession(payload)
    client = CFBDClient(api_key="fake-key", cache_dir=tmp_path, session=session)

    # First call goes to the "network" with the right URL, params, and auth.
    assert client.get("/recruiting/players", year=2021) == payload
    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url == "https://api.collegefootballdata.com/recruiting/players"
    assert kwargs["params"] == {"year": 2021}
    assert kwargs["headers"]["Authorization"] == "Bearer fake-key"

    # Response was cached as raw JSON under the temp dir.
    assert (tmp_path / "recruiting" / "players" / "year=2021.json").exists()

    # Second identical call is served from cache: no new network call.
    assert client.get("/recruiting/players", year=2021) == payload
    assert len(session.calls) == 1
    assert client.api_calls == 1
