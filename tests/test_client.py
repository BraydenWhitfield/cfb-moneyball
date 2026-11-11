from cfb.client import CFBDClient


class FakeResponse:
    def __init__(self, data):
        self._data = data
        self.headers = {"X-CallLimit-Remaining": "998"}

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
    assert client.calls_remaining == 998  # read from the response header


def test_session_retries_transient_errors_only():
    from cfb.client import _build_session

    adapter = _build_session().get_adapter("https://api.collegefootballdata.com")
    retry = adapter.max_retries

    assert retry.total == 3
    assert retry.allowed_methods == ["GET"]
    assert retry.respect_retry_after_header
    # Rate limits and server errors are retried...
    assert {429, 500, 502, 503, 504} <= set(retry.status_forcelist)
    # ...but auth / not-found errors are not (retrying won't fix them).
    assert not {401, 403, 404} & set(retry.status_forcelist)

    # Backoff before each retry: 0s, 2s, 4s (urllib3 2.x).
    waits = []
    for _ in range(3):
        retry = retry.increment(method="GET", url="/x")
        waits.append(retry.get_backoff_time())
    assert waits == [0, 2, 4]
