from cfb.client import CFBDClient
from cfb.fetch import build_requests, plan, run


class FakeResponse:
    def __init__(self, remaining):
        self.headers = {"X-CallLimit-Remaining": str(remaining)}

    def raise_for_status(self):
        pass

    def json(self):
        return [{"id": 1}]


class FakeSession:
    """Records calls; reports a fixed monthly quota remaining."""

    def __init__(self, remaining=900):
        self.remaining = remaining
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.remaining)


def make_client(tmp_path, session):
    return CFBDClient(api_key="fake-key", cache_dir=tmp_path, session=session)


def test_build_requests_counts():
    # 9 recruit classes + 5 portal + 5 PPA + 10 usage + 5 stats + 6 draft
    # + 6 coaches + 1 teams
    assert len(build_requests()) == 47
    # One year: one request per dataset (usage twice), all with year=2025.
    one_year = build_requests(years=[2025])
    assert len(one_year) == 9
    assert all(params["year"] == 2025 for _, _, params in one_year)


def test_plan_skips_cached_without_api_calls(tmp_path):
    session = FakeSession()
    client = make_client(tmp_path, session)
    cached_file = client.cache_path("/recruiting/players", {"year": 2025})
    cached_file.parent.mkdir(parents=True)
    cached_file.write_text("[]")

    cached, todo = plan(client, build_requests(years=[2025]))

    assert [name for name, _, _ in cached] == ["recruits"]
    assert len(todo) == 8
    assert session.calls == []


def test_run_fetches_each_uncached_request_once(tmp_path):
    session = FakeSession()
    client = make_client(tmp_path, session)
    _, todo = plan(client, build_requests(years=[2025]))

    assert run(client, todo) == []
    assert len(session.calls) == 9
    # Usage with and without garbage time land in separate cache files.
    _, todo_after = plan(client, build_requests(years=[2025]))
    assert todo_after == []


def test_run_stops_below_quota_floor(tmp_path):
    session = FakeSession(remaining=50)
    client = make_client(tmp_path, session)
    _, todo = plan(client, build_requests(years=[2025]))

    run(client, todo, min_remaining=100)

    # First call is made (quota unknown), then the guard sees 50 < 100 and stops.
    assert len(session.calls) == 1
