"""Fetch raw CFBD data into data/raw/, skipping anything already cached.

Usage:
    python -m cfb.fetch                      # dry run: show what would be fetched
    python -m cfb.fetch --years 2025         # dry run for one year
    python -m cfb.fetch --run --years 2025   # fetch one year per endpoint
    python -m cfb.fetch --run                # fetch everything not yet cached
"""

import argparse

import requests

from cfb.client import CFBDClient

# (name, endpoint, years, extra params). One request per dataset per year;
# each returns all teams and positions, so filtering happens later, offline.
DATASETS = [
    # 2017-2020 classes = original market price for most early transfers.
    ("recruits", "/recruiting/players", range(2017, 2026), {}),
    ("portal", "/player/portal", range(2021, 2026), {}),
    ("ppa_season", "/ppa/players/season", range(2021, 2026), {}),
    ("usage", "/player/usage", range(2021, 2026), {}),
    ("usage_no_garbage", "/player/usage", range(2021, 2026),
     {"excludeGarbageTime": "true"}),
    ("stats_season", "/stats/player/season", range(2021, 2026), {}),
    # 2026 draft has already happened and includes many portal-era players.
    ("draft", "/draft/picks", range(2021, 2027), {}),
]


def build_requests(years=None):
    """List every (name, endpoint, params) request, optionally limited to some years."""
    reqs = []
    for name, endpoint, dataset_years, extra in DATASETS:
        for year in dataset_years:
            if years is None or year in years:
                reqs.append((name, endpoint, {"year": year, **extra}))
    return reqs


def plan(client, reqs):
    """Split requests into (already cached, still to fetch) without any API calls."""
    cached, todo = [], []
    for req in reqs:
        _, endpoint, params = req
        is_cached = client.cache_path(endpoint, params).exists()
        (cached if is_cached else todo).append(req)
    return cached, todo


def run(client, todo, min_remaining=100):
    """Fetch each uncached request. Returns the requests that failed.

    Stops early if the monthly quota drops below min_remaining. One failed
    request doesn't stop the run; re-running picks up only what's missing.
    """
    failed = []
    for name, endpoint, params in todo:
        if client.calls_remaining is not None and client.calls_remaining < min_remaining:
            print(f"Stopping: {client.calls_remaining} calls left "
                  f"(floor is {min_remaining}).")
            break
        try:
            data = client.get(endpoint, **params)
        except requests.RequestException as exc:
            print(f"  FAILED  {name:<17} {params}: {exc}")
            failed.append((name, endpoint, params))
            continue
        rows = len(data) if isinstance(data, list) else "not a list"
        warn = "  <-- empty, check this" if rows == 0 else ""
        print(f"  fetched {name:<17} {params}  rows={rows}{warn}")
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true",
                        help="actually call the API (default is a dry run)")
    parser.add_argument("--years", type=int, nargs="+",
                        help="only these years, e.g. --years 2025")
    parser.add_argument("--min-remaining", type=int, default=100,
                        help="stop if monthly calls remaining drops below this")
    args = parser.parse_args(argv)

    client = CFBDClient()
    cached, todo = plan(client, build_requests(args.years))
    print(f"{len(cached)} cached (skipped), {len(todo)} to fetch "
          f"= {len(todo)} API calls (plus any retries).")
    for name, _, params in todo:
        print(f"  to fetch: {name:<17} {params}")

    if not args.run:
        print("Dry run: no API calls made. Add --run to fetch.")
        return

    failed = run(client, todo, args.min_remaining)
    print(f"Done. API calls made: {client.api_calls}. "
          f"Calls remaining this month: {client.calls_remaining}. "
          f"Failed: {len(failed)}.")


if __name__ == "__main__":
    main()
