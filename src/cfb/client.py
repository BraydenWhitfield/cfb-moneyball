"""Thin client for the CollegeFootballData.com (CFBD) API.

Every successful response is cached as raw JSON under data/raw/, so repeated
requests cost no API calls (the free tier allows 1,000 calls per month).
"""

import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.collegefootballdata.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "raw"


def _build_session() -> requests.Session:
    """A requests session that retries transient failures with backoff."""
    retry = Retry(
        total=3,
        backoff_factor=1,  # waits ~1s, 2s, 4s between attempts
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,  # let raise_for_status() report the final error
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class CFBDClient:
    def __init__(self, api_key=None, cache_dir=DEFAULT_CACHE_DIR, session=None,
                 base_url=BASE_URL, timeout=30):
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("CFBD_API_KEY")
        if not api_key:
            raise RuntimeError(
                "CFBD_API_KEY not found. Copy .env.example to .env and add your key."
            )
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.session = session if session is not None else _build_session()
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        self.api_calls = 0  # real network calls made by this client
        self._printed_headers = False

    def cache_path(self, endpoint: str, params: dict) -> Path:
        """Map an endpoint + params to a file, e.g.
        /recruiting/players?year=2021 -> data/raw/recruiting/players/year=2021.json
        Params are sorted so the same request always maps to the same file.
        """
        parts = [p for p in endpoint.strip("/").split("/") if p]
        if params:
            name = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
            name = re.sub(r"[^A-Za-z0-9=_.-]", "-", name)  # safe on Windows
        else:
            name = "all"
        return self.cache_dir.joinpath(*parts, f"{name}.json")

    def get(self, endpoint: str, refresh: bool = False, **params):
        """GET an endpoint, using the cached JSON when available."""
        params = {k: v for k, v in params.items() if v is not None}
        path = self.cache_path(endpoint, params)
        if path.exists() and not refresh:
            return json.loads(path.read_text(encoding="utf-8"))

        response = self.session.get(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        self.api_calls += 1
        if not self._printed_headers:
            # One-time look at what the API reports (e.g. remaining-call quota).
            print(f"CFBD response headers: {dict(response.headers)}")
            self._printed_headers = True
        response.raise_for_status()
        data = response.json()

        # Write to a temp file, then rename, so a crash never leaves half a file.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
        return data
