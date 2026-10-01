"""Polite fetching of public racing pages for the collector agents (``aikeiba-fetch``).

The agents read pages through this tool instead of a summarizing web fetcher, so they see
every row of a table. The tool is deliberately conservative:

- It only fetches from sites listed in ``SITES`` whose terms the user has recorded as
  reviewed (``aikeiba-fetch --review-terms <site>``, stored in
  ``data/scraping/terms_reviewed.json``). Collectors must not record this themselves.
- It obeys robots.txt, waits at least ``min_interval`` seconds between requests to the same
  host (across processes), identifies itself, and never sends credentials.
- Pages are cached on disk (``data/cache/http``), so a page is fetched once per ``max_age``.
- It extracts readable text and tables (as Markdown) from the HTML.

There is no site-specific parsing: the agents interpret the extracted text and tables.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
import urllib.robotparser
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

USER_AGENT = "aikeiba-research/0.1 (personal race analysis; polite crawler)"
DATA_DIR = Path(os.environ.get("AIKEIBA_DATA_DIR", "data"))


@dataclass(frozen=True)
class Site:
    key: str
    domains: tuple[str, ...]
    uses: str  # what the collectors use it for


SITES = {
    s.key: s
    for s in (
        Site(
            "netkeiba",
            ("www.netkeiba.com", "race.netkeiba.com", "db.netkeiba.com", "news.netkeiba.com"),
            "entries, odds, horse past performances (time, margin, passing order, popularity, "
            "last 3F), jockey/trainer/sire records, course data",
        ),
        Site(
            "keibalab",
            ("www.keibalab.jp",),
            "entries, past-edition trends, course data (gate, running style, sire), odds",
        ),
        Site(
            "keibabook",
            ("p.keibabook.co.jp",),
            "workouts and stable comments, free pages only",
        ),
        Site("umanity", ("umanity.jp",), "public predictions and race data"),
        Site("uma-x", ("uma-x.jp",), "public (AI) predictions and figures"),
        Site("note", ("note.com",), "free prediction articles, for the consensus share only"),
        Site("jra", ("www.jra.go.jp",), "official entries, results, going and cushion values"),
    )
}


class FetchRefused(Exception):
    """Raised when a URL may not be fetched (unlisted site, terms not reviewed, robots.txt)."""


def site_for(url: str) -> Site:
    host = urlparse(url).hostname or ""
    for site in SITES.values():
        if host in site.domains:
            return site
    raise FetchRefused(f"{host} is not a listed source; add it to SITES first")


def _terms_file() -> Path:
    return DATA_DIR / "scraping" / "terms_reviewed.json"


def reviewed_sites() -> dict:
    path = _terms_file()
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def record_terms_review(key: str, note: str = "") -> None:
    if key not in SITES:
        raise ValueError(f"unknown site {key!r}; one of {sorted(SITES)}")
    data = reviewed_sites()
    data[key] = {"reviewed_on": datetime.now(UTC).date().isoformat(), "note": note}
    path = _terms_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


class Fetcher:
    def __init__(
        self,
        cache_dir: Path | None = None,
        min_interval: float = 5.0,
        timeout: float = 20.0,
        session: requests.Session | None = None,
        require_review: bool = True,
    ):
        self.cache_dir = cache_dir or DATA_DIR / "cache" / "http"
        self.min_interval = min_interval
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.require_review = require_review
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    # --- politeness -------------------------------------------------------------------
    def _wait_turn(self, host: str) -> None:
        stamp = self.cache_dir / "_last_request" / host
        stamp.parent.mkdir(parents=True, exist_ok=True)
        if stamp.exists():
            wait = self.min_interval - (time.time() - stamp.stat().st_mtime)
            if wait > 0:
                time.sleep(wait)
        stamp.touch()

    def allowed_by_robots(self, url: str) -> bool:
        parts = urlparse(url)
        base = f"{parts.scheme}://{parts.netloc}"
        parser = self._robots.get(base)
        if parser is None:
            parser = urllib.robotparser.RobotFileParser()
            self._wait_turn(parts.netloc)
            resp = self.session.get(f"{base}/robots.txt", timeout=self.timeout)
            if resp.status_code >= 400:
                parser.allow_all = resp.status_code < 500  # no robots.txt: allowed
                parser.disallow_all = resp.status_code >= 500
            else:
                parser.parse(resp.text.splitlines())
            self._robots[base] = parser
        return parser.can_fetch(USER_AGENT, url)

    def check(self, url: str) -> Site:
        site = site_for(url)
        if self.require_review and site.key not in reviewed_sites():
            raise FetchRefused(
                f"the terms of {site.key} have not been recorded as reviewed; the user must "
                f"run `aikeiba-fetch --review-terms {site.key}` after reading them"
            )
        if not self.allowed_by_robots(url):
            raise FetchRefused(f"robots.txt of {urlparse(url).netloc} disallows {url}")
        return site

    # --- fetching ---------------------------------------------------------------------
    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache_dir / f"{key}.html", self.cache_dir / f"{key}.json"

    def get(self, url: str, max_age_hours: float = 24.0) -> tuple[str, dict]:
        """HTML text and metadata (url, fetched_at, status, from_cache)."""
        html_path, meta_path = self._cache_paths(url)
        if html_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text("utf-8"))
            age = time.time() - datetime.fromisoformat(meta["fetched_at"]).timestamp()
            if age < max_age_hours * 3600:
                return html_path.read_text("utf-8"), {**meta, "from_cache": True}
        self.check(url)
        self._wait_turn(urlparse(url).netloc)
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        html = decode(resp.content, resp.headers.get("content-type", ""))
        meta = {
            "url": url,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": resp.status_code,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, "utf-8")
        meta_path.write_text(json.dumps(meta), "utf-8")
        return html, {**meta, "from_cache": False}


_CHARSET = re.compile(rb"""charset=["']?([\w-]+)""", re.I)


def decode(content: bytes, content_type: str = "") -> str:
    """Decode HTML using the header or meta charset (many racing sites use EUC-JP)."""
    m = re.search(r"charset=([\w-]+)", content_type, re.I) or _CHARSET.search(content[:4096])
    candidates = []
    if m:
        name = m.group(1)
        candidates.append(name.decode() if isinstance(name, bytes) else name)
    candidates += ["utf-8", "euc_jp", "cp932"]
    for enc in candidates:
        try:
            return content.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return content.decode("utf-8", errors="replace")


def extract(html: str) -> dict:
    """Title, readable text and tables (as Markdown) from a page."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    tables = []
    for t in soup.find_all("table"):
        try:
            frames = pd.read_html(io.StringIO(str(t)))
        except ValueError:
            continue
        for df in frames:
            if df.size:
                tables.append(df.fillna("").astype(str).to_markdown(index=False))
    for t in soup.find_all("table"):
        t.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    title = soup.title.get_text(strip=True) if soup.title else ""
    return {"title": title, "text": text, "tables": tables}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", nargs="?")
    parser.add_argument("--format", choices=("all", "text", "tables", "html"), default="all")
    parser.add_argument("--max-age", type=float, default=24.0, help="cache lifetime in hours")
    parser.add_argument("--review-terms", metavar="SITE", help="record that you read the terms")
    parser.add_argument("--note", default="", help="note stored with --review-terms")
    parser.add_argument("--sites", action="store_true", help="list sources and review status")
    args = parser.parse_args(argv)

    if args.review_terms:
        record_terms_review(args.review_terms, args.note)
        print(f"recorded: terms of {args.review_terms} reviewed")
        return
    if args.sites or not args.url:
        reviewed = reviewed_sites()
        for site in SITES.values():
            mark = reviewed.get(site.key, {}).get("reviewed_on", "not reviewed")
            print(f"{site.key:<10} {mark:<13} {', '.join(site.domains)}\n{'':<24} {site.uses}")
        return

    try:
        html, meta = Fetcher().get(args.url, args.max_age)
    except FetchRefused as e:
        raise SystemExit(f"refused: {e}") from e
    print(f"<!-- {json.dumps(meta, ensure_ascii=False)} -->")
    if args.format == "html":
        print(html)
        return
    page = extract(html)
    print(f"# {page['title']}\n")
    if args.format in ("all", "text"):
        print(page["text"], "\n")
    if args.format in ("all", "tables"):
        for i, table in enumerate(page["tables"], 1):
            print(f"## table {i}\n\n{table}\n")


if __name__ == "__main__":
    main()
