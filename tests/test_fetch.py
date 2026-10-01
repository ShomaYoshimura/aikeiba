import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aikeiba import fetch
from aikeiba.fetch import Fetcher, FetchRefused, Site, decode, extract

PAGE = """<html><head><meta charset="EUC-JP"><title>出馬表</title></head><body>
<h1>天皇賞（秋）</h1><script>var x=1;</script>
<table><tr><th>馬番</th><th>馬名</th></tr><tr><td>1</td><td>テストホース</td></tr></table>
</body></html>"""


@pytest.fixture
def site(tmp_path, monkeypatch):
    root = tmp_path / "www"
    root.mkdir()
    (root / "robots.txt").write_text("User-agent: *\nDisallow: /private/\n")
    (root / "race.html").write_bytes(PAGE.encode("euc_jp"))
    (root / "private").mkdir()
    (root / "private" / "x.html").write_text("secret")
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(fetch, "DATA_DIR", tmp_path / "data")
    monkeypatch.setitem(fetch.SITES, "local", Site("local", ("127.0.0.1",), "test"))
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_refuses_unlisted_and_unreviewed_sites(site, tmp_path):
    f = Fetcher(cache_dir=tmp_path / "cache", min_interval=0)
    with pytest.raises(FetchRefused, match="not a listed source"):
        f.get("https://example.com/page")
    with pytest.raises(FetchRefused, match="not been recorded as reviewed"):
        f.get(f"{site}/race.html")


def test_fetches_decodes_caches_and_obeys_robots(site, tmp_path):
    fetch.record_terms_review("local", "test")
    f = Fetcher(cache_dir=tmp_path / "cache", min_interval=0)
    html, meta = f.get(f"{site}/race.html")
    assert "天皇賞（秋）" in html and not meta["from_cache"]
    _, meta2 = f.get(f"{site}/race.html")
    assert meta2["from_cache"]
    with pytest.raises(FetchRefused, match="robots.txt"):
        f.get(f"{site}/private/x.html")


def test_rate_limit_between_requests(site, tmp_path):
    import time

    fetch.record_terms_review("local")
    f = Fetcher(cache_dir=tmp_path / "cache", min_interval=0.3)
    start = time.time()
    f.get(f"{site}/race.html", max_age_hours=0)
    f.get(f"{site}/race.html", max_age_hours=0)
    assert time.time() - start >= 0.6  # robots.txt + two pages, 0.3 s apart


def test_extract_text_and_tables():
    page = extract(decode(PAGE.encode("euc_jp")))
    assert page["title"] == "出馬表"
    assert "天皇賞（秋）" in page["text"] and "var x" not in page["text"]
    assert "テストホース" in page["tables"][0]


def test_unknown_site_review_is_rejected():
    with pytest.raises(ValueError, match="unknown site"):
        fetch.record_terms_review("nope")
