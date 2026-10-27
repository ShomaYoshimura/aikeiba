import copy
import json

import pytest

from aikeiba.audit import audit, race_key
from aikeiba.cardtools import main as card_main
from aikeiba.pastruns import parse_date, parse_grade, parse_time, past_runs_from_html

TABLE = """<html><body><table>
<tr><th>日付</th><th>開催</th><th>天気</th><th>R</th><th>レース名</th><th>頭数</th><th>枠番</th>
<th>馬番</th><th>オッズ</th><th>人気</th><th>着順</th><th>騎手</th><th>斤量</th><th>距離</th>
<th>馬場</th><th>タイム</th><th>着差</th><th>通過</th><th>ペース</th><th>上り</th></tr>
<tr><td>2026/09/21</td><td>4阪神5</td><td>晴</td><td>11</td><td>神戸新聞杯(GII)</td><td>11</td>
<td>4</td><td>4</td><td>1.5</td><td>1</td><td>1</td><td>松山弘平</td><td>57</td><td>芝2400</td>
<td>良</td><td>2:25.9</td><td>-0.0</td><td>5-5-4-3</td><td>36.1-34.0</td><td>33.5</td></tr>
<tr><td>2026/05/31</td><td>2東京12</td><td>晴</td><td>11</td><td>東京優駿(GI)</td><td>18</td>
<td>3</td><td>5</td><td>3.2</td><td>2</td><td>1</td><td>松山弘平</td><td>57</td><td>芝2400</td>
<td>稍</td><td>2:23.8</td><td>-0.2</td><td>3-3-3-2</td><td>35.5-35.0</td><td>34.2</td></tr>
<tr><td>2026/04/05</td><td>3阪神4</td><td>雨</td><td>11</td><td>毎日杯(GIII)</td><td>12</td>
<td>1</td><td>1</td><td>2.0</td><td>1</td><td>取</td><td>松山弘平</td><td>56</td><td>芝1800</td>
<td>重</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>2025/12/27</td><td>5中山9</td><td>晴</td><td>11</td><td>ホープフルS(GI)</td><td>16</td>
<td>7</td><td>13</td><td>6.1</td><td>3</td><td>1</td><td>松山弘平</td><td>56</td><td>芝2000</td>
<td>良</td><td>2:00.5</td><td>-0.1</td><td>8-8-7-5</td><td>36.0-35.2</td><td>34.9</td></tr>
</table><table><tr><th>名前</th><th>値</th></tr><tr><td>父</td><td>X</td></tr></table>
</body></html>"""


def test_helpers():
    assert parse_date("2026/09/21") == "2026-09-21"
    assert parse_date("26.05.31") == "2026-05-31"
    assert parse_time("2:25.9") == 145.9
    assert parse_time("59.8") == 59.8
    assert parse_grade("東京優駿(GI)") == ("東京優駿", "G1")
    assert parse_grade("神戸新聞杯（ＧⅡ）") == ("神戸新聞杯", "G2")
    assert parse_grade("中山大障害(J・GI)") == ("中山大障害(J・GI)", None)


def test_past_runs_from_html_table():
    runs = past_runs_from_html(TABLE, source="https://example/horse")
    assert [r["date"] for r in runs] == ["2026-09-21", "2026-05-31", "2025-12-27"]  # 取消 skipped
    kobe = runs[0]
    assert kobe == {
        "date": "2026-09-21",
        "finish": 1,
        "race_name": "神戸新聞杯",
        "grade": "G2",
        "course": "阪神",
        "surface": "芝",
        "distance": 2400,
        "going": "良",
        "field_size": 11,
        "popularity": 1,
        "jockey": "松山弘平",
        "weight_carried": 57.0,
        "last3f": 33.5,
        "time": 145.9,
        "margin": 0.0,
        "early_position": 5,
        "source": "https://example/horse",
    }
    assert runs[1]["going"] == "稍重" and runs[1]["course"] == "東京"


def card():
    def run(date, name, finish, **kw):
        return {"date": date, "race_name": name, "finish": finish, **kw}

    return {
        "race": {
            "name": "t",
            "date": "2026-10-25",
            "course": "京都",
            "surface": "芝",
            "distance": 3000,
            "going": "良",
            "grade": "G1",
        },
        "runners": [
            {
                "number": 1,
                "horse": "A",
                "past_runs": [
                    run("2026-09-21", "神戸新聞杯", 1, field_size=11, going="良"),
                    run("2026-05-31", "日本ダービー", 3, field_size=18),
                ],
            },
            {
                "number": 2,
                "horse": "B",
                "past_runs": [
                    run("2026-09-21", "神戸新聞杯", 4),
                    run("2026-05-31", "東京優駿", 1, going="良"),
                ],
            },
            {
                "number": 3,
                "horse": "C",
                "past_runs": [
                    run("2026-05-03", "3歳未勝利", 1, course="京都", distance=2400),
                    run("2026-06-01", "スプリングステークス", 2),
                ],
            },
            {
                "number": 4,
                "horse": "D",
                "past_runs": [run("2026-05-03", "3歳未勝利", 1, course="東京", distance=2400)],
            },
        ],
    }


def test_audit_fills_agreed_race_facts():
    c = card()
    report = audit(c, fix=True)
    b = c["runners"][1]["past_runs"]
    assert b[0]["field_size"] == 11 and b[0]["going"] == "良"
    assert c["runners"][0]["past_runs"][1]["going"] == "良"  # ダービー = 東京優駿
    assert c["runners"][2]["past_runs"][0]["date"] == "2026-06-01"  # re-sorted
    assert any("sorted" in f for f in report.filled)
    assert report.ok  # different-course maiden races are not mixed up


def test_audit_finds_errors():
    c = card()
    c["runners"][1]["past_runs"][0]["finish"] = 1  # two winners of 神戸新聞杯
    c["runners"][1]["past_runs"][0]["going"] = "重"  # disagrees with A
    c["runners"][0]["past_runs"][0]["margin"] = 0.3  # winner with a margin
    c["runners"][3]["past_runs"].append({"date": "2026-10-25", "finish": 1})  # leak
    report = audit(copy.deepcopy(c))
    text = "\n".join(report.errors + report.warnings)
    assert "disagree on going" in text and "all finished 1" in text
    assert "winner with margin" in text and "leak" in text
    assert not report.ok


def test_race_key_normalization():
    assert race_key({"date": "d", "race_name": "スプリングステークス"}) == ("d", "スプリングS")
    assert race_key({"date": "d", "race_name": "日本ダービー"}) == ("d", "東京優駿")


def test_cli_runs_from_html_and_audit(tmp_path):
    html = tmp_path / "horse.html"
    html.write_text(TABLE, "utf-8")
    out = tmp_path / "frag.json"
    card_main(
        [
            "runs-from-html",
            str(html),
            "--number",
            "1",
            "--horse",
            "A",
            "--before",
            "2026-09-01",
            "-o",
            str(out),
        ]
    )
    frag = json.loads(out.read_text("utf-8"))
    assert [r["date"] for r in frag["runners"][0]["past_runs"]] == ["2026-05-31", "2025-12-27"]
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card(), ensure_ascii=False), "utf-8")
    card_main(["audit", str(path), "--fix"])
    fixed = json.loads(path.read_text("utf-8"))
    assert fixed["runners"][1]["past_runs"][0]["field_size"] == 11
    bad = card()
    bad["runners"][0]["past_runs"][0]["date"] = "2026-11-01"
    path.write_text(json.dumps(bad, ensure_ascii=False), "utf-8")
    with pytest.raises(SystemExit):
        card_main(["audit", str(path)])
