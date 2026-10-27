"""Read past-performance tables (戦績) into race-card past runs, by column headings.

Racing sites lay out a horse's record as a table whose headings are standard Japanese
terms (日付, レース名, 頭数, 着順, 騎手, 距離, 馬場, タイム, 着差, 通過, 上り, ...). This module
maps those headings, not page markup, so it works on any site that uses them and on HTML
the user saves from a browser. Parsing the numbers in code avoids transcription errors.
"""

from __future__ import annotations

import io
import re

import pandas as pd
from bs4 import BeautifulSoup

from aikeiba.racecard import GOING_ALIASES, SURFACE_ALIASES

COURSES = ("札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉")
# heading -> field; the first matching heading wins
HEADINGS = {
    "date": ("日付", "年月日"),
    "venue": ("開催", "競馬場", "場"),
    "race_name": ("レース名", "レース"),
    "field_size": ("頭数",),
    "finish": ("着順", "着"),
    "popularity": ("人気", "人"),
    "jockey": ("騎手",),
    "weight_carried": ("斤量",),
    "distance": ("距離", "コース"),
    "going": ("馬場", "馬場状態"),
    "time": ("タイム", "走破タイム"),
    "margin": ("着差",),
    "passing": ("通過", "通過順", "コーナー通過順"),
    "last3f": ("上り", "上がり", "上3F", "上がり3F"),
}
_GOING_SHORT = {"良": "good", "稍": "yielding", "重": "soft", "不": "heavy"}
_GRADE = re.compile(r"[（(]\s*(G|Ｇ|J・G)\s*(III|II|I|Ⅲ|Ⅱ|Ⅰ|１|２|３|1|2|3)\s*[)）]")
_GRADE_NUM = {"I": "G1", "Ⅰ": "G1", "1": "G1", "１": "G1", "II": "G2", "Ⅱ": "G2", "2": "G2",
              "２": "G2", "III": "G3", "Ⅲ": "G3", "3": "G3", "３": "G3"}  # fmt: skip


def _column_map(columns) -> dict:
    names = [re.sub(r"\s+", "", str(c)) for c in columns]
    out = {}
    for field, headings in HEADINGS.items():
        for h in headings:
            if h in names and names.index(h) not in out.values():
                out[field] = names.index(h)
                break
    return out


def is_past_performance_table(df: pd.DataFrame) -> bool:
    cols = _column_map(df.columns)
    return "date" in cols and "finish" in cols


def parse_date(text) -> str | None:
    m = re.search(r"(\d{2,4})[/.\-年](\d{1,2})[/.\-月](\d{1,2})", str(text))
    if not m:
        return None
    year = int(m.group(1))
    year += 2000 if year < 100 else 0
    return f"{year:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def parse_time(text) -> float | None:
    m = re.fullmatch(r"\s*(?:(\d+):)?(\d{1,2})[.:](\d)\s*", str(text))
    if not m:
        return None
    minutes = int(m.group(1) or 0)
    return round(minutes * 60 + int(m.group(2)) + int(m.group(3)) / 10, 1)


def parse_distance(text) -> tuple[str | None, int | None]:
    m = re.search(r"(芝|ダート|ダ|障)?\s*(\d{3,4})", str(text))
    if not m:
        return None, None
    surface = SURFACE_ALIASES.get(m.group(1)) if m.group(1) else None
    return surface, int(m.group(2))


def parse_going(text) -> str | None:
    t = str(text).strip()
    if t in GOING_ALIASES:
        return GOING_ALIASES[t]
    return _GOING_SHORT.get(t[:1]) if t else None


def parse_grade(race_name: str) -> tuple[str, str | None]:
    """Race name without the grade mark, and the grade (G1/G2/G3) if marked."""
    m = _GRADE.search(race_name)
    if not m or m.group(1) == "J・G":  # jump races are not graded flat races
        return race_name.strip(), None
    return _GRADE.sub("", race_name).strip(), _GRADE_NUM[m.group(2)]


def _int(text) -> int | None:
    m = re.fullmatch(r"\s*(\d+)\s*", str(text))
    return int(m.group(1)) if m else None


def _float(text) -> float | None:
    try:
        return float(str(text).strip())
    except ValueError:
        return None


def parse_row(row: list, cols: dict, source: str | None = None) -> dict | None:
    """One past run, or None when the row is not a finished run (取消, 除外, 中止...)."""
    get = {f: row[i] for f, i in cols.items()}
    date = parse_date(get.get("date", ""))
    finish = _int(get.get("finish", ""))
    if date is None or finish is None:
        return None
    run: dict = {"date": date, "finish": finish}
    if "race_name" in get:
        name, grade = parse_grade(str(get["race_name"]))
        run["race_name"] = name
        if grade:
            run["grade"] = grade
    venue = str(get.get("venue", ""))
    course = next((c for c in COURSES if c in venue), None)
    if course:
        run["course"] = course
    surface, distance = parse_distance(get.get("distance", ""))
    if surface:
        run["surface"] = {"turf": "芝", "dirt": "ダート"}[surface]
    if distance:
        run["distance"] = distance
    going = parse_going(get.get("going", ""))
    if going:
        run["going"] = {"good": "良", "yielding": "稍重", "soft": "重", "heavy": "不良"}[going]
    for field in ("field_size", "popularity"):
        value = _int(get.get(field, ""))
        if value:
            run[field] = value
    jockey = str(get.get("jockey", "")).strip()
    if jockey and jockey != "nan":
        run["jockey"] = jockey
    for field in ("weight_carried", "last3f"):
        value = _float(get.get(field, ""))
        if value:
            run[field] = value
    time = parse_time(get.get("time", ""))
    if time:
        run["time"] = time
    margin = _float(get.get("margin", ""))
    if margin is not None:
        run["margin"] = max(margin, 0.0)  # winners show the gap to second as negative
    passing = re.match(r"\s*(\d+)", str(get.get("passing", "")))
    if passing:
        run["early_position"] = int(passing.group(1))
    if source:
        run["source"] = source
    return run


def past_runs_from_tables(tables: list[pd.DataFrame], source: str | None = None) -> list[dict]:
    runs = []
    for df in tables:
        if not is_past_performance_table(df):
            continue
        cols = _column_map(df.columns)
        for row in df.astype(str).itertuples(index=False):
            run = parse_row(list(row), cols, source)
            if run:
                runs.append(run)
    runs.sort(key=lambda r: r["date"], reverse=True)
    return runs


def past_runs_from_html(html: str, source: str | None = None) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    tables = []
    for t in soup.find_all("table"):
        try:
            tables += pd.read_html(io.StringIO(str(t)))
        except ValueError:
            continue
    return past_runs_from_tables(tables, source)
