from __future__ import annotations

import base64
import html
import json
import os
import re
from datetime import date as date_type
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from st_aggrid import AgGrid, GridOptionsBuilder

try:
    from treningshistorikk_app.concept2_api import (
        Concept2ApiConfig,
        Concept2ApiError,
        fetch_all_results,
        fetch_result,
        fetch_strokes,
    )
except ModuleNotFoundError:
    from concept2_api import (
        Concept2ApiConfig,
        Concept2ApiError,
        fetch_all_results,
        fetch_result,
        fetch_strokes,
    )


MONTH_NO = {
    1: "Januar",
    2: "Februar",
    3: "Mars",
    4: "April",
    5: "Mai",
    6: "Juni",
    7: "Juli",
    8: "August",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}

_STANDARD_SESSION_TAGS = [
    "test",
    "rolig",
    "moderat",
    "terskel",
    "hard",
    "sprint",
    "langkjøring",
    "konkurranse",
    "intervaller",
    "teknikk",
    "styrke",
    "restitusjon",
]


def _find_data_dir(start: Path) -> Path:
    if start.is_file():
        start = start.parent

    # Preferred: system/Data next to system/treningshistorikk_app
    for candidate_root in (start, *start.parents):
        data_dir = candidate_root / "Data"
        if data_dir.is_dir():
            return data_dir

    # Fallback: repo-root/system/Data
    for candidate_root in (start, *start.parents):
        sys_data = candidate_root / "system" / "Data"
        if sys_data.is_dir():
            return sys_data

    raise FileNotFoundError("Fant ikke CSV-data. Forventet 'system/Data/' (eller 'Data/').")


def _season_from_filename(path: Path) -> str:
    match = re.search(r"season-(\d{4})", path.name)
    return match.group(1) if match else path.stem


def _training_season_label(value: object) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    start_year = ts.year if ts.month >= 5 else ts.year - 1
    end_year = start_year + 1
    return f"{str(start_year)[-2:]}/{str(end_year)[-2:]}"


def _training_season_start(value: object) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    start_year = ts.year if ts.month >= 5 else ts.year - 1
    return pd.Timestamp(year=int(start_year), month=5, day=1)


def _training_season_day(value: object) -> float:
    ts = pd.to_datetime(value, errors="coerce")
    season_start = _training_season_start(ts)
    if pd.isna(ts) or season_start is None:
        return float("nan")
    return float((ts.normalize() - season_start).days)


def _season_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    match = re.search(r"(\d{2,4})", text)
    if match:
        return int(match.group(1)), text
    return -1, text


def _benchmark_meta(row: pd.Series) -> tuple[str | None, str | None, float]:
    workout_format = str(row.get("workout_format") or "")
    desc = str(row.get("description") or "")
    distance_m = pd.to_numeric(pd.Series([row.get("work_distance_m")]), errors="coerce").iloc[0]
    work_time_s = pd.to_numeric(pd.Series([row.get("work_time_s")]), errors="coerce").iloc[0]
    pace_s_500 = pd.to_numeric(pd.Series([row.get("pace_s_500")]), errors="coerce").iloc[0]

    if workout_format == "Intervall":
        if re.search(r"10\s*[xX×]\s*500", desc):
            return "10 × 500 m", "lower", float(pace_s_500)
        if pd.notna(distance_m) and 4900 <= float(distance_m) <= 5100 and pd.notna(pace_s_500):
            return "10 × 500 m", "lower", float(pace_s_500)
        return None, None, float("nan")

    if pd.notna(distance_m) and 4990 <= float(distance_m) <= 5010 and pd.notna(work_time_s):
        return "5 000 m", "lower", float(work_time_s)
    if pd.notna(distance_m) and 9990 <= float(distance_m) <= 10010 and pd.notna(work_time_s):
        return "10 000 m", "lower", float(work_time_s)
    if pd.notna(work_time_s) and 3590 <= float(work_time_s) <= 3610 and pd.notna(distance_m):
        return "1 time", "higher", float(distance_m)
    return None, None, float("nan")


def _annotate_benchmark_status(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["benchmark_name"] = None
        out["benchmark_metric_direction"] = None
        out["benchmark_metric_value"] = np.nan
        out["is_pb"] = False
        out["is_season_best"] = False
        out["benchmark_status"] = ""
        return out

    out = df.copy()
    benchmark_meta = out.apply(_benchmark_meta, axis=1, result_type="expand")
    benchmark_meta.columns = ["benchmark_name", "benchmark_metric_direction", "benchmark_metric_value"]
    out[["benchmark_name", "benchmark_metric_direction", "benchmark_metric_value"]] = benchmark_meta
    out["is_pb"] = False
    out["is_season_best"] = False

    benchmark_rows = out[
        out["benchmark_name"].notna()
        & out["benchmark_metric_direction"].notna()
        & pd.to_numeric(out["benchmark_metric_value"], errors="coerce").notna()
    ]

    for (_, benchmark_name), idx in benchmark_rows.groupby(["machine_type", "benchmark_name"]).groups.items():
        metric = pd.to_numeric(out.loc[idx, "benchmark_metric_value"], errors="coerce")
        direction = out.loc[idx[0], "benchmark_metric_direction"]
        if direction == "lower":
            best_value = float(metric.min())
        else:
            best_value = float(metric.max())
        mask = np.isclose(metric.to_numpy(dtype=float), best_value, rtol=1e-9, atol=1e-9)
        out.loc[list(metric.index[mask]), "is_pb"] = True

    for (_, benchmark_name, season_name), idx in benchmark_rows.groupby(["machine_type", "benchmark_name", "season"]).groups.items():
        metric = pd.to_numeric(out.loc[idx, "benchmark_metric_value"], errors="coerce")
        direction = out.loc[idx[0], "benchmark_metric_direction"]
        if direction == "lower":
            best_value = float(metric.min())
        else:
            best_value = float(metric.max())
        mask = np.isclose(metric.to_numpy(dtype=float), best_value, rtol=1e-9, atol=1e-9)
        out.loc[list(metric.index[mask]), "is_season_best"] = True

    out["benchmark_status"] = np.select(
        [
            out["is_pb"] & out["is_season_best"],
            out["is_pb"],
            out["is_season_best"],
        ],
        [
            "PB • Sesongbeste",
            "PB",
            "Sesongbeste",
        ],
        default="",
    )

    return out


def _benchmark_trend_specs(benchmark_name: str) -> list[dict[str, str]]:
    specs = [
        {"label": "Watt", "col": "avg_watts", "y_title": "W", "direction": "higher"},
        {"label": "W/kg", "col": "w_kg", "y_title": "W/kg", "direction": "higher"},
        {"label": "Puls", "col": "avg_hr", "y_title": "bpm", "direction": "lower"},
    ]
    if benchmark_name == "1 time":
        return [
            {"label": "Distanse", "col": "work_distance_m", "y_title": "meter", "direction": "higher"},
            *specs,
            {"label": "Pace", "col": "pace_s_500", "y_title": "sek/500m", "direction": "lower"},
        ]
    return [
        {"label": "Tid", "col": "work_time_s", "y_title": "sek", "direction": "lower"},
        {"label": "Pace", "col": "pace_s_500", "y_title": "sek/500m", "direction": "lower"},
        *specs,
    ]


def _all_session_trend_specs() -> list[dict[str, str]]:
    return [
        {"label": "Tid", "col": "work_time_s", "y_title": "sek", "direction": "lower"},
        {"label": "Distanse", "col": "work_distance_m", "y_title": "meter", "direction": "higher"},
        {"label": "Pace", "col": "pace_s_500", "y_title": "sek/500m", "direction": "lower"},
        {"label": "Watt", "col": "avg_watts", "y_title": "W", "direction": "higher"},
        {"label": "W/kg", "col": "w_kg", "y_title": "W/kg", "direction": "higher"},
        {"label": "Puls", "col": "avg_hr", "y_title": "bpm", "direction": "lower"},
    ]


def _session_load_score(avg_hr: object, work_time_s: object, hf_maks: int, zones: list[dict]) -> float:
    try:
        duration_min = float(work_time_s) / 60.0
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(duration_min) or duration_min <= 0:
        return float("nan")

    zone_weights = {
        "I1": 1.0,
        "I2": 1.25,
        "I3": 1.6,
        "I4": 2.0,
        "I5": 2.5,
    }
    try:
        hr_f = float(avg_hr)
    except (TypeError, ValueError):
        hr_f = float("nan")

    if np.isfinite(hr_f):
        for zone in zones:
            lo_bpm = hf_maks * zone["lo"] / 100.0
            hi_bpm = hf_maks * zone["hi"] / 100.0
            if lo_bpm <= hr_f <= hi_bpm:
                return round(duration_min * zone_weights.get(zone["name"], 1.0), 1)
        if hr_f > hf_maks * zones[-1]["hi"] / 100.0:
            return round(duration_min * 2.8, 1)
        return round(duration_min * 0.9, 1)

    return round(duration_min, 1)


def _normalize_response_to_target(
    response_value: object,
    response_direction: str,
    actual_target_value: object,
    target_reference_value: object,
) -> float:
    response_num = pd.to_numeric(pd.Series([response_value]), errors="coerce").iloc[0]
    actual_num = pd.to_numeric(pd.Series([actual_target_value]), errors="coerce").iloc[0]
    reference_num = pd.to_numeric(pd.Series([target_reference_value]), errors="coerce").iloc[0]
    if pd.isna(response_num) or pd.isna(actual_num) or pd.isna(reference_num):
        return float(response_num) if pd.notna(response_num) else float("nan")
    if float(actual_num) <= 0 or float(reference_num) <= 0:
        return float(response_num)

    ratio = float(actual_num) / float(reference_num)
    if response_direction == "lower":
        return float(response_num) * ratio
    if response_direction == "higher":
        return float(response_num) / ratio
    return float(response_num)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_pace_to_seconds(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return float("nan")

    parts = text.split(":")
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return float("nan")

    return float("nan")


def _format_pace(seconds: float) -> str:
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return ""
    total = float(seconds)
    minutes = int(total // 60)
    sec = total - minutes * 60
    return f"{minutes}:{sec:04.1f}"


def _format_duration(seconds: float) -> str:
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return ""
    td = pd.to_timedelta(float(seconds), unit="s")
    hh = int(td.total_seconds() // 3600)
    mm = int((td.total_seconds() % 3600) // 60)
    ss = td.total_seconds() % 60
    if hh > 0:
        return f"{hh}:{mm:02d}:{ss:04.1f}"
    return f"{mm}:{ss:04.1f}"


def _interval_protocol_time_label(seconds: float) -> str:
    rounded_seconds = int(round(float(seconds) / 5.0) * 5)
    if rounded_seconds % 60 == 0:
        return f"{rounded_seconds // 60}min"
    minutes, remaining_seconds = divmod(rounded_seconds, 60)
    return f"{minutes}:{remaining_seconds:02d}" if minutes else f"{remaining_seconds}s"


def _interval_protocol_pause_label(seconds: float) -> str:
    rounded_seconds = int(round(float(seconds) / 5.0) * 5)
    minutes, remaining_seconds = divmod(rounded_seconds, 60)
    return f"P:{minutes}:{remaining_seconds:02d}"


def _interval_protocol_label(row: pd.Series) -> str | None:
    """Return a stable, human-readable setup label for an interval session."""
    intervals = row.get("_workout_intervals")
    if not isinstance(intervals, list) or not intervals:
        return None

    work_times: list[float] = []
    work_distances: list[float] = []
    rest_times: list[float] = []
    for interval in intervals:
        if not isinstance(interval, dict):
            return None
        work_time = pd.to_numeric(interval.get("time"), errors="coerce")
        work_distance = pd.to_numeric(interval.get("distance"), errors="coerce")
        rest_time = pd.to_numeric(interval.get("rest_time"), errors="coerce")
        if pd.notna(work_time):
            work_times.append(float(work_time) / 10.0)
        if pd.notna(work_distance):
            work_distances.append(float(work_distance))
        if pd.notna(rest_time):
            rest_times.append(float(rest_time) / 10.0)

    interval_count = len(intervals)
    has_uniform_time = len(work_times) == interval_count and max(work_times) - min(work_times) <= 5
    has_uniform_distance = len(work_distances) == interval_count and max(work_distances) - min(work_distances) <= 10
    if has_uniform_time:
        work_label = _interval_protocol_time_label(float(np.median(work_times)))
    elif has_uniform_distance:
        work_label = f"{int(round(float(np.median(work_distances))))}m"
    elif len(work_times) == interval_count:
        work_label = "/".join(_interval_protocol_time_label(value) for value in work_times)
    else:
        work_label = "var"

    if len(rest_times) != interval_count:
        rest_label = " P:?"
    elif max(rest_times) - min(rest_times) <= 5:
        rest_seconds = float(np.median(rest_times))
        rest_label = "" if rest_seconds <= 2 else f" {_interval_protocol_pause_label(rest_seconds)}"
    else:
        rest_label = " P:var"

    return f"{interval_count}x{work_label}{rest_label}"


def _table_total_distance(value: object) -> str:
    distance_m = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(distance_m):
        return ""
    rounded_m = int(round(float(distance_m)))
    if rounded_m < 1000:
        return f"{rounded_m} m"
    distance_km = round(rounded_m / 1000.0, 1)
    return f"{distance_km:g} km"


def _share_card_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    font_dir = windir / "Fonts"
    candidates = [
        font_dir / ("arialbd.ttf" if bold else "arial.ttf"),
        font_dir / ("segoeuib.ttf" if bold else "segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _wrap_share_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _share_card_filename(row: pd.Series) -> str:
    desc = re.sub(r"[^A-Za-z0-9_-]+", "-", str(row.get("description") or "okt").strip()).strip("-")
    if not desc:
        desc = "okt"
    date_text = pd.to_datetime(row.get("date"), errors="coerce")
    prefix = date_text.strftime("%Y-%m-%d") if pd.notna(date_text) else "session"
    return f"strava-share-{prefix}-{desc[:40]}.png"


def _zone_hit_for_hr(avg_hr: object, hf_maks: int, zones: list[dict]) -> dict | None:
    hr_val = pd.to_numeric(pd.Series([avg_hr]), errors="coerce").iloc[0]
    if pd.isna(hr_val):
        return None
    hr_float = float(hr_val)
    return next(
        (
            zone for zone in zones
            if hf_maks * zone["lo"] / 100 <= hr_float <= hf_maks * zone["hi"] / 100
        ),
        None,
    )


def _share_card_badge_text(row: pd.Series) -> str:
    zones = _get_zones()
    hf_maks = _get_hf_maks()
    zone_hit = _zone_hit_for_hr(row.get("avg_hr"), hf_maks, zones)
    if zone_hit is not None:
        return f"Sone {zone_hit['name']} - {zone_hit['label']}"

    for candidate in [row.get("benchmark_status"), row.get("benchmark_name"), row.get("workout_format")]:
        if candidate is None or pd.isna(candidate):
            continue
        text = str(candidate).strip()
        if text and text.lower() != "nan":
            return text
    return "SkiErg"


def _resample_hr_points_per_second(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points

    ordered = sorted((float(x), float(y)) for x, y in points if np.isfinite(x) and np.isfinite(y))
    if not ordered:
        return []

    deduped: list[tuple[float, float]] = [ordered[0]]
    for x_val, y_val in ordered[1:]:
        if x_val <= deduped[-1][0]:
            deduped[-1] = (max(deduped[-1][0], x_val), y_val)
        else:
            deduped.append((x_val, y_val))

    if len(deduped) < 2:
        return deduped

    x_vals = np.array([pt[0] for pt in deduped], dtype=float)
    y_vals = np.array([pt[1] for pt in deduped], dtype=float)
    start_s = int(np.floor(x_vals[0]))
    end_s = int(np.ceil(x_vals[-1]))
    if end_s <= start_s:
        return deduped

    second_grid = np.arange(start_s, end_s + 1, dtype=float)
    interp_vals = np.interp(second_grid, x_vals, y_vals)
    return list(zip(second_grid.tolist(), interp_vals.tolist()))


def _stroke_hr_points_full_session(row: pd.Series) -> list[tuple[float, float]]:
    strokes = row.get("_strokes")
    if not isinstance(strokes, list) or not strokes:
        return []

    intervals = row.get("_workout_intervals")
    has_intervals = isinstance(intervals, list) and len(intervals) > 0

    groups: list[list[tuple[float, float]]] = []
    current_group: list[tuple[float, float]] = []
    prev_t_raw: float | None = None
    for stroke in strokes:
        if not isinstance(stroke, dict):
            continue
        t_raw = pd.to_numeric(stroke.get("t"), errors="coerce")
        hr_raw = pd.to_numeric(stroke.get("hr"), errors="coerce")
        if pd.isna(t_raw) or pd.isna(hr_raw):
            continue
        t_s = float(t_raw) / 10.0
        if prev_t_raw is not None and t_s < prev_t_raw:
            if current_group:
                groups.append(current_group)
            current_group = []
        current_group.append((t_s, float(hr_raw)))
        prev_t_raw = t_s
    if current_group:
        groups.append(current_group)

    if not groups:
        return []

    points: list[tuple[float, float]] = []
    if has_intervals and len(groups) <= len(intervals):
        elapsed_s = 0.0
        for idx, group in enumerate(groups):
            interval = intervals[idx] if idx < len(intervals) and isinstance(intervals[idx], dict) else {}
            for raw_t_s, hr_val in group:
                points.append((elapsed_s + raw_t_s, hr_val))

            group_end_s = group[-1][0]
            elapsed_s += group_end_s

            rest_t = pd.to_numeric(interval.get("rest_time"), errors="coerce")
            rest_hr = pd.to_numeric((interval.get("heart_rate") or {}).get("rest"), errors="coerce")
            if pd.notna(rest_t) and float(rest_t) > 0:
                elapsed_s += float(rest_t) / 10.0
                if pd.notna(rest_hr):
                    points.append((elapsed_s, float(rest_hr)))
                else:
                    points.append((elapsed_s, group[-1][1]))
        return _resample_hr_points_per_second(points)

    elapsed_offset = 0.0
    for group in groups:
        for raw_t_s, hr_val in group:
            points.append((elapsed_offset + raw_t_s, hr_val))
        elapsed_offset += group[-1][0]
    return _resample_hr_points_per_second(points)


def _share_card_hr_points(row: pd.Series) -> list[tuple[float, float]]:
    stroke_points = _stroke_hr_points_full_session(row)
    if len(stroke_points) >= 2:
        return stroke_points

    intervals = row.get("_workout_intervals")
    if isinstance(intervals, list) and intervals:
        points = []
        elapsed_s = 0.0
        for idx, interval in enumerate(intervals):
            if not isinstance(interval, dict):
                continue
            heart_rate = interval.get("heart_rate") or {}
            hr_avg = pd.to_numeric(heart_rate.get("average"), errors="coerce")
            work_t = pd.to_numeric(interval.get("time"), errors="coerce")
            rest_t = pd.to_numeric(interval.get("rest_time"), errors="coerce")
            rest_hr = pd.to_numeric(heart_rate.get("rest"), errors="coerce")

            if pd.notna(work_t) and float(work_t) > 0 and pd.notna(hr_avg):
                elapsed_s += float(work_t) / 10.0
                points.append((elapsed_s, float(hr_avg)))
            elif pd.notna(hr_avg):
                points.append((float(idx + 1), float(hr_avg)))

            if pd.notna(rest_t) and float(rest_t) > 0:
                elapsed_s += float(rest_t) / 10.0
                pause_hr = float(rest_hr) if pd.notna(rest_hr) else (float(hr_avg) if pd.notna(hr_avg) else None)
                if pause_hr is not None:
                    points.append((elapsed_s, pause_hr))

        if len(points) >= 2:
            return points

    avg_hr = pd.to_numeric(pd.Series([row.get("avg_hr")]), errors="coerce").iloc[0]
    work_time_s = pd.to_numeric(pd.Series([row.get("work_time_s")]), errors="coerce").iloc[0]
    if pd.notna(avg_hr):
        total_s = float(work_time_s) if pd.notna(work_time_s) and float(work_time_s) > 0 else 1.0
        return [(0.0, float(avg_hr)), (total_s, float(avg_hr))]
    return []


def _build_share_card_png(row: pd.Series) -> bytes:
    width, height = 1080, 1350
    image = Image.new("RGB", (width, height), "#f5efe6")
    draw = ImageDraw.Draw(image)

    bg = "#f5efe6"
    panel = "#fffaf3"
    accent = "#fc5200"
    muted = "#6c6259"
    text_dark = "#1f2328"
    border = "#eadfce"

    panel_box = (44, 44, width - 44, height - 44)
    draw.rounded_rectangle(panel_box, radius=42, fill=panel, outline=border, width=2)

    title_font = _share_card_font(40, bold=True)
    subtitle_font = _share_card_font(24)
    value_font = _share_card_font(46, bold=True)
    label_font = _share_card_font(24)
    small_font = _share_card_font(22)
    badge_font = _share_card_font(22, bold=True)
    graph_title_font = _share_card_font(28, bold=True)

    desc = str(row.get("description") or "SkiErg").strip()
    title_lines = _wrap_share_text(draw, desc, title_font, width - 170)
    title_lines = title_lines[:2]
    title_text = "\n".join(title_lines)
    draw.multiline_text((84, 92), title_text, font=title_font, fill=text_dark, spacing=4)

    date_val = pd.to_datetime(row.get("date"), errors="coerce")
    workout_format = str(row.get("workout_format") or "Økt")
    subtitle_parts = [workout_format]
    if pd.notna(date_val):
        subtitle_parts.append(date_val.strftime("%d.%m.%Y %H:%M"))
    subtitle = "  •  ".join(subtitle_parts)
    draw.text((84, 202), subtitle, font=subtitle_font, fill=muted)

    badge_text = _share_card_badge_text(row)
    badge_box = (84, 246, 84 + min(560, int(draw.textlength(badge_text, font=badge_font)) + 44), 294)
    draw.rounded_rectangle(badge_box, radius=16, fill=accent)
    draw.text((badge_box[0] + 20, badge_box[1] + 11), badge_text, font=badge_font, fill="#ffffff")

    weight_val = pd.to_numeric(pd.Series([row.get("vekt")]), errors="coerce").iloc[0]
    wkg_val = pd.to_numeric(pd.Series([row.get("w_kg")]), errors="coerce").iloc[0]
    if pd.isna(wkg_val):
        avg_watts = pd.to_numeric(pd.Series([row.get("avg_watts")]), errors="coerce").iloc[0]
        if pd.notna(avg_watts) and pd.notna(weight_val) and float(weight_val) > 0:
            wkg_val = float(avg_watts) / float(weight_val)

    dist_km = pd.to_numeric(pd.Series([row.get("work_distance_km")]), errors="coerce").iloc[0]
    metrics = [
        ("Distanse", f"{float(dist_km):.2f} km" if pd.notna(dist_km) else "—"),
        ("Tid", _format_duration(row.get("work_time_s")) or "—"),
        ("Pace", _format_pace(row.get("pace_s_500")) or "—"),
        ("Watt", f"{float(row.get('avg_watts')):.0f} W" if pd.notna(row.get("avg_watts")) else "—"),
        ("Puls", f"{float(row.get('avg_hr')):.0f} bpm" if pd.notna(row.get("avg_hr")) else "—"),
        ("W/kg", f"{float(wkg_val):.2f}" if pd.notna(wkg_val) else "—"),
    ]

    card_left = 84
    card_top = 350
    card_width = 452
    card_height = 152
    gap_x = 32
    gap_y = 24
    for idx, (label, value) in enumerate(metrics):
        row_idx = idx // 2
        col_idx = idx % 2
        x0 = card_left + col_idx * (card_width + gap_x)
        y0 = card_top + row_idx * (card_height + gap_y)
        x1 = x0 + card_width
        y1 = y0 + card_height
        draw.rounded_rectangle((x0, y0, x1, y1), radius=28, fill=bg, outline=border, width=2)
        draw.text((x0 + 24, y0 + 24), label.upper(), font=label_font, fill=muted)
        draw.text((x0 + 24, y0 + 66), value, font=value_font, fill=text_dark)

    graph_box = (84, 872, width - 84, 1146)
    draw.rounded_rectangle(graph_box, radius=30, fill=bg, outline=border, width=2)
    draw.text((graph_box[0] + 24, graph_box[1] + 18), "Pulskurve", font=graph_title_font, fill=text_dark)

    hr_points = _share_card_hr_points(row)
    plot_left = graph_box[0] + 26
    plot_top = graph_box[1] + 68
    plot_right = graph_box[2] - 26
    plot_bottom = graph_box[3] - 32
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#cdbba6", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#cdbba6", width=2)
    if hr_points:
        x_vals = [pt[0] for pt in hr_points]
        y_vals = [pt[1] for pt in hr_points]
        x_min = min(x_vals)
        x_max = max(x_vals)
        y_min = min(y_vals)
        y_max = max(y_vals)
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_min = max(40.0, y_min - 10.0)
            y_max = y_max + 10.0
        y_pad = max(5.0, (y_max - y_min) * 0.15)
        y_min -= y_pad
        y_max += y_pad
        path: list[tuple[float, float]] = []
        for x_val, y_val in hr_points:
            x_px = plot_left + ((x_val - x_min) / (x_max - x_min)) * (plot_right - plot_left)
            y_px = plot_bottom - ((y_val - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)
            path.append((x_px, y_px))
        if len(path) >= 2:
            draw.line(path, fill=accent, width=6, joint="curve")
        for bpm in [int(round(y_min)), int(round((y_min + y_max) / 2)), int(round(y_max))]:
            y_px = plot_bottom - ((float(bpm) - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)
            draw.line((plot_left, y_px, plot_right, y_px), fill="#e7d7c3", width=1)
            draw.text((plot_left + 8, y_px - 18), f"{bpm} bpm", font=small_font, fill=muted)
    else:
        draw.text((plot_left, plot_top + 40), "Ingen pulsdata tilgjengelig", font=small_font, fill=muted)

    intervals = row.get("_workout_intervals")
    interval_count = len(intervals) if isinstance(intervals, list) else 0
    tags = row.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    footer_lines = [
        f"Maskin: {row.get('machine_type') or 'SkiErg'}",
        f"Sesong: {row.get('season') or '—'}",
    ]
    if interval_count > 0:
        footer_lines.append(f"Intervaller: {interval_count}")
    if pd.notna(row.get("stroke_rate")):
        footer_lines.append(f"SPM: {float(row.get('stroke_rate')):.0f}")
    if pd.notna(row.get("drag_factor")):
        footer_lines.append(f"Drag: {float(row.get('drag_factor')):.0f}")
    if tags:
        footer_lines.append("Tagger: " + ", ".join(str(tag) for tag in tags[:4]))

    footer_y = 1188
    draw.text((72, footer_y), " • ".join(footer_lines[:3]), font=small_font, fill=muted)
    if len(footer_lines) > 3:
        draw.text((72, footer_y + 36), " • ".join(footer_lines[3:]), font=small_font, fill=muted)

    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _safe_range(series: pd.Series) -> tuple[float, float] | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.min()), float(s.max())


def _numeric_slider_options(min_value: float, max_value: float, step: float) -> list[float]:
    if step <= 0:
        return [float(min_value), float(max_value)]
    count = max(1, int(round((float(max_value) - float(min_value)) / float(step))))
    decimals = max(0, len(str(step).split(".")[-1]) if "." in str(step) else 0)
    return [
        round(float(min_value) + i * float(step), decimals)
        for i in range(count + 1)
    ]


def _watts_from_pace_s_500(pace_s_500: pd.Series) -> pd.Series:
    # Concept2 relationship (RowErg/SkiErg): W = 2.8 / (t/500)^3
    t = pd.to_numeric(pace_s_500, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 2.8 / np.power(t / 500.0, 3)
    w = pd.Series(w, index=pace_s_500.index)
    w[(t <= 0) | (~np.isfinite(w))] = np.nan
    return w


# ---------------------------------------------------------------------------
# User settings storage (Data/settings.json)
# ---------------------------------------------------------------------------

# Olympiatoppen pulse zones (% of HF maks)
_OPT_ZONES_DEFAULT: list[dict] = [
    {"name": "I1", "label": "Rolig utholdenhet",   "lo": 60, "hi": 72,  "color": "#5bc8f5"},
    {"name": "I2", "label": "Moderat utholdenhet", "lo": 72, "hi": 82,  "color": "#74c476"},
    {"name": "I3", "label": "Terskel",             "lo": 82, "hi": 87,  "color": "#fed976"},
    {"name": "I4", "label": "VO₂maks",             "lo": 87, "hi": 92,  "color": "#fd8d3c"},
    {"name": "I5", "label": "Fart / Sprint",       "lo": 92, "hi": 97,  "color": "#e31a1c"},
]


def _settings_path() -> Path:
    here = Path(__file__).resolve()
    data_dir = _find_data_dir(here)
    return data_dir / "settings.json"


def _load_settings() -> dict:
    p = _settings_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    p = _settings_path()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_settings(patch: dict) -> dict:
    settings = _load_settings()
    settings.update(patch)
    _save_settings(settings)
    return settings


_APP_NAME_DEFAULT = "Skiergportal"


def _get_app_name() -> str:
    """Return stored app/portal name, default Skiergportal."""
    name = _load_settings().get("app_name", _APP_NAME_DEFAULT)
    name = str(name).strip()
    return name if name else _APP_NAME_DEFAULT


def _get_hf_maks() -> int:
    """Return stored HF maks, default 200."""
    return int(_load_settings().get("hf_maks", 200))


def _get_zones() -> list[dict]:
    """Return stored zone boundaries (list of {name, label, lo, hi, color}), default Olympiatoppen."""
    stored = _load_settings().get("zones")
    if isinstance(stored, list) and len(stored) == len(_OPT_ZONES_DEFAULT):
        return stored
    return [z.copy() for z in _OPT_ZONES_DEFAULT]


def _delete_csv_file(path: Path) -> None:
    if path.exists():
        path.unlink()


@st.dialog("Slett CSV")
def _confirm_csv_file_deletion(path_text: str) -> None:
    csv_path = Path(path_text)
    st.write(f"Er du sikker på at du vil slette `{csv_path.name}`?")
    st.caption("Filen fjernes permanent fra system/Data.")
    cancel_col, delete_col = st.columns(2)
    if cancel_col.button("Avbryt", key=f"csv_cancel_{csv_path.name}", width="stretch"):
        st.rerun()
    if delete_col.button("Slett CSV", key=f"csv_confirm_{csv_path.name}", type="primary", width="stretch"):
        _delete_csv_file(csv_path)
        st.rerun()


# ---------------------------------------------------------------------------
# Per-session weight storage (Data/session_weights.json)
# ---------------------------------------------------------------------------

def _weights_path() -> Path:
    here = Path(__file__).resolve()
    data_dir = _find_data_dir(here)
    return data_dir / "session_weights.json"


def _session_weight_key(row: pd.Series) -> str:
    """Stable key for a session: prefer log_id, fall back to date+description."""
    lid = row.get("log_id")
    try:
        if lid is not None and pd.notna(lid):
            return f"id:{int(lid)}"
    except (TypeError, ValueError):
        pass
    date_val = row.get("date") or row.get("date_only")
    desc = str(row.get("description") or "")[:40]
    return f"date:{str(date_val)[:16]}|{desc}"


def _load_weights() -> dict[str, float]:
    p = _weights_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_weight(key: str, weight: float | None) -> None:
    p = _weights_path()
    data = _load_weights()
    if weight is None or weight <= 0:
        data.pop(key, None)
    else:
        data[key] = round(float(weight), 1)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _comments_path() -> Path:
    here = Path(__file__).resolve()
    data_dir = _find_data_dir(here)
    return data_dir / "session_comments.json"


def _load_comments() -> dict[str, str]:
    p = _comments_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_comment(key: str, comment: str) -> None:
    p = _comments_path()
    data = _load_comments()
    stripped = comment.strip()
    if stripped:
        data[key] = stripped
    else:
        data.pop(key, None)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _autosave_comment(key: str, widget_key: str) -> None:
    _save_comment(key, str(st.session_state.get(widget_key, "")))


def _tags_path() -> Path:
    here = Path(__file__).resolve()
    data_dir = _find_data_dir(here)
    return data_dir / "session_tags.json"


def _load_tags() -> dict[str, list[str]]:
    p = _tags_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, list[str]] = {}
    for key, values in raw.items():
        if isinstance(values, list):
            cleaned[key] = [str(v).strip() for v in values if str(v).strip()]
    return cleaned


def _save_tags(key: str, tags: list[str]) -> None:
    p = _tags_path()
    data = _load_tags()
    cleaned_tags = []
    seen: set[str] = set()
    for tag in tags:
        tag_clean = str(tag).strip()
        tag_norm = tag_clean.lower()
        if not tag_clean or tag_norm in seen:
            continue
        seen.add(tag_norm)
        cleaned_tags.append(tag_clean)
    if cleaned_tags:
        data[key] = cleaned_tags
    else:
        data.pop(key, None)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _autosave_tags(key: str, widget_key: str) -> None:
    value = st.session_state.get(widget_key, [])
    _save_tags(key, value if isinstance(value, list) else [])


def _configured_tag_catalog() -> list[str]:
    """Return active standard and user-defined tags for the tag library."""
    settings = _load_settings()
    configured = settings.get("tag_catalog", [])
    hidden = settings.get("hidden_tag_catalog", [])
    hidden_normalized = {str(tag).strip().casefold() for tag in hidden if str(tag).strip()}
    tags = [*_STANDARD_SESSION_TAGS, *(configured if isinstance(configured, list) else [])]
    unique: dict[str, str] = {}
    for tag in tags:
        cleaned = str(tag).strip()
        if cleaned and cleaned.casefold() not in hidden_normalized:
            unique.setdefault(cleaned.casefold(), cleaned)
    return sorted(unique.values(), key=str.casefold)


def _tag_catalog() -> list[str]:
    """Return the tag library together with tags already assigned to sessions."""
    assigned = [tag for tags in _load_tags().values() for tag in tags]
    tags = [*_configured_tag_catalog(), *assigned]
    unique: dict[str, str] = {}
    for tag in tags:
        cleaned = str(tag).strip()
        if cleaned:
            unique.setdefault(cleaned.casefold(), cleaned)
    return sorted(unique.values(), key=str.casefold)


def _remove_tag_from_sessions(tag_to_remove: str) -> int:
    """Remove one tag from every assigned session and return affected-session count."""
    normalized = tag_to_remove.strip().casefold()
    if not normalized:
        return 0
    data = _load_tags()
    affected_sessions = 0
    for key, session_tags in list(data.items()):
        retained = [tag for tag in session_tags if tag.casefold() != normalized]
        if len(retained) == len(session_tags):
            continue
        affected_sessions += 1
        if retained:
            data[key] = retained
        else:
            data.pop(key, None)
    if affected_sessions:
        _tags_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return affected_sessions


def _lactate_path() -> Path:
    here = Path(__file__).resolve()
    data_dir = _find_data_dir(here)
    return data_dir / "session_lactate.json"


def _normalize_lactate_value(value: object) -> float | None:
    try:
        lactate = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lactate) or lactate <= 0 or lactate > 30:
        return None
    return round(lactate, 1)


def _clean_interval_lactates(interval_lactates: object) -> dict[str, float]:
    if not isinstance(interval_lactates, dict):
        return {}
    cleaned: dict[str, float] = {}
    for idx, value in interval_lactates.items():
        lactate = _normalize_lactate_value(value)
        if lactate is None:
            continue
        cleaned[str(idx)] = lactate
    return cleaned


def _clean_threshold_test(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    try:
        interval_count = int(value.get("interval_count"))
        interval_duration_s = int(value.get("interval_duration_s"))
        rest_duration_s = int(value.get("rest_duration_s", 0))
    except (TypeError, ValueError):
        return None
    if not 1 <= interval_count <= 30 or not 30 <= interval_duration_s <= 3600 or not 0 <= rest_duration_s <= 1800:
        return None
    protocol_name = str(value.get("protocol_name") or "").strip()
    return {
        "protocol_name": protocol_name or f"{interval_count} x {interval_duration_s // 60} min terskeltest",
        "interval_count": interval_count,
        "interval_duration_s": interval_duration_s,
        "rest_duration_s": rest_duration_s,
    }


def _load_lactates() -> dict[str, dict[str, object]]:
    p = _lactate_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    cleaned: dict[str, dict[str, object]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        session_lactate = _normalize_lactate_value(value.get("session_lactate"))
        note = str(value.get("session_lactate_note") or "").strip()
        timing = str(value.get("session_lactate_measure_timing") or "ukjent").strip() or "ukjent"
        interval_lactates = _clean_interval_lactates(value.get("interval_lactates"))
        threshold_test = _clean_threshold_test(value.get("threshold_test"))
        if session_lactate is None and not note and not interval_lactates and threshold_test is None:
            continue
        entry: dict[str, object] = {
            "session_lactate_note": note,
            "session_lactate_measure_timing": timing,
            "interval_lactates": interval_lactates,
        }
        if session_lactate is not None:
            entry["session_lactate"] = session_lactate
        if threshold_test is not None:
            entry["threshold_test"] = threshold_test
        cleaned[str(key)] = entry
    return cleaned


def _save_lactate_entry(
    key: str,
    *,
    session_lactate: float | None,
    note: str,
    timing: str,
    interval_lactates: dict[str, float] | None,
    threshold_test: dict[str, object] | None = None,
) -> None:
    p = _lactate_path()
    data = _load_lactates()

    clean_session_lactate = _normalize_lactate_value(session_lactate)
    clean_note = note.strip()
    clean_timing = str(timing).strip() or "ukjent"
    clean_interval_lactates = _clean_interval_lactates(interval_lactates or {})
    clean_threshold_test = _clean_threshold_test(threshold_test)

    if clean_session_lactate is None and not clean_note and not clean_interval_lactates and clean_threshold_test is None:
        data.pop(key, None)
    else:
        payload: dict[str, object] = {
            "session_lactate_note": clean_note,
            "session_lactate_measure_timing": clean_timing,
            "interval_lactates": clean_interval_lactates,
        }
        if clean_session_lactate is not None:
            payload["session_lactate"] = clean_session_lactate
        if clean_threshold_test is not None:
            payload["threshold_test"] = clean_threshold_test
        data[key] = payload

    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _interval_lactate_values(interval_lactates: object) -> list[float]:
    if not isinstance(interval_lactates, dict):
        return []
    pairs: list[tuple[int, float]] = []
    for key, value in interval_lactates.items():
        lactate = _normalize_lactate_value(value)
        if lactate is None:
            continue
        try:
            idx = int(str(key))
        except (TypeError, ValueError):
            idx = 0
        pairs.append((idx, lactate))
    return [value for _, value in sorted(pairs, key=lambda x: x[0])]


def _lactate_timing_options(interval_count: int = 0) -> list[str]:
    options = ["ukjent", "etter økt", "etter siste drag"]
    if interval_count > 0:
        options.extend([f"etter drag {idx}" for idx in range(1, interval_count + 1)])
    return options


def _merge_lactate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["session_lactate"] = np.nan
        out["session_lactate_note"] = ""
        out["session_lactate_measure_timing"] = "ukjent"
        out["interval_lactates"] = [{} for _ in range(len(out))]
        out["interval_lactate_count"] = 0
        out["interval_lactate_avg"] = np.nan
        out["interval_lactate_min"] = np.nan
        out["interval_lactate_max"] = np.nan
        out["interval_lactate_last"] = np.nan
        out["interval_lactate_delta_last_first"] = np.nan
        out["threshold_test"] = [{} for _ in range(len(out))]
        out["is_threshold_test"] = False
        out["threshold_protocol_name"] = ""
        out["has_lactate"] = False
        return out

    lactates = _load_lactates()
    lactate_entries = out.apply(lambda row: lactates.get(_session_weight_key(row), {}), axis=1)

    out["session_lactate"] = pd.to_numeric(
        lactate_entries.map(lambda entry: entry.get("session_lactate") if isinstance(entry, dict) else None),
        errors="coerce",
    )
    out["session_lactate_note"] = lactate_entries.map(
        lambda entry: str(entry.get("session_lactate_note") or "") if isinstance(entry, dict) else ""
    )
    out["session_lactate_measure_timing"] = lactate_entries.map(
        lambda entry: str(entry.get("session_lactate_measure_timing") or "ukjent") if isinstance(entry, dict) else "ukjent"
    )
    out["interval_lactates"] = lactate_entries.map(
        lambda entry: dict(entry.get("interval_lactates") or {}) if isinstance(entry, dict) else {}
    )
    interval_values = out["interval_lactates"].map(_interval_lactate_values)
    out["interval_lactate_count"] = interval_values.map(len).astype(int)
    out["interval_lactate_avg"] = interval_values.map(lambda values: float(np.mean(values)) if values else np.nan)
    out["interval_lactate_min"] = interval_values.map(lambda values: float(min(values)) if values else np.nan)
    out["interval_lactate_max"] = interval_values.map(lambda values: float(max(values)) if values else np.nan)
    out["interval_lactate_last"] = interval_values.map(lambda values: values[-1] if values else np.nan)
    out["interval_lactate_delta_last_first"] = interval_values.map(
        lambda values: (values[-1] - values[0]) if len(values) >= 2 else np.nan
    )
    out["threshold_test"] = lactate_entries.map(
        lambda entry: dict(entry.get("threshold_test") or {}) if isinstance(entry, dict) else {}
    )
    out["is_threshold_test"] = out["threshold_test"].map(bool)
    out["threshold_protocol_name"] = out["threshold_test"].map(
        lambda test: str(test.get("protocol_name") or "") if isinstance(test, dict) else ""
    )
    out["has_lactate"] = out["interval_lactate_count"] > 0
    return out


def _format_lactate(value: object) -> str:
    try:
        lactate = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(lactate):
        return "—"
    return f"{lactate:.1f}"


def _table_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy with floating-point columns rounded to one decimal."""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].round(1)
    return display


def _interpolate_threshold_metrics(row: pd.Series, target_lactate: float) -> dict[str, float] | None:
    """Interpolate workload metrics where two adjacent samples bracket target lactate."""
    intervals = row.get("_workout_intervals")
    items = intervals if isinstance(intervals, list) and intervals else row.get("_workout_splits")
    lactates = row.get("interval_lactates")
    if not isinstance(items, list) or not isinstance(lactates, dict):
        return None

    samples: list[dict[str, float]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        lactate = _normalize_lactate_value(lactates.get(str(index)))
        time_tenths = item.get("time")
        distance_m = item.get("distance")
        if lactate is None or time_tenths is None or distance_m is None or float(distance_m) <= 0:
            continue
        time_s = float(time_tenths) / 10.0
        pace_s_500 = time_s / float(distance_m) * 500.0
        if pace_s_500 <= 0:
            continue
        heart_rate = (item.get("heart_rate") or {}).get("average")
        samples.append(
            {
                "lactate": lactate,
                "watts": 2.8 / (pace_s_500 / 500.0) ** 3,
                "pace_s_500": pace_s_500,
                "heart_rate": float(heart_rate) if heart_rate is not None else float("nan"),
            }
        )

    for lower, upper in zip(samples, samples[1:]):
        lower_lactate = lower["lactate"]
        upper_lactate = upper["lactate"]
        if not min(lower_lactate, upper_lactate) <= target_lactate <= max(lower_lactate, upper_lactate):
            continue
        if np.isclose(lower_lactate, upper_lactate):
            continue
        fraction = (target_lactate - lower_lactate) / (upper_lactate - lower_lactate)
        return {
            metric: lower[metric] + fraction * (upper[metric] - lower[metric])
            for metric in ("watts", "pace_s_500", "heart_rate")
        }
    return None


@st.cache_data(show_spinner=False)
def _load_weight_csv() -> pd.DataFrame:
    """Load vekter/weight.csv and return a clean DataFrame sorted by date descending.

    Returns columns: weight_date (datetime64), weight_kg (float).
    Returns empty DataFrame if file not found.
    """
    try:
        here = Path(__file__).resolve()
        data_dir = _find_data_dir(here)
        p = data_dir / "vekter" / "weight.csv"
        if not p.exists():
            return pd.DataFrame(columns=["weight_date", "weight_kg"])
        raw = pd.read_csv(p)
        raw.columns = raw.columns.str.strip()
        # Column is named 'Weight (kg)'
        wkg_col = next((c for c in raw.columns if "weight" in c.lower() and "kg" in c.lower() and "fat" not in c.lower() and "bone" not in c.lower() and "muscle" not in c.lower()), None)
        if wkg_col is None:
            return pd.DataFrame(columns=["weight_date", "weight_kg"])
        out = pd.DataFrame({
            "weight_date": pd.to_datetime(raw["Date"], errors="coerce"),
            "weight_kg": pd.to_numeric(raw[wkg_col], errors="coerce"),
        }).dropna()
        return out.sort_values("weight_date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["weight_date", "weight_kg"])


def _match_weight(session_date: pd.Timestamp, weight_df: pd.DataFrame, max_days: int = 14) -> tuple[float | None, str | None]:
    """Return (weight_kg, date_str) for the closest measurement within max_days, else (None, None)."""
    if weight_df.empty or pd.isna(session_date):
        return None, None
    dates = weight_df["weight_date"]
    diff = (dates - session_date).abs()
    idx = diff.idxmin()
    if diff[idx].days > max_days:
        return None, None
    w = float(weight_df.loc[idx, "weight_kg"])
    d = weight_df.loc[idx, "weight_date"].strftime("%d.%m.%Y")
    return w, d


# ---------------------------------------------------------------------------
# Persistent disk cache for API results (Data/api_cache_{type}.json)
# Avoids re-fetching the full history on every app restart.
# ---------------------------------------------------------------------------

def _api_cache_path(type_filter: str | None) -> Path:
    slug = (type_filter or "all").replace("/", "_")
    here = Path(__file__).resolve()
    data_dir = _find_data_dir(here)
    return data_dir / f"api_cache_{slug}.json"


def _load_api_disk_cache(type_filter: str | None) -> dict[str, dict]:
    p = _api_cache_path(type_filter)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_api_disk_cache(type_filter: str | None, cache: dict[str, dict]) -> None:
    p = _api_cache_path(type_filter)
    p.write_text(
        json.dumps(cache, ensure_ascii=False, default=str, separators=(",", ":")),
        encoding="utf-8",
    )


_CONCEPT2_REQUIRED_CSV_COLUMNS = {
    "Log ID",
    "Date",
    "Description",
    "Work Time (Seconds)",
    "Work Distance",
    "Pace",
    "Avg Watts",
    "Type",
}


def _validate_concept2_csv_frame(frame: pd.DataFrame, source_name: str) -> None:
    missing = sorted(_CONCEPT2_REQUIRED_CSV_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "Ugyldig Concept2 CSV-format i "
            f"{source_name}. Mangler kolonner: {', '.join(missing)}. "
            "Bruk CSV-eksport fra Concept2 Logbook i samme format som appen allerede leser."
        )


def _normalize_concept2_csv_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    raw = pd.concat(frames, ignore_index=True)

    rename_map = {
        "Log ID": "log_id",
        "Date": "date",
        "Description": "description",
        "Work Time (Formatted)": "work_time_fmt",
        "Work Time (Seconds)": "work_time_s",
        "Rest Time (Formatted)": "rest_time_fmt",
        "Rest Time (Seconds)": "rest_time_s",
        "Work Distance": "work_distance_m",
        "Rest Distance": "rest_distance_m",
        "Stroke Rate/Cadence": "stroke_rate",
        "Stroke Count": "stroke_count",
        "Pace": "pace_str",
        "Avg Watts": "avg_watts",
        "Cal/Hour": "cal_per_hour",
        "Total Cal": "total_cal",
        "Avg Heart Rate": "avg_hr",
        "Drag Factor": "drag_factor",
        "Age": "age",
        "Weight": "weight",
        "Type": "machine_type",
        "Ranked": "ranked",
        "Comments": "comments",
        "Date Entered": "date_entered",
    }

    df = raw.rename(columns=rename_map)

    df["log_id"] = _to_numeric(df.get("log_id")).astype("Int64")
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["date_entered"] = pd.to_datetime(df.get("date_entered"), errors="coerce")

    for col in [
        "work_time_s",
        "rest_time_s",
        "work_distance_m",
        "rest_distance_m",
        "stroke_rate",
        "stroke_count",
        "avg_watts",
        "cal_per_hour",
        "total_cal",
        "avg_hr",
        "drag_factor",
        "age",
        "weight",
    ]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    df["pace_s_500"] = df.get("pace_str").map(_parse_pace_to_seconds)

    df["work_time_min"] = df.get("work_time_s") / 60.0
    df["work_distance_km"] = df.get("work_distance_m") / 1000.0

    has_rest = (df.get("rest_time_s").fillna(0) > 0) | (df.get("rest_distance_m").fillna(0) > 0)
    desc = df.get("description").fillna("").astype(str)
    looks_like_interval = has_rest | desc.str.contains(r"\d+\s*x\s*\d+|/\s*\d", regex=True) | desc.str.startswith("v")

    df["workout_format"] = np.where(
        looks_like_interval,
        "Intervall",
        np.where(df.get("work_distance_m").fillna(0) > 0, "Distanse", "Tid"),
    )

    df["weekday"] = df["date"].dt.day_name()
    df["date_only"] = df["date"].dt.date

    df["label"] = (
        df["date"].dt.strftime("%Y-%m-%d %H:%M")
        + " — "
        + desc
        + " (#"
        + df["log_id"].astype(str)
        + ")"
    )

    return df


def _load_concept2_csv_from_paths(csv_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        _validate_concept2_csv_frame(df, path.name)
        df["source_file"] = path.name
        df["season"] = _season_from_filename(path)
        frames.append(df)
    return _normalize_concept2_csv_frames(frames)


def _load_concept2_csv_from_uploads(uploaded_files: list) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for uploaded in uploaded_files:
        uploaded.seek(0)
        df = pd.read_csv(uploaded)
        _validate_concept2_csv_frame(df, uploaded.name)
        df["source_file"] = uploaded.name
        df["season"] = _season_from_filename(Path(uploaded.name))
        frames.append(df)
    return _normalize_concept2_csv_frames(frames)


def _empty_concept2_history() -> pd.DataFrame:
    """Return the normalized schema used to render the portal before data is loaded."""
    return pd.DataFrame({
        "log_id": pd.Series(dtype="Int64"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "date_entered": pd.Series(dtype="datetime64[ns]"),
        "date_only": pd.Series(dtype="object"),
        "weekday": pd.Series(dtype="object"),
        "description": pd.Series(dtype="object"),
        "machine_type": pd.Series(dtype="object"),
        "work_time_s": pd.Series(dtype="float64"),
        "work_time_min": pd.Series(dtype="float64"),
        "work_distance_m": pd.Series(dtype="float64"),
        "work_distance_km": pd.Series(dtype="float64"),
        "pace_s_500": pd.Series(dtype="float64"),
        "avg_watts": pd.Series(dtype="float64"),
        "avg_hr": pd.Series(dtype="float64"),
        "stroke_rate": pd.Series(dtype="float64"),
        "stroke_count": pd.Series(dtype="float64"),
        "drag_factor": pd.Series(dtype="float64"),
        "comments": pd.Series(dtype="object"),
        "ranked": pd.Series(dtype="object"),
        "source_file": pd.Series(dtype="object"),
        "workout_format": pd.Series(dtype="object"),
        "label": pd.Series(dtype="object"),
        "_workout_intervals": pd.Series(dtype="object"),
        "_workout_splits": pd.Series(dtype="object"),
        "_strokes": pd.Series(dtype="object"),
    })


def _interval_work_avg_hr(intervals: object) -> float | None:
    if not isinstance(intervals, list):
        return None

    weighted_hr = 0.0
    total_work_s = 0.0
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        time_tenths = interval.get("time")
        heart_rate = (interval.get("heart_rate") or {}).get("average")
        if time_tenths is None or heart_rate is None:
            continue
        work_s = float(time_tenths) / 10.0
        hr_value = float(heart_rate)
        if work_s <= 0 or np.isnan(hr_value):
            continue
        weighted_hr += hr_value * work_s
        total_work_s += work_s

    if total_work_s <= 0:
        return None
    return weighted_hr / total_work_s


def _avg_hr_from_strokes(strokes: object) -> float | None:
    if not isinstance(strokes, list) or len(strokes) < 2:
        return None

    weighted_hr = 0.0
    total_work_s = 0.0
    prev_t_s: float | None = None
    prev_hr: float | None = None

    for stroke in strokes:
        if not isinstance(stroke, dict):
            continue

        t_raw = pd.to_numeric(stroke.get("t"), errors="coerce")
        hr_raw = pd.to_numeric(stroke.get("hr"), errors="coerce")
        t_s = float(t_raw) / 10.0 if pd.notna(t_raw) else None
        hr_value = float(hr_raw) if pd.notna(hr_raw) else None

        if prev_t_s is not None and prev_hr is not None and t_s is not None:
            duration_s = t_s - prev_t_s
            if duration_s > 0:
                weighted_hr += prev_hr * duration_s
                total_work_s += duration_s

        prev_t_s = t_s
        prev_hr = hr_value

    if total_work_s <= 0:
        return None
    return weighted_hr / total_work_s


@st.cache_data(show_spinner=False)
def load_concept2_history_from_csv(csv_file_paths: tuple[str, ...]) -> pd.DataFrame:
    csv_files = [Path(path_str) for path_str in csv_file_paths]
    if not csv_files:
        raise FileNotFoundError("Fant ingen aktive 'concept2-season-*.csv' i system/Data")
    return _load_concept2_csv_from_paths(csv_files)


def load_concept2_history_from_api(
    *,
    base_url: str,
    access_token: str,
    from_date: date_type | None,
    to_date: date_type | None,
    type_filter: str | None,
    _progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    cfg = Concept2ApiConfig(base_url=base_url, access_token=access_token)

    def _cache_interval_strokes(result_id: int, payload_data: dict[str, object]) -> bool:
        workout_type_value = str(payload_data.get("workout_type") or "")
        if "interval" not in workout_type_value.lower():
            return False
        if isinstance(payload_data.get("_strokes"), list):
            return False
        try:
            payload_data["_strokes"] = fetch_strokes(cfg, result_id=int(result_id))
        except Concept2ApiError:
            payload_data["_strokes"] = []
        return True

    # --- Disk cache: load existing results ---
    disk_cache = _load_api_disk_cache(type_filter)
    cache_count = len(disk_cache)

    # Determine effective from_date: if no explicit date filter is set and we
    # have cached data, only fetch sessions newer than our latest cached result
    # (with a 3-day overlap to catch any late-appearing entries).
    effective_from = from_date
    if from_date is None and disk_cache:
        cached_dates = [
            pd.to_datetime(v.get("date"), errors="coerce")
            for v in disk_cache.values()
            if v.get("date")
        ]
        valid_dates = [d for d in cached_dates if not pd.isna(d)]
        if valid_dates:
            latest = max(valid_dates)
            effective_from = (latest - pd.Timedelta(days=3)).date()

    items = fetch_all_results(
        cfg,
        from_date=effective_from,
        to_date=to_date,
        type_filter=type_filter,
    )

    # Fetch detail for each item and merge into disk_cache.
    # Existing cache entries are only overwritten when we get fresh data.
    new_count = 0
    stroke_backfill_count = 0
    total_items = len(items)
    if _progress_callback is not None:
        _progress_callback(0, total_items)

    for item_index, item in enumerate(items, start=1):
        result_id = item.get("id")
        if result_id is None:
            if _progress_callback is not None:
                _progress_callback(item_index, total_items)
            continue
        str_id = str(result_id)
        try:
            payload = fetch_result(cfg, result_id=int(result_id))
            data = payload.get("data")
            if isinstance(data, dict):
                if _cache_interval_strokes(int(result_id), data):
                    stroke_backfill_count += 1
                disk_cache[str_id] = data
                new_count += 1
            elif str_id not in disk_cache:
                disk_cache[str_id] = item
                new_count += 1
        except Exception:
            if str_id not in disk_cache:
                disk_cache[str_id] = item
                new_count += 1
        if _progress_callback is not None:
            _progress_callback(item_index, total_items)

    for cached_id, cached_data in disk_cache.items():
        if not isinstance(cached_data, dict):
            continue
        try:
            result_id = int(cached_id)
        except (TypeError, ValueError):
            continue
        if _cache_interval_strokes(result_id, cached_data):
            stroke_backfill_count += 1

    # Persist updated cache to disk whenever something changed.
    if new_count > 0 or cache_count == 0 or stroke_backfill_count > 0:
        _save_api_disk_cache(type_filter, disk_cache)

    # Store summary for UI display (works even inside @st.cache_data).
    try:
        st.session_state["_api_cache_info"] = {
            "total": len(disk_cache),
            "from_cache": cache_count,
            "new": new_count,
        }
    except Exception:
        pass

    # Build the full list from the merged cache and process it.
    detailed: list[dict] = list(disk_cache.values())

    raw = pd.json_normalize(detailed)

    df = pd.DataFrame()
    df["log_id"] = pd.to_numeric(raw.get("id"), errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(raw.get("date"), errors="coerce")
    df["comments"] = raw.get("comments")

    api_type = raw.get("type").fillna("").astype(str)
    df["machine_type"] = api_type.str.title().replace({"Skierg": "SkiErg", "Rower": "RowErg"})

    df["work_distance_m"] = pd.to_numeric(raw.get("distance"), errors="coerce")
    time_tenths = pd.to_numeric(raw.get("time"), errors="coerce")
    df["work_time_s"] = time_tenths / 10.0

    df["pace_s_500"] = np.where(
        (df["work_distance_m"].fillna(0) > 0) & (df["work_time_s"].fillna(0) > 0),
        (df["work_time_s"] / df["work_distance_m"]) * 500.0,
        np.nan,
    )
    df["avg_watts"] = _watts_from_pace_s_500(df["pace_s_500"])

    time_fmt = raw.get("time_formatted").fillna("").astype(str)
    workout_type = raw.get("workout_type").fillna("").astype(str)
    src = raw.get("source").fillna("").astype(str)

    dist = df["work_distance_m"].fillna(0)
    dist_desc = np.where(dist > 0, dist.round(0).astype(int).astype(str) + "m", "")

    # Fallback description if user has no comments
    auto_desc = (
        (dist_desc + " ").astype(str)
        + df["machine_type"].fillna("").astype(str)
        + np.where(workout_type != "", " — " + workout_type, "")
        + np.where(time_fmt != "", " — " + time_fmt, "")
        + np.where(src != "", " (" + src + ")", "")
    ).str.strip()

    # Use user-entered comments (same as CSV Description) when available
    comments = raw.get("comments").fillna("").astype(str) if raw.get("comments") is not None else pd.Series([""] * len(df))
    df["description"] = np.where(comments.str.strip() != "", comments.str.strip(), auto_desc)

    ranked_raw = raw.get("ranked")
    if ranked_raw is None:
        df["ranked"] = np.nan
    else:
        r = pd.Series(ranked_raw)
        df["ranked"] = np.where(r.fillna(False).astype(bool), "Yes", "No")

    df["stroke_rate"] = pd.to_numeric(raw.get("stroke_rate"), errors="coerce")
    df["stroke_count"] = pd.to_numeric(raw.get("stroke_count"), errors="coerce")
    df["drag_factor"] = pd.to_numeric(raw.get("drag_factor"), errors="coerce")

    hr_avg = raw.get("heart_rate.average")
    if hr_avg is None and "heart_rate" in raw.columns:
        hr_avg = raw["heart_rate"].apply(lambda x: x.get("average") if isinstance(x, dict) else np.nan)
    df["avg_hr"] = pd.to_numeric(hr_avg, errors="coerce")

    # Store per-interval / per-split data for session detail view.
    # GET /results/{id} returns the workout body including workout.intervals
    # (for interval workout types) or workout.splits (for split workout types).
    # After pd.json_normalize the nested workout dict expands into columns named
    # "workout.intervals" and "workout.splits" respectively.
    df["_workout_intervals"] = (
        raw["workout.intervals"].tolist()
        if "workout.intervals" in raw.columns
        else [None] * len(raw)
    )
    df["_workout_splits"] = (
        raw["workout.splits"].tolist()
        if "workout.splits" in raw.columns
        else [None] * len(raw)
    )
    df["_strokes"] = raw.get("_strokes", pd.Series([None] * len(raw))).tolist()

    df["season"] = "api"
    df["source_file"] = "concept2_api"

    df["work_time_min"] = df["work_time_s"] / 60.0
    df["work_distance_km"] = df["work_distance_m"] / 1000.0

    df["workout_format"] = np.where(
        workout_type.str.contains("Interval", case=False, na=False),
        "Intervall",
        np.where(df["work_distance_m"].fillna(0) > 0, "Distanse", "Tid"),
    )

    interval_stroke_avg_hr = df["_strokes"].apply(_avg_hr_from_strokes)
    df["avg_hr"] = pd.to_numeric(
        np.where(
            df["workout_format"].eq("Intervall") & interval_stroke_avg_hr.notna(),
            interval_stroke_avg_hr,
            df["avg_hr"],
        ),
        errors="coerce",
    )

    df["weekday"] = df["date"].dt.day_name()
    df["date_only"] = df["date"].dt.date

    df["label"] = (
        df["date"].dt.strftime("%Y-%m-%d %H:%M")
        + " — "
        + df["description"].fillna("").astype(str)
        + " (#"
        + df["log_id"].astype(str)
        + ")"
    )

    for missing in ["rest_time_s", "rest_distance_m"]:
        if missing not in df.columns:
            df[missing] = np.nan

    return df


def _parse_workout_to_df(items: list, is_interval: bool) -> pd.DataFrame | None:
    """Parse Concept2 API workout.intervals or workout.splits into a display DataFrame.

    Interval format (workout_type = Fixed*/Variable*Interval):
      Each item in workout.intervals has:
        time          – work time in tenths of a second
        rest_time     – rest time in tenths of a second
        distance      – work distance in meters
        rest_distance – rest distance in meters (VariableInterval only)
        stroke_rate   – average stroke rate
        calories_total
        heart_rate    – object with optional keys: average, min, max, ending, rest

    Split format (workout_type = FixedDistance/TimeSplits etc.):
      Each item in workout.splits has the same fields minus rest_time/rest_distance.
    """
    if not isinstance(items, list) or len(items) == 0:
        return None
    rows = []
    for i, s in enumerate(items):
        if not isinstance(s, dict):
            continue
        time_tenths = s.get("time")
        time_s = float(time_tenths) / 10.0 if time_tenths is not None else None
        dist_m = s.get("distance")
        pace_s: float | None = None
        if time_s and dist_m and float(dist_m) > 0:
            pace_s = time_s / float(dist_m) * 500.0
        watt: float | None = None
        if pace_s and pace_s > 0:
            try:
                watt = round(2.8 / (pace_s / 500.0) ** 3, 1)
            except Exception:
                watt = None
        hr = s.get("heart_rate") or {}
        row: dict = {
            "Nr": i + 1,
            "Distanse (m)": int(float(dist_m)) if dist_m is not None else None,
            "Tid": _format_duration(time_s),
            "Pace": _format_pace(pace_s),
            "Watt": watt,
            "SPM": s.get("stroke_rate"),
            "Kalorier": s.get("calories_total"),
            "Puls (avg)": hr.get("average"),
            "Puls (slutt)": hr.get("ending"),
        }
        if is_interval:
            rest_tenths = s.get("rest_time")
            rest_s = float(rest_tenths) / 10.0 if rest_tenths is not None else None
            row["Hvile (tid)"] = _format_duration(rest_s)
            row["Puls (hvile)"] = hr.get("rest")
            rest_dist = s.get("rest_distance")
            if rest_dist is not None:
                row["Hvile (m)"] = int(float(rest_dist))
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def _render_session_detail(row: pd.Series) -> None:
    """Render a detail panel for a selected session row."""
    desc = row.get("description") or "—"
    _wkey = _session_weight_key(row)
    _wkey_safe = _wkey.replace(":", "_").replace("|", "_").replace(" ", "_")[:60]
    st.subheader(f"Detaljer — {desc}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Dato", str(row.get("date", ""))[:16])
    c2.metric("Tid", _format_duration(row.get("work_time_s")))
    dist_km = row.get("work_distance_km")
    c3.metric("Distanse", f"{float(dist_km):.3f} km" if pd.notna(dist_km) else "—")
    c4.metric("Pace", _format_pace(row.get("pace_s_500")))
    w = row.get("avg_watts")
    c5.metric("Watt", f"{float(w):.0f} W" if pd.notna(w) else "—")
    hr = row.get("avg_hr")
    c6.metric("Puls", f"{float(hr):.0f}" if pd.notna(hr) else "—")

    benchmark_name = row.get("benchmark_name")
    benchmark_status = row.get("benchmark_status") or ""
    if benchmark_name:
        if benchmark_status:
            st.markdown(
                f"<small><b>Benchmark:</b> {benchmark_name} &nbsp;"
                f"<span style='background:#d9f99d;color:#111827;padding:3px 10px;border-radius:999px;font-weight:600'>"
                f"{benchmark_status}</span></small>",
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"Benchmark: {benchmark_name}")

    share_png = _build_share_card_png(row)
    share_b64 = base64.b64encode(share_png).decode("ascii")
    share_file = _share_card_filename(row)
    st.markdown(
        (
            f'<a href="data:image/png;base64,{share_b64}" '
            f'download="{share_file}" '
            'style="display:inline-flex;align-items:center;justify-content:center;'
            'padding:0.42rem 0.72rem;background:#fc5200;color:#ffffff;'
            'border-radius:0.65rem;text-decoration:none;font-weight:700;font-size:0.88rem;'
            'box-shadow:0 6px 16px rgba(252, 82, 0, 0.18);margin:0.25rem 0 0.15rem 0;" '
            'title="Lager et enkelt delingskort i PNG med nøkkeltall fra denne økten.">'
            'Del på Strava (PNG)'
            '</a>'
        ),
        unsafe_allow_html=True,
    )

    # Zone badge from avg_hr
    if pd.notna(hr):
        _zn_hdr = _get_zones()
        _hf_hdr = _get_hf_maks()
        _z_hit = next(
            (z for z in _zn_hdr
             if _hf_hdr * z["lo"] / 100 <= float(hr) <= _hf_hdr * z["hi"] / 100),
            None,
        )
        _badge = (
            f"<span style='background:{_z_hit['color']};color:#000;"
            f"padding:3px 12px;border-radius:5px;font-weight:bold'>"
            f"Sone {_z_hit['name']} — {_z_hit['label']}</span>"
            f"&nbsp;<small style='color:#888'>snitt-puls, {_hf_hdr} bpm maks</small>"
        ) if _z_hit else (
            f"<small style='color:#888'>Snitt-puls {float(hr):.0f} bpm — utenfor definerte soner ({_hf_hdr} bpm maks)</small>"
        )
        st.markdown(_badge, unsafe_allow_html=True)

    # --- Vekt og W/kg ---
    _stored_weights = _load_weights()
    _current_weight = _stored_weights.get(_wkey, 0.0)
    wc1, wc2, wc3 = st.columns([1, 1, 2])
    with wc1:
        _new_weight = st.number_input(
            "Vekt (kg)",
            min_value=0.0,
            max_value=250.0,
            value=float(_current_weight),
            step=0.5,
            format="%.1f",
            key=f"weight_input_{_wkey_safe}",
        )
    with wc2:
        _watt_val = row.get("avg_watts")
        if pd.notna(_watt_val) and _new_weight > 0:
            st.metric("W/kg", f"{float(_watt_val) / _new_weight:.2f}")
        else:
            st.metric("W/kg", "—")
    with wc3:
        st.write("")
        st.write("")
        if st.button("Lagre vekt", key=f"weight_save_{_wkey_safe}"):
            _save_weight(_wkey, _new_weight if _new_weight > 0 else None)
            st.rerun()

    # --- Kommentar ---
    _comments = _load_comments()
    _current_comment = _comments.get(_wkey, "")
    _comment_widget_key = f"comment_input_{_wkey_safe}"
    _new_comment = st.text_area(
        "Kommentar",
        value=_current_comment,
        height=80,
        key=_comment_widget_key,
        on_change=_autosave_comment,
        args=(_wkey, _comment_widget_key),
    )

    # --- Tagger ---
    _tags_map = _load_tags()
    _current_tags = _tags_map.get(_wkey, [])
    _tag_options = _tag_catalog()
    _tags_widget_key = f"tags_select_{_wkey_safe}"
    _selected_tags = st.multiselect(
        "Tagger",
        options=_tag_options,
        default=_current_tags,
        key=_tags_widget_key,
        help="Administrer og legg til tagger under Innstillinger.",
        on_change=_autosave_tags,
        args=(_wkey, _tags_widget_key),
    )

    _lactates = _load_lactates()
    _lactate_entry = _lactates.get(_wkey, {}) if isinstance(_lactates, dict) else {}

    intervals_raw = row.get("_workout_intervals")
    splits_raw_val = row.get("_workout_splits")
    is_interval = isinstance(intervals_raw, list) and len(intervals_raw) > 0
    items = intervals_raw if is_interval else (
        splits_raw_val if isinstance(splits_raw_val, list) and len(splits_raw_val) > 0 else None
    )

    workout_df = _parse_workout_to_df(items, is_interval) if items else None

    if workout_df is not None:
        _interval_lactates = _lactate_entry.get("interval_lactates") if isinstance(_lactate_entry, dict) else {}
        _stored_threshold_test = _lactate_entry.get("threshold_test") if isinstance(_lactate_entry, dict) else None
        _stored_threshold_test = _clean_threshold_test(_stored_threshold_test)
        workout_df = workout_df.copy()
        lactate_column_position = workout_df.columns.get_loc("Puls (slutt)") + 1
        workout_df.insert(
            lactate_column_position,
            "Laktat",
            workout_df["Nr"].map(
                lambda nr: _interval_lactates.get(str(int(nr)))
                if isinstance(_interval_lactates, dict) and pd.notna(nr)
                else None
            ),
        )
        header = "Intervaller og pauser" if is_interval else "Splits"
        st.markdown(f"#### {header}")
        edited_workout_df = st.data_editor(
            _table_display(workout_df),
            use_container_width=True,
            hide_index=True,
            disabled=[column for column in workout_df.columns if column != "Laktat"],
            column_config={
                "Laktat": st.column_config.NumberColumn(
                    "Laktat",
                    min_value=0.0,
                    max_value=30.0,
                    step=0.1,
                    format="%.1f",
                )
            },
            key=f"workout_lactate_editor_{_wkey_safe}",
        )

        actual_durations = [
            float(item.get("time")) / 10.0
            for item in items
            if isinstance(item, dict) and item.get("time") is not None
        ]
        default_duration_min = round(actual_durations[0] / 60.0, 1) if actual_durations else 4.0
        is_threshold_test = st.checkbox(
            "Dette er en terskeltest",
            value=_stored_threshold_test is not None,
            key=f"threshold_test_enabled_{_wkey_safe}",
        )
        threshold_test_payload: dict[str, object] | None = None
        if is_threshold_test:
            st.markdown("##### Terskeltest-oppsett")
            tc1, tc2, tc3, tc4 = st.columns([2, 1, 1, 1])
            with tc1:
                threshold_name = st.text_input(
                    "Testnavn",
                    value=str(_stored_threshold_test.get("protocol_name") or "") if _stored_threshold_test else "",
                    placeholder="f.eks. SkiErg 6 x 5 min",
                    key=f"threshold_test_name_{_wkey_safe}",
                )
            with tc2:
                threshold_interval_count = st.number_input(
                    "Antall drag",
                    min_value=1,
                    max_value=30,
                    value=int(_stored_threshold_test.get("interval_count")) if _stored_threshold_test else len(items),
                    step=1,
                    key=f"threshold_test_count_{_wkey_safe}",
                )
            with tc3:
                threshold_duration_min = st.number_input(
                    "Arbeidstid (min)",
                    min_value=0.5,
                    max_value=60.0,
                    value=float(_stored_threshold_test.get("interval_duration_s", 0)) / 60.0 if _stored_threshold_test else default_duration_min,
                    step=0.5,
                    key=f"threshold_test_duration_{_wkey_safe}",
                )
            with tc4:
                threshold_rest_s = st.number_input(
                    "Pause (sek)",
                    min_value=0,
                    max_value=1800,
                    value=int(_stored_threshold_test.get("rest_duration_s", 0)) if _stored_threshold_test else 30,
                    step=15,
                    key=f"threshold_test_rest_{_wkey_safe}",
                )
            expected_duration_s = int(round(float(threshold_duration_min) * 60))
            count_matches = int(threshold_interval_count) == len(items)
            duration_matches = bool(actual_durations) and all(abs(duration - expected_duration_s) <= 15 for duration in actual_durations)
            if count_matches and duration_matches:
                st.success("Økten matcher valgt terskeltest-oppsett.")
            else:
                st.warning(f"Økten har {len(items)} drag/splitter. Sjekk at antall og arbeidstid samsvarer med oppsettet.")
            threshold_test_payload = {
                "protocol_name": threshold_name,
                "interval_count": int(threshold_interval_count),
                "interval_duration_s": expected_duration_s,
                "rest_duration_s": int(threshold_rest_s),
            }

        _new_interval_lactates = {
            str(int(lactate_row["Nr"])): float(lactate_row["Laktat"])
            for _, lactate_row in edited_workout_df.iterrows()
            if pd.notna(lactate_row["Laktat"]) and float(lactate_row["Laktat"]) > 0
        }
        _stored_interval_lactates = _clean_interval_lactates(_interval_lactates)
        if (
            _new_interval_lactates != _stored_interval_lactates
            or _clean_threshold_test(threshold_test_payload) != _stored_threshold_test
        ):
            _save_lactate_entry(
                _wkey,
                session_lactate=_lactate_entry.get("session_lactate"),
                note=str(_lactate_entry.get("session_lactate_note") or ""),
                timing=str(_lactate_entry.get("session_lactate_measure_timing") or "ukjent"),
                interval_lactates=_new_interval_lactates,
                threshold_test=threshold_test_payload,
            )
            st.toast("Laktat og terskeltest er lagret.")

        if "Watt" in workout_df.columns:
            watt_series = pd.to_numeric(workout_df["Watt"], errors="coerce").dropna()
            if not watt_series.empty:
                plot_df = workout_df.copy()
                plot_df["Watt"] = pd.to_numeric(plot_df["Watt"], errors="coerce")
                fig = px.bar(plot_df.dropna(subset=["Watt"]), x="Nr", y="Watt", text="Watt")
                fig.update_traces(texttemplate="%{text:.0f} W", textposition="outside")
                fig.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Intervall nr." if is_interval else "Split nr.",
                    yaxis_title="W",
                )
                fig.update_yaxes(range=[0, float(watt_series.max()) * 1.18])
                st.plotly_chart(fig, use_container_width=True)

        # --- Pulssoner fra per-intervall HR ---
        if items:
            _iv_hr_pairs: list[tuple[float, float]] = []
            for _s in items:
                if not isinstance(_s, dict):
                    continue
                _t10 = _s.get("time")
                _ts_iv = float(_t10) / 10.0 if _t10 is not None else None
                _hr_avg = (_s.get("heart_rate") or {}).get("average")
                if _ts_iv and _hr_avg and _ts_iv > 0:
                    _iv_hr_pairs.append((float(_hr_avg), _ts_iv))
            if _iv_hr_pairs:
                _hf_iv = _get_hf_maks()
                _zn_iv = _get_zones()
                st.markdown("#### Pulssoner (per intervall)")
                _render_zone_chart(
                    _compute_zone_times(_iv_hr_pairs, _zn_iv, _hf_iv),
                    _zn_iv, _hf_iv,
                    source_label=f"snitt-puls per intervall ({len(_iv_hr_pairs)} intervaller med pulsdata)",
                )
    else:
        st.info(
            "Ingen per-intervall data tilgjengelig. "
            "Intervalldata vises kun når datakilde er Concept2 API og økten er registrert med intervaller."
        )

        # Fallback zone from avg_hr when no interval/stroke data
        _avg_hr_fb = row.get("avg_hr")
        _work_t_fb = row.get("work_time_s")
        if pd.notna(_avg_hr_fb) and pd.notna(_work_t_fb) and float(_work_t_fb) > 0:
            _hf_fb = _get_hf_maks()
            _zn_fb = _get_zones()
            st.markdown("#### Pulssone (estimat)")
            _render_zone_chart(
                _compute_zone_times([(float(_avg_hr_fb), float(_work_t_fb))], _zn_fb, _hf_fb),
                _zn_fb, _hf_fb,
                source_label="snitt-puls for hele økten (tilordnet én sone)",
            )

    # --- Per-slag data (HR / pace / SPM) ---
    # Fetched on demand from GET /api/users/me/results/{id}/strokes.
    # Not all workouts have stroke data; a 404 is expected and handled.
    api_cfg = st.session_state.get("_api_cfg")
    log_id = row.get("log_id")
    if api_cfg and pd.notna(log_id):
        stroke_key = f"_strokes_{int(log_id)}"
        if stroke_key not in st.session_state:
            if st.button(
                "Hent per-slag data (puls / pace / SPM)",
                key=f"btn_strokes_{int(log_id)}",
            ):
                with st.spinner("Henter stroke data …"):
                    try:
                        _cfg = Concept2ApiConfig(
                            base_url=api_cfg["base_url"],
                            access_token=api_cfg["access_token"],
                        )
                        st.session_state[stroke_key] = fetch_strokes(
                            _cfg, result_id=int(log_id)
                        )
                    except Concept2ApiError as _exc:
                        st.session_state[stroke_key] = []
                        st.warning(str(_exc))
                st.rerun()

        strokes = st.session_state.get(stroke_key)
        if isinstance(strokes, list) and len(strokes) > 0:
            st.markdown("#### Per-slag data")
            sdf = pd.DataFrame(strokes)
            # t: tenths of sec (cumulative per interval – resets to 0 at each
            # interval boundary for interval workouts).
            # Build a globally cumulative t_s by detecting resets in the RAW
            # t values and accumulating an offset. Using a separate t_raw array
            # for comparison prevents false reset detections on modified values.
            sdf["_t_raw"] = pd.to_numeric(sdf.get("t"), errors="coerce") / 10.0
            t_raw = sdf["_t_raw"].to_numpy(dtype=float, na_value=float("nan")).copy()
            t_global = t_raw.copy()
            offset = 0.0
            for _i in range(1, len(t_raw)):
                r_cur = t_raw[_i]
                r_prev = t_raw[_i - 1]
                if r_cur == r_cur and r_prev == r_prev:
                    if r_cur < r_prev:
                        offset += r_prev
                t_global[_i] = t_raw[_i] + offset
            sdf["t_s"] = t_global
            sdf["pace_s"] = pd.to_numeric(sdf.get("p"), errors="coerce") / 10.0
            sdf["hr"] = pd.to_numeric(sdf.get("hr"), errors="coerce")
            sdf["spm"] = pd.to_numeric(sdf.get("spm"), errors="coerce")

            col1, col2 = st.columns(2)
            with col1:
                if sdf["hr"].notna().any():
                    fig_hr = px.line(
                        sdf.dropna(subset=["hr"]),
                        x="t_s",
                        y="hr",
                        title="Puls",
                    )
                    fig_hr.update_yaxes(range=[0, 200])
                    fig_hr.update_layout(
                        height=220,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title="Tid (s)",
                        yaxis_title="Puls (bpm)",
                    )
                    st.plotly_chart(fig_hr, use_container_width=True)
            with col2:
                pace_clean = sdf.dropna(subset=["pace_s"])
                pace_clean = pace_clean[pace_clean["pace_s"] > 0]
                if not pace_clean.empty:
                    fig_pace = px.line(
                        pace_clean,
                        x="t_s",
                        y="pace_s",
                        title="Pace",
                    )
                    pv = pace_clean["pace_s"]
                    y_pad = (pv.max() - pv.min()) * 0.08 + 1
                    fig_pace.update_yaxes(range=[pv.max() + y_pad, pv.min() - y_pad])
                    fig_pace.update_layout(
                        height=220,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title="Tid (s)",
                        yaxis_title="Pace (s/500 m)",
                    )
                    st.plotly_chart(fig_pace, use_container_width=True)

            # --- Pulssoner fra per-slag pulsdata (mest nøyaktig) ---
            if sdf["hr"].notna().any():
                st.markdown("#### Pulssoner (per slag — nøyaktig)")
                _hf_sk = _get_hf_maks()
                _zn_sk = _get_zones()
                _t_a = sdf["t_s"].to_numpy(dtype=float)
                _hr_a = sdf["hr"].to_numpy(dtype=float)
                _sk_pairs: list[tuple[float, float]] = []
                for _si in range(len(_t_a) - 1):
                    _dur_sk = float(_t_a[_si + 1] - _t_a[_si])
                    if _dur_sk > 0 and np.isfinite(_hr_a[_si]):
                        _sk_pairs.append((_hr_a[_si], _dur_sk))
                if _sk_pairs:
                    _render_zone_chart(
                        _compute_zone_times(_sk_pairs, _zn_sk, _hf_sk),
                        _zn_sk, _hf_sk,
                        source_label=f"per-slag måling ({len(_sk_pairs)} slag med pulsdata)",
                    )


def _compute_zone_times(
    hr_time_pairs: list[tuple[float, float]],
    zones: list[dict],
    hf_maks: int,
) -> dict[str, float]:
    """Compute accumulated seconds in each HR zone.

    hr_time_pairs: list of (hr_bpm, duration_s) — duration may be fractional.
    Returns dict  zone_name -> seconds, plus '< I1' and '> I5' buckets.
    """
    result: dict[str, float] = {z["name"]: 0.0 for z in zones}
    result["< I1"] = 0.0
    result["> I5"] = 0.0
    boundaries = [
        (hf_maks * z["lo"] / 100.0, hf_maks * z["hi"] / 100.0, z["name"])
        for z in zones
    ]
    for hr_val, dur_s in hr_time_pairs:
        try:
            hr_f, dur_f = float(hr_val), float(dur_s)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(hr_f) and np.isfinite(dur_f) and dur_f > 0):
            continue
        placed = False
        for lo_bpm, hi_bpm, zone_name in boundaries:
            if lo_bpm <= hr_f <= hi_bpm:
                result[zone_name] += dur_f
                placed = True
                break
        if not placed:
            if hr_f < boundaries[0][0]:
                result["< I1"] += dur_f
            else:
                result["> I5"] += dur_f
    return result


def _render_zone_chart(
    zone_times: dict[str, float],
    zones: list[dict],
    hf_maks: int,
    source_label: str = "",
) -> None:
    """Render time-in-zone as coloured bar chart + summary table."""
    total_s = sum(zone_times.values())
    if total_s <= 0:
        st.caption("Ingen pulsdata tilgjengelig for soneinndeling.")
        return

    rows: list[dict] = []
    below_t = zone_times.get("< I1", 0.0)
    if below_t > 0:
        rows.append({
            "Sone": "< I1", "Navn": "Under I1",
            "Pulsområde": f"< {round(hf_maks * zones[0]['lo'] / 100)} bpm",
            "_color": "#cccccc", "_t": below_t,
            "Tid": _format_duration(below_t),
            "%": round(below_t / total_s * 100, 1),
        })
    for z in zones:
        t = zone_times.get(z["name"], 0.0)
        rows.append({
            "Sone": z["name"], "Navn": z["label"],
            "Pulsområde": f"{round(hf_maks * z['lo'] / 100)}–{round(hf_maks * z['hi'] / 100)} bpm",
            "_color": z["color"], "_t": t,
            "Tid": _format_duration(t),
            "%": round(t / total_s * 100, 1),
        })
    above_t = zone_times.get("> I5", 0.0)
    if above_t > 0:
        rows.append({
            "Sone": "> I5", "Navn": "Over I5",
            "Pulsområde": f"> {round(hf_maks * zones[-1]['hi'] / 100)} bpm",
            "_color": "#800026", "_t": above_t,
            "Tid": _format_duration(above_t),
            "%": round(above_t / total_s * 100, 1),
        })

    zdf = pd.DataFrame(rows)
    zdf_vis = zdf[zdf["_t"] > 0].copy()

    if zdf_vis.empty:
        return

    if source_label:
        st.caption(f"Kilde: {source_label}")

    color_map = dict(zip(zdf_vis["Sone"], zdf_vis["_color"]))
    fig = px.bar(
        zdf_vis, x="Sone", y="_t",
        color="Sone",
        color_discrete_map=color_map,
        text="%",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        customdata=zdf_vis[["Tid", "Navn", "Pulsområde"]].values,
        hovertemplate="<b>%{x}</b> — %{customdata[1]}<br>%{customdata[2]}<br>%{customdata[0]} (%{text:.1f}%)<extra></extra>",
    )
    fig.update_yaxes(showticklabels=False, title="")
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
        xaxis_title="",
    )
    max_pct = float(zdf_vis["%"].max())
    fig.update_yaxes(range=[0, float(zdf_vis["_t"].max()) * (1 + max_pct / 60)])
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        zdf_vis[["Sone", "Navn", "Pulsområde", "Tid", "%"]],
        use_container_width=True,
        hide_index=True,
    )


def _render_settings_tab() -> None:
    """Render the Innstillinger (Settings) tab."""
    settings = _load_settings()
    hf_maks_cur = int(settings.get("hf_maks", 200))
    zones_cur = _get_zones()
    st.subheader("Makspuls (HF maks)")
    hf_col, _ = st.columns([1, 3])
    with hf_col:
        new_hf = st.number_input(
            "HF maks (slag/min)",
            min_value=100,
            max_value=250,
            value=hf_maks_cur,
            step=1,
            help="Brukes til å beregne pulssoner og tid i sone.",
        )

    st.divider()
    st.subheader("Pulssoner — Olympiatoppen (I1–I5)")
    st.caption(
        "Soneinndelingen følger Olympiatoppens modell: fem intensitetssoner basert på "
        "prosentandel av makspuls. Juster grensene manuelt ved behov."
    )

    new_zones: list[dict] = []
    header_cols = st.columns([1, 2, 1, 1, 1, 1])
    for col_w, lbl in zip(header_cols, ["Sone", "Beskrivelse", "Fra (%)", "Til (%)", "Fra (bpm)", "Til (bpm)"]):
        col_w.markdown(f"**{lbl}**")

    for z in zones_cur:
        c_name, c_lbl, c_lo, c_hi, c_lo_bpm, c_hi_bpm = st.columns([1, 2, 1, 1, 1, 1])
        lo_bpm = round(new_hf * z["lo"] / 100)
        hi_bpm = round(new_hf * z["hi"] / 100)
        with c_name:
            st.markdown(
                f"<span style='background:{z['color']};color:#000;padding:2px 8px;"
                f"border-radius:4px;font-weight:bold'>{z['name']}</span>",
                unsafe_allow_html=True,
            )
        c_lbl.write(z["label"])
        new_lo = c_lo.number_input(
            "Fra %", min_value=0, max_value=100, value=int(z["lo"]),
            step=1, key=f"zone_lo_{z['name']}", label_visibility="collapsed"
        )
        new_hi = c_hi.number_input(
            "Til %", min_value=0, max_value=100, value=int(z["hi"]),
            step=1, key=f"zone_hi_{z['name']}", label_visibility="collapsed"
        )
        c_lo_bpm.write(f"{round(new_hf * new_lo / 100)} bpm")
        c_hi_bpm.write(f"{round(new_hf * new_hi / 100)} bpm")
        new_zones.append({**z, "lo": new_lo, "hi": new_hi})

    st.divider()
    # Visual zone bar
    fig_zones = go.Figure()
    for z in new_zones:
        lo_b = round(new_hf * z["lo"] / 100)
        hi_b = round(new_hf * z["hi"] / 100)
        fig_zones.add_trace(go.Bar(
            x=[hi_b - lo_b],
            base=[lo_b],
            orientation="h",
            name=f"{z['name']} {z['label']}",
            marker_color=z["color"],
            text=f"{z['name']}<br>{lo_b}–{hi_b} bpm",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=(
                f"<b>{z['name']} — {z['label']}</b><br>"
                f"{z['lo']}–{z['hi']}% av HF maks<br>"
                f"{lo_b}–{hi_b} bpm<extra></extra>"
            ),
        ))
    fig_zones.update_layout(
        barmode="stack",
        height=80,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(range=[0, new_hf + 5], title="Puls (bpm)"),
        yaxis=dict(showticklabels=False),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_zones, use_container_width=True)

    if st.button("💾 Lagre innstillinger", type="primary"):
        _update_settings({"hf_maks": int(new_hf), "zones": new_zones})
        st.success(f"Lagret! HF maks = {int(new_hf)} bpm, {len(new_zones)} soner oppdatert.")
        st.rerun()

    st.divider()
    if st.button("↩️ Tilbakestill til Olympiatoppen-standard", key="reset_zones_btn"):
        _update_settings({"hf_maks": int(new_hf), "zones": [z.copy() for z in _OPT_ZONES_DEFAULT]})
        st.success("Soner tilbakestilt til Olympiatoppen-standard.")
        st.rerun()

    st.divider()
    st.subheader("Tagger")
    tag_catalog = _configured_tag_catalog()
    tag_usage = _load_tags()
    tag_counts: dict[str, int] = {}
    for session_tags in tag_usage.values():
        for tag in session_tags:
            tag_counts[tag.casefold()] = tag_counts.get(tag.casefold(), 0) + 1

    tag_colors = [
        ("#dceefe", "#174a7e"),
        ("#dff4e8", "#17623c"),
        ("#fff0cf", "#805500"),
        ("#fce1e1", "#8b2424"),
        ("#eee5fb", "#56348b"),
        ("#ddf3f2", "#176965"),
    ]
    tag_cards = []
    for index, tag in enumerate(tag_catalog):
        background, foreground = tag_colors[index % len(tag_colors)]
        count = tag_counts.get(tag.casefold(), 0)
        tag_cards.append(
            f'<span style="display:inline-block;margin:0 8px 8px 0;padding:7px 10px;'
            f'border-radius:6px;background:{background};color:{foreground};font-weight:600">'
            f'{html.escape(tag)} <span style="font-weight:400">({count})</span></span>'
        )
    if tag_cards:
        st.markdown("".join(tag_cards), unsafe_allow_html=True)
    new_tags_raw = st.text_input(
        "Legg til tagger",
        placeholder="f.eks. konkurranse, test / høydetrening",
        help="Bruk komma eller / for flere tagger.",
    )
    if st.button("Lagre nye tagger", key="save_tag_catalog_btn"):
        additions = [part.strip() for part in re.split(r"[,/]", new_tags_raw) if part.strip()]
        configured = settings.get("tag_catalog", [])
        hidden = settings.get("hidden_tag_catalog", [])
        addition_keys = {tag.casefold() for tag in additions}
        updated_hidden = [
            tag for tag in hidden
            if str(tag).strip().casefold() not in addition_keys
        ] if isinstance(hidden, list) else []
        _update_settings(
            {
                "tag_catalog": [*configured, *additions] if isinstance(configured, list) else additions,
                "hidden_tag_catalog": updated_hidden,
            }
        )
        st.rerun()

    if tag_catalog:
        tag_to_delete = st.selectbox("Slett tagg", tag_catalog)
        delete_mode = st.radio(
            "Hva skal skje med eksisterende økter?",
            ["Behold taggen på eksisterende økter", "Fjern taggen fra alle økter"],
            horizontal=True,
        )
        if st.button("Slett valgt tagg", key="delete_tag_catalog_btn"):
            tag_key = tag_to_delete.casefold()
            configured = settings.get("tag_catalog", [])
            hidden = settings.get("hidden_tag_catalog", [])
            updated_catalog = [
                tag for tag in configured
                if str(tag).strip().casefold() != tag_key
            ] if isinstance(configured, list) else []
            updated_hidden = [
                *(hidden if isinstance(hidden, list) else []),
                tag_to_delete,
            ]
            _update_settings(
                {
                    "tag_catalog": updated_catalog,
                    "hidden_tag_catalog": updated_hidden,
                }
            )
            if delete_mode == "Fjern taggen fra alle økter":
                _remove_tag_from_sessions(tag_to_delete)
            st.rerun()


def _render_interval_comparison(all_rows: list, labels: list) -> None:
    """Per-interval comparison across sessions with best-value highlighting."""

    def _extract_raw(sess_row: pd.Series):
        ivs = sess_row.get("_workout_intervals")
        sps = sess_row.get("_workout_splits")
        is_iv = isinstance(ivs, list) and len(ivs) > 0
        items = ivs if is_iv else (sps if isinstance(sps, list) and len(sps) > 0 else None)
        if not items:
            return None
        rows = []
        for i, s in enumerate(items):
            if not isinstance(s, dict):
                continue
            t10 = s.get("time")
            time_s = float(t10) / 10.0 if t10 is not None else None
            dist_m = s.get("distance")
            pace_s = None
            if time_s and dist_m and float(dist_m) > 0:
                pace_s = time_s / float(dist_m) * 500.0
            watt = None
            if pace_s and pace_s > 0:
                try:
                    watt = round(2.8 / (pace_s / 500.0) ** 3, 1)
                except Exception:
                    pass
            hr_obj = s.get("heart_rate") or {}
            rows.append({
                "Nr": i + 1,
                "_time_s": time_s,
                "_pace_s": pace_s,
                "_watt": watt,
                "_puls": hr_obj.get("average"),
                "_spm": s.get("stroke_rate"),
            })
        return pd.DataFrame(rows) if rows else None

    sessions_data: list = []
    for sess_row, label in zip(all_rows, labels):
        df = _extract_raw(sess_row)
        if df is not None and not df.empty:
            sessions_data.append((label, df))

    if len(sessions_data) < 2:
        return

    st.markdown("#### Per-intervall sammenligning")
    max_nr = int(max(df["Nr"].max() for _, df in sessions_data))

    def _num_pivot(col: str) -> pd.DataFrame:
        frames = {}
        for lbl, df in sessions_data:
            if col in df.columns:
                frames[lbl] = df.set_index("Nr")[col].reindex(range(1, max_nr + 1))
        return pd.DataFrame(frames) if frames else pd.DataFrame()

    # Colour map: current session = gold, others = blue
    clr_map = {lbl: ("gold" if lbl.startswith("\u2605") else "#4a90d9") for lbl, _ in sessions_data}

    # --- Grouped bar chart: Watt per interval ---
    long_w: list = []
    for lbl, df in sessions_data:
        for _, r in df.iterrows():
            w = r.get("_watt")
            if pd.notna(w):
                long_w.append({"Nr": int(r["Nr"]), "Session": lbl, "Watt": float(w)})
    if long_w:
        ldf = pd.DataFrame(long_w)
        fig_w = px.bar(ldf, x="Nr", y="Watt", color="Session", barmode="group",
                       color_discrete_map=clr_map, title="Watt per intervall", text="Watt")
        fig_w.update_traces(texttemplate="%{text:.0f} W", textposition="outside")
        fig_w.update_yaxes(range=[0, float(ldf["Watt"].max()) * 1.2])
        fig_w.update_xaxes(dtick=1)
        fig_w.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                             xaxis_title="Intervall nr.", yaxis_title="W")
        st.plotly_chart(fig_w, use_container_width=True)

    # --- Grouped bar chart: Pace per interval ---
    long_p: list = []
    for lbl, df in sessions_data:
        for _, r in df.iterrows():
            p = r.get("_pace_s")
            if pd.notna(p) and float(p) > 0:
                long_p.append({"Nr": int(r["Nr"]), "Session": lbl,
                                "pace_s": float(p), "Pace": _format_pace(p)})
    if long_p:
        lpdf = pd.DataFrame(long_p)
        fig_p = px.bar(lpdf, x="Nr", y="pace_s", color="Session", barmode="group",
                       color_discrete_map=clr_map,
                       title="Pace per intervall (lavere = raskere)", text="Pace")
        fig_p.update_traces(textposition="outside")
        pv = lpdf["pace_s"]
        pad = (float(pv.max()) - float(pv.min())) * 0.15 + 1.0
        fig_p.update_yaxes(range=[max(0.0, float(pv.min()) - pad), float(pv.max()) + pad],
                            title="s/500 m")
        fig_p.update_xaxes(dtick=1)
        fig_p.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                             xaxis_title="Intervall nr.")
        st.plotly_chart(fig_p, use_container_width=True)

    # --- Pivot tables per metric with best-value cell highlighted green ---
    metric_specs = [
        ("_watt",   "Watt (W) per intervall",   True,  lambda x: f"{float(x):.0f}"),
        ("_pace_s", "Pace per intervall",        False, _format_pace),
        ("_time_s", "Tid per intervall",         False, _format_duration),
        ("_puls",   "Puls (bpm) per intervall",  False, lambda x: f"{float(x):.0f}"),
        ("_spm",    "SPM per intervall",         False, lambda x: f"{float(x):.0f}"),
    ]

    def _make_row_hl(np_inner: pd.DataFrame, hb_inner: bool):
        def _hl(row_s: pd.Series) -> list:
            nr = row_s.name
            if nr not in np_inner.index:
                return [""] * len(row_s)
            raw_row = np_inner.loc[nr].dropna()
            if len(raw_row) < 2:
                return [""] * len(row_s)
            best = raw_row.max() if hb_inner else raw_row.min()
            return [
                "background-color: #90EE90; color: black"
                if (
                    c in np_inner.columns
                    and pd.notna(np_inner.loc[nr, c])
                    and float(np_inner.loc[nr, c]) == float(best)
                )
                else ""
                for c in row_s.index
            ]
        return _hl

    for col, title, higher_better, fmt_fn in metric_specs:
        num_piv = _num_pivot(col)
        if num_piv.empty or num_piv.isna().all().all():
            continue
        if num_piv.notna().any().sum() < 2:
            continue
        # Formatted display pivot
        disp_piv = num_piv.copy().astype(object)
        for c in disp_piv.columns:
            disp_piv[c] = num_piv[c].apply(lambda x: fmt_fn(x) if pd.notna(x) else "—")
        disp_piv.index.name = "Nr"
        styler = disp_piv.style.apply(_make_row_hl(num_piv, higher_better), axis=1)
        st.caption(title)
        st.dataframe(styler, use_container_width=True)


def _render_similar_sessions(det_row: pd.Series, det_df: pd.DataFrame) -> None:
    """Find and render similar sessions with side-by-side comparison."""
    if det_df.empty:
        return

    def _closeness_score(candidate: pd.Series, target: pd.Series, col: str, rel_tol: float | None = None, abs_tol: float | None = None) -> float:
        cand_val = pd.to_numeric(pd.Series([candidate.get(col)]), errors="coerce").iloc[0]
        targ_val = pd.to_numeric(pd.Series([target.get(col)]), errors="coerce").iloc[0]
        if pd.isna(cand_val) or pd.isna(targ_val):
            return 0.0
        diff = abs(float(cand_val) - float(targ_val))
        if abs_tol is not None and abs_tol > 0:
            scale = abs_tol
        elif rel_tol is not None and rel_tol > 0:
            scale = max(abs(float(targ_val)) * rel_tol, 1e-9)
        else:
            return 0.0
        return max(0.0, 1.0 - (diff / scale))

    st.divider()
    st.subheader("📊 Liknende økter")

    det_machine = det_row.get("machine_type")
    det_fmt = str(det_row.get("workout_format", ""))
    det_lid = det_row.get("log_id")
    det_dist = det_row.get("work_distance_m")
    det_time = det_row.get("work_time_s")
    det_benchmark = det_row.get("benchmark_name")
    det_season = det_row.get("season")

    sim = det_df.copy()
    if det_machine:
        sim = sim[sim["machine_type"] == det_machine]
    sim = sim[sim["workout_format"] == det_fmt]

    # Narrow by distance/time for more relevant matches
    if det_fmt == "Distanse" and pd.notna(det_dist) and float(det_dist) > 0:
        tol = float(det_dist) * 0.20
        sim = sim[(sim["work_distance_m"] - float(det_dist)).abs() <= tol]
    elif det_fmt == "Tid" and pd.notna(det_time) and float(det_time) > 0:
        tol = float(det_time) * 0.20
        sim = sim[(sim["work_time_s"] - float(det_time)).abs() <= tol]
    elif det_fmt == "Intervall" and pd.notna(det_dist) and float(det_dist) > 0:
        tol = float(det_dist) * 0.25
        sim = sim[(sim["work_distance_m"] - float(det_dist)).abs() <= tol]

    # Exclude current session
    try:
        if pd.notna(det_lid):
            sim = sim[sim["log_id"] != det_lid]
    except Exception:
        pass

    if not sim.empty:
        sim = sim.copy()
        sim["similarity_score"] = 0.0
        if det_benchmark:
            sim["similarity_score"] += np.where(sim["benchmark_name"].eq(det_benchmark), 30.0, 0.0)
        if det_season:
            sim["similarity_score"] += np.where(sim["season"].eq(det_season), 12.0, 0.0)

        sim["similarity_score"] += sim.apply(
            lambda r: 20.0 * _closeness_score(r, det_row, "work_distance_m", rel_tol=0.20),
            axis=1,
        )
        sim["similarity_score"] += sim.apply(
            lambda r: 20.0 * _closeness_score(r, det_row, "work_time_s", rel_tol=0.20),
            axis=1,
        )
        sim["similarity_score"] += sim.apply(
            lambda r: 12.0 * _closeness_score(r, det_row, "pace_s_500", abs_tol=6.0),
            axis=1,
        )
        sim["similarity_score"] += sim.apply(
            lambda r: 10.0 * _closeness_score(r, det_row, "avg_hr", abs_tol=10.0),
            axis=1,
        )
        sim["similarity_score"] += sim.apply(
            lambda r: 10.0 * _closeness_score(r, det_row, "avg_watts", rel_tol=0.15),
            axis=1,
        )
        sim["similarity_score"] = sim["similarity_score"].round(1)

    sim = sim.sort_values(["similarity_score", "date"], ascending=[False, False]).head(30).reset_index(drop=False)

    if sim.empty:
        st.info("Ingen liknende økter funnet i valgt datasett/filter.")
        return

    st.caption(f"Viser {len(sim)} liknende {det_fmt.lower()}-økter, rangert etter likhetsscore. Velg en eller flere for sammenligning.")

    def _fmt_date(d):
        try:
            return pd.Timestamp(d).strftime("%d.%m.%Y")
        except Exception:
            return str(d)[:10]

    sim_display = pd.DataFrame({
        "Likhet": sim["similarity_score"].apply(lambda x: round(float(x), 1) if pd.notna(x) else None),
        "Dato": sim["date"].map(_fmt_date),
        "Tid": sim["work_time_s"].map(_format_duration),
        "Distanse (km)": sim["work_distance_km"].apply(lambda x: round(float(x), 1) if pd.notna(x) else None),
        "Pace": sim["pace_s_500"].map(_format_pace),
        "Watt": sim["avg_watts"].apply(lambda x: round(float(x), 1) if pd.notna(x) else None),
        "W/kg": sim["w_kg"].apply(lambda x: round(float(x), 1) if pd.notna(x) else None),
        "Puls": sim["avg_hr"].apply(lambda x: round(float(x), 1) if pd.notna(x) else None),
        "Beskrivelse": sim["description"].fillna("").str[:50],
    })

    sim_sel = st.dataframe(
        sim_display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="similar_sessions_sel",
    )

    sel_rows = sim_sel.selection.rows

    btn_cols = st.columns([1, 1, 6])
    if len(sel_rows) == 1:
        with btn_cols[0]:
            if st.button("→ Åpne økten", key="open_similar_btn"):
                st.session_state["_detail_row"] = sim.iloc[sel_rows[0]]
                st.rerun()
    if sel_rows:
        with btn_cols[1]:
            if st.button("Nullstill valg", key="clear_similar_btn"):
                st.rerun()

    if not sel_rows:
        return

    # --- Comparison table ---
    st.markdown("#### Sammenligning")

    def _row_to_cmp(r: pd.Series, label: str) -> dict:
        return {
            "Session": label,
            "Dato": _fmt_date(r.get("date", "")),
            "Tid": _format_duration(r.get("work_time_s")),
            "Distanse (km)": round(float(r["work_distance_km"]), 1) if pd.notna(r.get("work_distance_km")) else None,
            "Pace": _format_pace(r.get("pace_s_500")),
            "Watt": round(float(r["avg_watts"]), 1) if pd.notna(r.get("avg_watts")) else None,
            "W/kg": round(float(r["w_kg"]), 1) if pd.notna(r.get("w_kg")) else None,
            "Puls": round(float(r["avg_hr"]), 1) if pd.notna(r.get("avg_hr")) else None,
            "SPM": round(float(r["stroke_rate"]), 1) if pd.notna(r.get("stroke_rate")) else None,
            "Vekt (kg)": round(float(r["vekt"]), 1) if pd.notna(r.get("vekt")) else None,
        }

    det_label = f"★ {_fmt_date(det_row.get('date', ''))}"
    cmp_rows = [_row_to_cmp(det_row, det_label)]
    for i in sel_rows:
        if i < len(sim):
            r = sim.iloc[i]
            cmp_rows.append(_row_to_cmp(r, _fmt_date(r.get("date", ""))))

    cmp_df = pd.DataFrame(cmp_rows)
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    # --- Bar charts for key numeric metrics ---
    chart_metrics = [
        ("Watt", "W"),
        ("W/kg", "W/kg"),
        ("Puls", "bpm"),
        ("Distanse (km)", "km"),
    ]
    chart_pairs = [
        m for m in chart_metrics
        if m[0] in cmp_df.columns
        and cmp_df[m[0]].notna().sum() >= 2
    ]

    if chart_pairs:
        chart_cols = st.columns(min(2, len(chart_pairs)))
        for ci, (metric, unit) in enumerate(chart_pairs):
            data = cmp_df[["Session", metric]].dropna(subset=[metric]).copy()
            data[metric] = pd.to_numeric(data[metric], errors="coerce")
            data = data.dropna(subset=[metric])
            if data.empty:
                continue
            with chart_cols[ci % 2]:
                fig = px.bar(data, x="Session", y=metric, text=metric, title=metric)
                fig.update_traces(
                    texttemplate=f"%{{text:.1f}} {unit}",
                    textposition="outside",
                    marker_color=[
                        "gold" if s.startswith("★") else "rgba(79,124,255,0.85)"
                        for s in data["Session"]
                    ],
                )
                fig.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=30, b=60),
                    xaxis_title="",
                    yaxis_title=unit,
                    showlegend=False,
                )
                max_val = float(data[metric].max())
                fig.update_yaxes(range=[0, max_val * 1.22])
                fig.update_xaxes(tickangle=-25)
                st.plotly_chart(fig, use_container_width=True)

    # --- Per-interval comparison ---
    _render_interval_comparison(
        [det_row] + [sim.iloc[i] for i in sel_rows if i < len(sim)],
        [det_label] + [_fmt_date(sim.iloc[i].get("date", "")) for i in sel_rows if i < len(sim)],
    )


def _render_lactate_analysis_tab(filtered: pd.DataFrame) -> None:
    st.subheader("Laktatanalyse")

    lactate_df = filtered[filtered["has_lactate"]].copy() if "has_lactate" in filtered.columns else pd.DataFrame()
    if lactate_df.empty:
        st.info("Ingen drag eller splitter med registrert laktat i valgt filter.")
        return

    lactate_df["analysis_session_type"] = np.where(
        lactate_df["benchmark_name"].fillna("").astype(str).str.strip() != "",
        lactate_df["benchmark_name"].fillna("").astype(str).str.strip(),
        lactate_df["description"].fillna("").astype(str).str.strip(),
    )

    session_type_options = [
        "Alle",
        *sorted([v for v in lactate_df["analysis_session_type"].dropna().unique().tolist() if str(v).strip()], key=str.lower),
    ]
    lactate_avg_range = _safe_range(lactate_df["interval_lactate_avg"])

    c1, c2 = st.columns([2, 2])
    with c1:
        selected_session_type = st.selectbox("Økttype", session_type_options, key="lactate_analysis_session_type")
    with c2:
        if lactate_avg_range is None:
            selected_lactate_range = None
        else:
            l_lo, l_hi = lactate_avg_range
            if float(l_lo) == float(l_hi):
                selected_lactate_range = (float(l_lo), float(l_hi))
                st.number_input(
                    "Snittlaktat",
                    min_value=float(l_lo),
                    max_value=float(l_hi),
                    value=float(l_lo),
                    step=0.1,
                    key="lactate_analysis_average_range_single",
                    disabled=True,
                )
            else:
                selected_lactate_range = st.slider(
                    "Snittlaktat",
                    min_value=float(l_lo),
                    max_value=float(l_hi),
                    value=(float(l_lo), float(l_hi)),
                    step=0.1,
                    key="lactate_analysis_average_range",
                )

    analysis_df = lactate_df.copy()
    if selected_session_type != "Alle":
        analysis_df = analysis_df[analysis_df["analysis_session_type"] == selected_session_type]
    if selected_lactate_range is not None:
        lo, hi = selected_lactate_range
        analysis_df = analysis_df[
            analysis_df["interval_lactate_avg"].notna()
            & (analysis_df["interval_lactate_avg"] >= lo)
            & (analysis_df["interval_lactate_avg"] <= hi)
        ]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Økter med laktat", f"{int(len(analysis_df))}")
    m2.metric("Registrerte drag/splitter", f"{int(analysis_df['interval_lactate_count'].sum())}")
    latest_row = analysis_df.dropna(subset=["date"]).sort_values("date", ascending=False).head(1)
    latest_label = latest_row["date"].dt.strftime("%d.%m.%Y").iloc[0] if not latest_row.empty else "—"
    m3.metric("Siste laktatøkt", latest_label)
    mean_lactate = analysis_df["interval_lactate_avg"].dropna().mean() if analysis_df["interval_lactate_avg"].notna().any() else float("nan")
    m4.metric("Snittlaktat", _format_lactate(mean_lactate))

    st.markdown("#### Terskeltester")
    threshold_tests = analysis_df[analysis_df["is_threshold_test"]].copy()
    if threshold_tests.empty:
        st.caption("Marker en økt som terskeltest i øktdetaljene for å beregne watt og W/kg ved valgt laktatnivå.")
    else:
        protocol_options = sorted(
            [name for name in threshold_tests["threshold_protocol_name"].dropna().unique().tolist() if str(name).strip()],
            key=str.casefold,
        )
        test_col, target_col = st.columns([2, 1])
        with test_col:
            selected_protocol = st.selectbox("Testoppsett", protocol_options, key="threshold_analysis_protocol")
        with target_col:
            target_lactate = st.number_input(
                "Laktatmål (mmol/L)",
                min_value=0.5,
                max_value=15.0,
                value=4.0,
                step=0.1,
                key="threshold_analysis_target_lactate",
            )

        protocol_tests = threshold_tests[threshold_tests["threshold_protocol_name"] == selected_protocol].copy()
        results: list[dict[str, object]] = []
        for _, test_row in protocol_tests.iterrows():
            metrics = _interpolate_threshold_metrics(test_row, float(target_lactate))
            if metrics is None:
                continue
            weight = pd.to_numeric(pd.Series([test_row.get("vekt")]), errors="coerce").iloc[0]
            watts = metrics["watts"]
            results.append(
                {
                    "date": test_row.get("date"),
                    "description": test_row.get("description", ""),
                    "watts": watts,
                    "w_kg": watts / float(weight) if pd.notna(weight) and float(weight) > 0 else np.nan,
                    "heart_rate": metrics["heart_rate"],
                    "pace_s_500": metrics["pace_s_500"],
                    "weight": weight,
                    "source_row": test_row,
                }
            )
        results_df = pd.DataFrame(results)
        if results_df.empty:
            st.info("Ingen av testene har to påfølgende laktatmålinger som omslutter valgt målverdi.")
        else:
            results_df = results_df.sort_values("date", ascending=False)
            latest_threshold = results_df.iloc[0]
            t1, t2, t3, t4 = st.columns(4)
            t1.metric(f"Watt ved {float(target_lactate):.1f} mmol/L", f"{float(latest_threshold['watts']):.1f} W")
            t2.metric("W/kg", f"{float(latest_threshold['w_kg']):.2f}" if pd.notna(latest_threshold["w_kg"]) else "Mangler vekt")
            t3.metric("Puls", f"{float(latest_threshold['heart_rate']):.0f} bpm" if pd.notna(latest_threshold["heart_rate"]) else "—")
            t4.metric("Pace", _format_pace(latest_threshold["pace_s_500"]))

            history = results_df.copy()
            history["Dato"] = pd.to_datetime(history["date"]).dt.strftime("%Y-%m-%d")
            history["Watt"] = history["watts"].round(1)
            history["W/kg"] = history["w_kg"].round(1)
            history["Puls"] = history["heart_rate"].round(1)
            history["Pace"] = history["pace_s_500"].map(_format_pace)
            history["Vekt (kg)"] = history["weight"].round(1)
            st.dataframe(
                history[["Dato", "Watt", "W/kg", "Puls", "Pace", "Vekt (kg)", "description"]].rename(columns={"description": "Beskrivelse"}),
                use_container_width=True,
                hide_index=True,
            )

            history_plot = results_df.dropna(subset=["date"]).sort_values("date")
            if len(history_plot) >= 2:
                fig_history = go.Figure()
                fig_history.add_trace(go.Scatter(
                    x=history_plot["date"],
                    y=history_plot["watts"],
                    mode="lines+markers",
                    name="Watt",
                    hovertemplate="%{x|%d.%m.%Y}<br>%{y:.1f} W<extra></extra>",
                ))
                fig_history.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Dato",
                    yaxis_title=f"Watt ved {float(target_lactate):.1f} mmol/L",
                )
                st.plotly_chart(fig_history, use_container_width=True)

            fig_profile = go.Figure()
            for _, test_row in protocol_tests.iterrows():
                intervals = test_row.get("_workout_intervals")
                items = intervals if isinstance(intervals, list) and intervals else test_row.get("_workout_splits")
                lactates = test_row.get("interval_lactates")
                if not isinstance(items, list) or not isinstance(lactates, dict):
                    continue
                points = []
                for index, item in enumerate(items, start=1):
                    if not isinstance(item, dict):
                        continue
                    lactate = _normalize_lactate_value(lactates.get(str(index)))
                    time_tenths, distance_m = item.get("time"), item.get("distance")
                    if lactate is None or time_tenths is None or distance_m is None or float(distance_m) <= 0:
                        continue
                    pace = (float(time_tenths) / 10.0) / float(distance_m) * 500.0
                    points.append((2.8 / (pace / 500.0) ** 3, lactate))
                if points:
                    test_date = pd.Timestamp(test_row["date"]).strftime("%d.%m.%Y") if pd.notna(test_row.get("date")) else "Udatert"
                    fig_profile.add_trace(go.Scatter(
                        x=[point[0] for point in points],
                        y=[point[1] for point in points],
                        mode="lines+markers",
                        name=test_date,
                    ))
            if fig_profile.data:
                fig_profile.add_hline(y=float(target_lactate), line_dash="dash", line_color="#d97706")
                fig_profile.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Watt",
                    yaxis_title="Laktat (mmol/L)",
                )
                st.plotly_chart(fig_profile, use_container_width=True)

    st.divider()
    st.markdown("#### Laktatøkter")
    table_df = analysis_df.dropna(subset=["date"]).sort_values("date", ascending=False).copy()
    if table_df.empty:
        st.info("Ingen økter matcher valgene i laktatanalysen.")
    else:
        table_df["date_display"] = table_df["date"].dt.strftime("%Y-%m-%d %H:%M")
        table_df["interval_lactate_avg_display"] = table_df["interval_lactate_avg"].map(_format_lactate)
        table_df["interval_lactate_max_display"] = table_df["interval_lactate_max"].map(_format_lactate)
        table_df["interval_lactate_last_display"] = table_df["interval_lactate_last"].map(_format_lactate)
        table_df["interval_lactate_delta_display"] = table_df["interval_lactate_delta_last_first"].map(_format_lactate)
        table_view = table_df[
            [
                "date_display",
                "description",
                "analysis_session_type",
                "workout_format",
                "interval_lactate_count",
                "interval_lactate_avg_display",
                "interval_lactate_max_display",
                "interval_lactate_last_display",
                "interval_lactate_delta_display",
                "avg_hr",
                "avg_watts",
                "pace_s_500",
            ]
        ].rename(
            columns={
                "date_display": "Dato",
                "description": "Beskrivelse",
                "analysis_session_type": "Økttype",
                "workout_format": "Format",
                "interval_lactate_count": "Drag/splitter",
                "interval_lactate_avg_display": "Snitt laktat",
                "interval_lactate_max_display": "Maks laktat",
                "interval_lactate_last_display": "Siste laktat",
                "interval_lactate_delta_display": "Endring",
                "avg_hr": "Puls",
                "avg_watts": "Watt",
                "pace_s_500": "Pace",
            }
        )
        table_view["Pace"] = table_df["pace_s_500"].map(_format_pace)
        selection = st.dataframe(
            _table_display(table_view),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="lactate_analysis_table",
        )
        sel_rows = selection.selection.rows
        if sel_rows:
            st.session_state["_detail_row"] = table_df.iloc[sel_rows[0]]
            st.session_state["_detail_df"] = filtered
            st.rerun()

    st.divider()
    st.markdown("#### Radanalyse")
    interval_df = analysis_df[analysis_df["interval_lactate_count"] > 0].dropna(subset=["date"]).sort_values("date", ascending=False).copy()
    if interval_df.empty:
        st.info("Ingen økter med registrert laktat på drag eller splitter i valgt utvalg.")
        return

    interval_options = interval_df["label"].dropna().tolist()
    selected_interval_label = st.selectbox(
        "Velg økt",
        interval_options,
        key="lactate_analysis_interval_session",
    )
    selected_interval_row = interval_df[interval_df["label"] == selected_interval_label].head(1)
    if selected_interval_row.empty:
        return
    chosen_row = selected_interval_row.iloc[0]
    interval_items = chosen_row.get("_workout_intervals")
    is_interval = isinstance(interval_items, list) and bool(interval_items)
    if not is_interval:
        interval_items = chosen_row.get("_workout_splits")
    if not isinstance(interval_items, list) or not interval_items:
        st.info("Valgt økt mangler drag- eller splitstruktur i datagrunnlaget.")
        return

    interval_workout_df = _parse_workout_to_df(interval_items, is_interval=is_interval)
    if interval_workout_df is None or interval_workout_df.empty:
        st.info("Kunne ikke bygge intervalltabell for valgt økt.")
        return

    interval_lactates = chosen_row.get("interval_lactates") if isinstance(chosen_row.get("interval_lactates"), dict) else {}
    interval_workout_df = interval_workout_df.copy()
    interval_workout_df["Laktat"] = interval_workout_df["Nr"].map(
        lambda nr: interval_lactates.get(str(int(nr))) if pd.notna(nr) else None
    )
    interval_workout_df["Laktat Δ"] = pd.to_numeric(interval_workout_df["Laktat"], errors="coerce").diff()
    st.dataframe(_table_display(interval_workout_df), use_container_width=True, hide_index=True)

    plot_df = interval_workout_df.dropna(subset=["Laktat"]).copy()
    if not plot_df.empty:
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(
            x=plot_df["Nr"],
            y=plot_df["Laktat"],
            mode="lines+markers",
            name="Laktat",
        ))
        fig_iv.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Drag nr.",
            yaxis_title="mmol/L",
            legend_title="",
        )
        st.plotly_chart(fig_iv, use_container_width=True)

        if "Watt" in plot_df.columns and pd.to_numeric(plot_df["Watt"], errors="coerce").notna().any():
            fig_watt = px.bar(
                plot_df.dropna(subset=["Watt"]),
                x="Nr",
                y="Watt",
                text="Watt",
            )
            fig_watt.update_traces(texttemplate="%{text:.0f} W", textposition="outside")
            fig_watt.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Drag nr.",
                yaxis_title="W",
            )
            st.plotly_chart(fig_watt, use_container_width=True)


def main() -> None:
    app_name = _get_app_name()
    st.set_page_config(page_title=app_name, layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            max-width: 1280px;
            padding-left: 24px;
            padding-right: 24px;
        }
        [data-testid="stSidebar"] {
            background: #f7f9fa;
            border-right: 1px solid #dbe2e7;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-baseweb="select"] {
            color: #111827;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #dbe2e7;
            border-radius: 6px;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            font-size: 0.85rem;
            font-weight: 700;
        }
        [data-testid="stSidebar"] .stButton > button {
            min-height: 32px;
            padding: 0.25rem 0.6rem;
            font-size: 0.82rem;
        }
        [data-testid="stSidebar"] .st-key-api_base_url_input [data-baseweb="input"],
        [data-testid="stSidebar"] .st-key-api_token_input [data-baseweb="input"],
        [data-testid="stSidebar"] .st-key-api_type_filter_input [data-baseweb="select"] {
            border: 2px solid #dc2626;
            border-radius: 5px;
            box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.12);
        }
        [data-testid="stSidebar"] .st-key-api_base_url_input [data-baseweb="input"]:focus-within,
        [data-testid="stSidebar"] .st-key-api_token_input [data-baseweb="input"]:focus-within,
        [data-testid="stSidebar"] .st-key-api_type_filter_input [data-baseweb="select"]:focus-within {
            border-color: #991b1b;
            box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.22);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Detail page: shown instead of main content when a session is selected
    # ------------------------------------------------------------------ #
    if "_detail_row" in st.session_state:
        _det_row = st.session_state["_detail_row"]
        _det_df = st.session_state.get("_detail_df", pd.DataFrame())
        if st.button("← Tilbake"):
            del st.session_state["_detail_row"]
            st.session_state.pop("_detail_df", None)
            st.rerun()
        _render_session_detail(_det_row)
        _render_similar_sessions(_det_row, _det_df)
        return

    title_col, edit_name_col = st.columns([7, 1])
    with title_col:
        st.title(app_name)
    with edit_name_col:
        with st.popover("Rediger navn"):
            new_app_name = st.text_input("Portalnavn", value=app_name)
            if st.button("Lagre navn", key="save_app_name_btn"):
                _update_settings({"app_name": new_app_name.strip() or _APP_NAME_DEFAULT})
                st.rerun()

    with st.sidebar:
        with st.expander("Datakilde", expanded=False):
            data_source = st.radio(
                "Velg datakilde",
                ["CSV (system/Data)", "Concept2 API"],
                index=0,
                horizontal=True,
            )

            csv_files_on_disk = sorted(_find_data_dir(Path(__file__).resolve()).glob("concept2-season-*.csv"))
            if data_source == "CSV (system/Data)":
                uploaded_csv_files = st.file_uploader(
                    "Last opp CSV",
                    type=["csv"],
                    accept_multiple_files=True,
                    help="Bruk en Concept2 Logbook CSV-eksport med Log ID, Date, Description, Pace og Avg Watts.",
                )
                if uploaded_csv_files:
                    st.caption(f"{len(uploaded_csv_files)} opplastede filer")
                    for uploaded in uploaded_csv_files:
                        st.caption(uploaded.name)
                elif csv_files_on_disk:
                    st.caption(f"{len(csv_files_on_disk)} CSV-filer i system/Data")
                    for csv_path in csv_files_on_disk:
                        file_name_col, delete_col = st.columns([12, 1])
                        file_name_col.caption(csv_path.name)
                        if delete_col.button(
                            "×",
                            key=f"csv_delete_{csv_path.name}",
                            help="Slett CSV",
                            type="primary",
                            width="content",
                        ):
                            _confirm_csv_file_deletion(str(csv_path))
                else:
                    st.warning("Fant ingen concept2-season-*.csv i system/Data. Last opp en CSV fra Concept2 Logbook.")
            else:
                api_base_url = st.text_input(
                    "API base URL",
                    value="https://log.concept2.com",
                    key="api_base_url_input",
                )
                api_type = st.selectbox(
                    "API type-filter",
                    options=[
                        "skierg",
                        "rower",
                        "bike",
                        "dynamic",
                        "slides",
                        "paddle",
                        "water",
                        "snow",
                        "rollerski",
                        "multierg",
                        "(ingen filter)",
                    ],
                    index=0,
                    key="api_type_filter_input",
                )
                api_type_filter = None if api_type == "(ingen filter)" else api_type

                api_token = st.text_input(
                    "Access token",
                    value="",
                    type="password",
                    key="api_token_input",
                )
                use_api_date_filter = st.checkbox(
                    "Begrens API dato",
                    value=False,
                    help="Hvis av: henter alle resultater (kan ta litt tid første gang).",
                )

                api_from_to: tuple[date_type, date_type] | None = None
                if use_api_date_filter:
                    today = pd.Timestamp.today().date()
                    default_from = date_type(today.year - 1, today.month, min(today.day, 28))
                    api_from_to = st.date_input("API dato", value=(default_from, today))

    try:
        if data_source == "Concept2 API":
            if not api_token.strip():
                st.info("Legg inn Concept2 access token for å hente data fra API. Portalen vises i tomtilstand imens.")
                df = _empty_concept2_history()
            else:
                from_d: date_type | None = None
                to_d: date_type | None = None
                if api_from_to is not None and len(api_from_to) == 2:
                    from_d, to_d = api_from_to

                api_request_signature = (
                    api_base_url,
                    api_token.strip(),
                    from_d.isoformat() if from_d else None,
                    to_d.isoformat() if to_d else None,
                    api_type_filter,
                )
                cached_signature = st.session_state.get("_api_df_signature")
                cached_df = st.session_state.get("_api_df")

                if cached_signature == api_request_signature and isinstance(cached_df, pd.DataFrame):
                    df = cached_df.copy()
                else:
                    api_progress = st.progress(0, text="Henter resultatliste fra Concept2 API …")

                    def _update_api_progress(loaded_count: int, total_count: int) -> None:
                        percent = 100 if total_count == 0 else round(loaded_count / total_count * 100)
                        api_progress.progress(
                            percent,
                            text=f"{loaded_count} av {total_count} treningsøkter er lastet inn. {percent} %",
                        )

                    try:
                        df = load_concept2_history_from_api(
                            base_url=api_base_url,
                            access_token=api_token.strip(),
                            from_date=from_d,
                            to_date=to_d,
                            type_filter=api_type_filter,
                            _progress_callback=_update_api_progress,
                        )
                    finally:
                        api_progress.empty()
                    st.session_state["_api_df_signature"] = api_request_signature
                    st.session_state["_api_df"] = df.copy()
                _ci = st.session_state.get("_api_cache_info")
                if _ci:
                    if _ci["new"] == 0:
                        st.caption(f"ℹ️ {_ci['total']:,} økter lastet fra lokal cache (ingen nye).")
                    else:
                        st.caption(f"ℹ️ {_ci['from_cache']:,} fra cache + {_ci['new']:,} nye hentet = {_ci['total']:,} økter totalt.")
                # Store API config so _render_session_detail can fetch stroke data
                st.session_state["_api_cfg"] = {
                    "base_url": api_base_url,
                    "access_token": api_token.strip(),
                }
        else:
            if uploaded_csv_files:
                df = _load_concept2_csv_from_uploads(uploaded_csv_files)
            else:
                csv_paths = tuple(str(path) for path in csv_files_on_disk)
                if not csv_paths:
                    st.info("Last opp en Concept2 Logbook CSV-eksport for å fylle portalen med dine egne data. Du ser nå tomtilstanden.")
                    df = _empty_concept2_history()
                else:
                    df = load_concept2_history_from_csv(csv_paths)
            st.session_state.pop("_api_df_signature", None)
            st.session_state.pop("_api_df", None)
            st.session_state.pop("_api_cfg", None)

    except Concept2ApiError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    df["season"] = df["date"].map(_training_season_label)
    df = _annotate_benchmark_status(df)
    df = _merge_lactate_columns(df)

    with st.sidebar:
        st.header("Filtre")

        seasons = sorted(
            df["season"].dropna().unique().tolist(),
            key=lambda s: int(str(s).split("/")[0]),
            reverse=True,
        ) if "season" in df.columns else []
        formats = sorted(df["workout_format"].dropna().unique().tolist()) if "workout_format" in df.columns else []
        date_min = df["date"].min() if "date" in df.columns else pd.NaT
        date_max = df["date"].max() if "date" in df.columns else pd.NaT

        default_filter_state: dict[str, object] = {
            "filter_seasons": seasons,
            "filter_formats": formats,
        }
        if not pd.isna(date_min) and not pd.isna(date_max):
            default_filter_state["filter_date_range"] = (date_min.date(), date_max.date())

        def range_for(col: str):
            return _safe_range(df[col]) if col in df.columns else None

        dist_rng = range_for("work_distance_km")
        time_rng = range_for("work_time_min")
        watts_rng = range_for("avg_watts")
        hr_rng = range_for("avg_hr")
        pace_rng = range_for("pace_s_500")
        interval_lactate_avg_rng = range_for("interval_lactate_avg")

        if dist_rng is not None:
            default_filter_state["filter_dist_km"] = dist_rng
        if time_rng is not None:
            default_filter_state["filter_time_min"] = time_rng
        if watts_rng is not None:
            default_filter_state["filter_watts"] = watts_rng
        if hr_rng is not None:
            default_filter_state["filter_hr"] = hr_rng
        if pace_rng is not None:
            default_filter_state["filter_pace_s"] = pace_rng
        if interval_lactate_avg_rng is not None:
            default_filter_state["filter_interval_lactate_avg"] = interval_lactate_avg_rng
        default_filter_state["filter_has_lactate"] = "Alle"

        filter_heading, reset_col = st.columns([4, 1])
        filter_heading.subheader("Filtre")
        if reset_col.button(
            "↺",
            key="clear_filters_btn",
            help="Nullstill filtre",
            type="tertiary",
            width="content",
        ):
            for state_key, default_value in default_filter_state.items():
                st.session_state[state_key] = default_value
            st.rerun()

        st.markdown("##### Grunnutvalg")
        selected_seasons = st.multiselect("Sesong", seasons, default=seasons, key="filter_seasons")
        selected_formats = st.multiselect("Øktformat", formats, default=formats, key="filter_formats")

        date_range = None
        if not pd.isna(date_min) and not pd.isna(date_max):
            date_range = st.date_input(
                "Dato",
                value=(date_min.date(), date_max.date()),
                min_value=date_min.date(),
                max_value=date_max.date(),
                key="filter_date_range",
            )

        def slider_for(col: str, label: str, step: float, key: str, rng: tuple[float, float] | None):
            if rng is None:
                return None
            lo, hi = rng
            if lo == hi:
                return (lo, hi)
            return st.slider(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=(float(lo), float(hi)),
                step=step,
                key=key,
            )

        with st.expander("Ytelse", expanded=False):
            dist_km = slider_for("work_distance_km", "Distanse (km)", step=0.1, key="filter_dist_km", rng=dist_rng)
            time_min = slider_for("work_time_min", "Tid (min)", step=0.5, key="filter_time_min", rng=time_rng)
            watts = slider_for("avg_watts", "Snitt W", step=1.0, key="filter_watts", rng=watts_rng)
            hr = slider_for("avg_hr", "Snittpuls", step=1.0, key="filter_hr", rng=hr_rng)

            if pace_rng is None:
                pace_s = None
            else:
                p_lo, p_hi = pace_rng
                pace_options = _numeric_slider_options(float(p_lo), float(p_hi), 0.1)
                pace_default = (
                    min(pace_options, key=lambda v: abs(v - float(p_lo))),
                    min(pace_options, key=lambda v: abs(v - float(p_hi))),
                )
                pace_s = st.select_slider(
                    "Pace (min:sek/500m)",
                    options=pace_options,
                    value=pace_default,
                    format_func=_format_pace,
                    key="filter_pace_s",
                )
                st.caption(f"Valgt pace: {_format_pace(pace_s[0])}–{_format_pace(pace_s[1])}")

        with st.expander("Laktat", expanded=False):
            has_lactate_filter = st.selectbox(
                "Laktat registrert",
                ["Alle", "Ja", "Nei"],
                key="filter_has_lactate",
            )
            interval_lactate_avg = slider_for(
                "interval_lactate_avg",
                "Snittlaktat (drag/splitt)",
                step=0.1,
                key="filter_interval_lactate_avg",
                rng=interval_lactate_avg_rng,
            )

    filtered = df.copy()

    if selected_seasons:
        filtered = filtered[filtered["season"].isin(selected_seasons)]

    if selected_formats:
        filtered = filtered[filtered["workout_format"].isin(selected_formats)]

    if date_range and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[(filtered["date"].dt.date >= start) & (filtered["date"].dt.date <= end)]

    def apply_range(col: str, rng):
        nonlocal filtered
        if rng is None or col not in filtered.columns:
            return
        lo, hi = rng
        filtered = filtered[(filtered[col] >= lo) & (filtered[col] <= hi)]

    apply_range("work_distance_km", dist_km)
    apply_range("work_time_min", time_min)
    apply_range("avg_watts", watts)
    apply_range("avg_hr", hr)
    apply_range("pace_s_500", pace_s)
    if has_lactate_filter == "Ja":
        filtered = filtered[filtered["has_lactate"]]
    elif has_lactate_filter == "Nei":
        filtered = filtered[~filtered["has_lactate"]]
    interval_lactate_filter_active = False
    if interval_lactate_avg is not None and interval_lactate_avg_rng is not None:
        rng_lo, rng_hi = interval_lactate_avg_rng
        selected_lo, selected_hi = interval_lactate_avg
        interval_lactate_filter_active = (
            abs(float(selected_lo) - float(rng_lo)) > 1e-9
            or abs(float(selected_hi) - float(rng_hi)) > 1e-9
        )
    if interval_lactate_filter_active:
        filtered = filtered[
            filtered["interval_lactate_avg"].notna()
            & (filtered["interval_lactate_avg"] >= float(interval_lactate_avg[0]))
            & (filtered["interval_lactate_avg"] <= float(interval_lactate_avg[1]))
        ]

    # Merge per-session weights: manual entry (session_weights.json) overrides;
    # otherwise use nearest measurement from vekter/weight.csv (within 14 days).
    _manual_weights = _load_weights()
    _weight_csv = _load_weight_csv()

    def _resolve_weight(r: pd.Series) -> tuple[float | None, str | None]:
        # Manual weight takes precedence (no source date shown for manual)
        _mk = _session_weight_key(r)
        _mw = _manual_weights.get(_mk)
        if _mw:
            return float(_mw), "manuell"
        # Fall back to nearest CSV measurement
        _sd = r.get("date")
        if _sd is None or (isinstance(_sd, float) and np.isnan(_sd)):
            return None, None
        return _match_weight(pd.Timestamp(_sd), _weight_csv, max_days=14)

    _resolved = filtered.apply(_resolve_weight, axis=1)
    filtered["vekt"] = _resolved.map(lambda x: x[0]).astype(float)
    filtered["vekt_dato"] = _resolved.map(lambda x: x[1] if x[1] and x[1] != "manuell" else "")
    filtered["w_kg"] = np.where(
        filtered["vekt"].notna()
        & filtered["avg_watts"].notna()
        & (filtered["vekt"] > 0),
        (filtered["avg_watts"] / filtered["vekt"]).round(2),
        np.nan,
    )
    _tags_map = _load_tags()
    filtered["tags"] = filtered.apply(
        lambda r: ", ".join(_tags_map.get(_session_weight_key(r), [])),
        axis=1,
    )
    _hf_load = _get_hf_maks()
    _zones_load = _get_zones()
    filtered["load_score"] = filtered.apply(
        lambda r: _session_load_score(r.get("avg_hr"), r.get("work_time_s"), _hf_load, _zones_load),
        axis=1,
    )
    filtered["analysis_session_type"] = np.where(
        filtered["benchmark_name"].fillna("").astype(str).str.strip() != "",
        filtered["benchmark_name"].fillna("").astype(str).str.strip(),
        filtered["description"].fillna("").astype(str).str.strip(),
    )

    tab_overview, tab_sessions, tab_leaderboard, tab_lactate, tab_settings = st.tabs(["Oversikt", "Økter", "Leaderboard", "Laktatanalyse", "⚙️ Innstillinger"])

    with tab_overview:
        st.subheader("Oversikt")
        st.caption(f"Viser {len(filtered):,} økter (etter filter)")

        st.markdown("#### Volum")

        c1, c2 = st.columns([1, 1])
        with c1:
            x_granularity = st.radio("X-akse", ["Uker", "Måneder", "År"], index=1, horizontal=True)
        with c2:
            y_metric = st.radio("Y-akse", ["Timer", "Distanse"], index=0, horizontal=True)

        base = filtered.dropna(subset=["date"]).copy()
        period_code = {"Uker": "W", "Måneder": "M", "År": "Y"}[x_granularity]
        base["period_start"] = base["date"].dt.to_period(period_code).dt.start_time

        agg = (
            base.groupby("period_start", as_index=False)
            .agg(
                sessions=("log_id", "count"),
                total_time_h=("work_time_s", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum()) / 3600.0),
                total_dist_km=("work_distance_km", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            )
            .sort_values("period_start")
        )

        metric_col = "total_time_h" if y_metric == "Timer" else "total_dist_km"
        y_title = "Timer" if y_metric == "Timer" else "km"
        x_title = {"Uker": "Uke", "Måneder": "Måned", "År": "År"}[x_granularity]

        if x_granularity == "Måneder":
            agg["period_label"] = agg["period_start"].dt.month.map(MONTH_NO).fillna("") + " " + agg["period_start"].dt.strftime("%y")
        elif x_granularity == "År":
            agg["period_label"] = agg["period_start"].dt.strftime("%Y")
        else:
            iso_week = agg["period_start"].dt.isocalendar().week.astype(int).astype(str)
            yy = agg["period_start"].dt.strftime("%y")
            agg["period_label"] = "Uke " + iso_week + " " + yy

        labels_in_order = agg["period_label"].tolist()

        unit_suffix = " t" if y_metric == "Timer" else " km"
        agg["value_text"] = pd.to_numeric(agg[metric_col], errors="coerce").fillna(0).map(lambda v: f"{v:.1f}{unit_suffix}")

        fig = px.bar(
            agg,
            x="period_label",
            y=metric_col,
            hover_data=["sessions", "total_time_h", "total_dist_km"],
            category_orders={"period_label": labels_in_order},
        )

        if not agg.empty:
            values = pd.to_numeric(agg[metric_col], errors="coerce").fillna(0).to_numpy()
            max_pos = int(values.argmax())

            colors = ["rgba(31,119,180,0.85)" for _ in range(len(agg))]
            colors[max_pos] = "gold"
            fig.update_traces(marker_color=colors)

            fig.update_traces(text=agg["value_text"], textposition="outside", cliponaxis=False)

            max_x = agg["period_label"].iloc[max_pos]
            max_y = float(values[max_pos])
            bump = max(0.05 * max_y, 0.1)
            fig.add_annotation(
                x=max_x,
                y=max_y + bump,
                text="★",
                showarrow=False,
                font=dict(size=20, color="gold"),
            )

            fig.update_yaxes(range=[0, max_y + bump * 4])

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title=y_title,
            xaxis_title=x_title,
        )
        if x_granularity in ("Uker", "Måneder"):
            fig.update_xaxes(tickangle=-35)

        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        total_time_s = float(filtered["work_time_s"].fillna(0).sum()) if "work_time_s" in filtered.columns else 0.0
        total_dist_km = float(filtered["work_distance_km"].fillna(0).sum()) if "work_distance_km" in filtered.columns else 0.0

        avg_watts_val = float(filtered["avg_watts"].dropna().mean()) if "avg_watts" in filtered.columns else float("nan")
        avg_pace_val = float(filtered["pace_s_500"].dropna().mean()) if "pace_s_500" in filtered.columns else float("nan")

        col1.metric("Økter", f"{len(filtered):,}")
        col2.metric("Total tid", _format_duration(total_time_s))
        col3.metric("Total distanse", f"{total_dist_km:,.1f} km")
        col4.metric("Snitt watt", "" if np.isnan(avg_watts_val) else f"{avg_watts_val:.0f}")
        col5.metric("Snitt pace", "" if np.isnan(avg_pace_val) else _format_pace(avg_pace_val))

        if not filtered.empty:
            st.divider()

            st.markdown("#### Sesongsammenligning")
            compare_seasons = sorted(filtered["season"].dropna().unique().tolist(), key=_season_sort_key, reverse=True)
            if len(compare_seasons) < 2:
                st.info("Velg minst to sesonger i filteret for å sammenligne sesonger.")
            else:
                sc1, sc2 = st.columns(2)
                with sc1:
                    selected_seasons = st.multiselect(
                        "Sesonger",
                        compare_seasons,
                        default=compare_seasons,
                        key="season_compare_multi",
                        help="Velg så mange sesonger du vil sammenligne. Alle valgte sesonger vises i tabellen og figuren.",
                    )
                with sc2:
                    reference_season = st.selectbox(
                        "Referansesesong",
                        selected_seasons or compare_seasons,
                        index=0,
                        key="season_compare_reference",
                        help="Alle differanser beregnes mot denne sesongen ved samme punkt i treningsåret.",
                    )

                if len(selected_seasons) < 2:
                    st.info("Velg minst to sesonger for å sammenligne sesonger.")
                else:
                    cmp_source = filtered[filtered["season"].isin(selected_seasons)].copy()
                    cmp_source["season_day"] = cmp_source["date"].map(_training_season_day)

                    ref_rows = cmp_source[cmp_source["season"] == reference_season].dropna(subset=["date", "season_day"]).copy()
                    ref_cutoff_day = int(ref_rows["season_day"].max()) if not ref_rows.empty else None
                    ref_cutoff_date = ref_rows["date"].max() if not ref_rows.empty else pd.NaT

                    def _aggregate_totals(data: pd.DataFrame) -> dict[str, float | int | None]:
                        best_5k = _best_5k_time(data)
                        return {
                            "Økter": int(len(data)),
                            "Timer": round(float(data["work_time_s"].fillna(0).sum()) / 3600.0, 1),
                            "Distanse (km)": round(float(data["work_distance_km"].fillna(0).sum()), 1),
                            "Snitt watt": round(float(data["avg_watts"].dropna().mean()), 0) if data["avg_watts"].notna().any() else None,
                            "Best 5k": _format_duration(best_5k) if best_5k is not None else "—",
                            "Load": round(float(data["load_score"].fillna(0).sum()), 0),
                        }

                    def _best_5k_time(data: pd.DataFrame) -> float | None:
                        cand = data[
                            data["work_distance_m"].between(4990, 5010)
                            & data["work_time_s"].notna()
                            & (data["workout_format"] != "Intervall")
                        ]
                        if cand.empty:
                            return None
                        return float(pd.to_numeric(cand["work_time_s"], errors="coerce").min())

                    ref_same_time_df = (
                        cmp_source[
                            (cmp_source["season"] == reference_season)
                            & cmp_source["season_day"].notna()
                            & (cmp_source["season_day"] <= ref_cutoff_day)
                        ].copy()
                        if ref_cutoff_day is not None else pd.DataFrame(columns=cmp_source.columns)
                    )
                    ref_same_time_time = round(float(ref_same_time_df["work_time_s"].fillna(0).sum()) / 3600.0, 1)
                    ref_same_time_dist = round(float(ref_same_time_df["work_distance_km"].fillna(0).sum()), 1)

                    rows = []
                    for season_name in selected_seasons:
                        season_df = cmp_source[cmp_source["season"] == season_name].copy()
                        same_time_df = (
                            season_df[
                                season_df["season_day"].notna()
                                & (season_df["season_day"] <= ref_cutoff_day)
                            ].copy()
                            if ref_cutoff_day is not None else pd.DataFrame(columns=season_df.columns)
                        )
                        row = {
                            "Sesong": season_name,
                            **_aggregate_totals(season_df),
                            "Økter samme tid": int(len(same_time_df)),
                            "Timer samme tid": round(float(same_time_df["work_time_s"].fillna(0).sum()) / 3600.0, 1),
                            "Distanse samme tid (km)": round(float(same_time_df["work_distance_km"].fillna(0).sum()), 1),
                        }
                        row["Diff timer vs referanse"] = round(float(row["Timer samme tid"]) - ref_same_time_time, 1)
                        row["Diff distanse vs referanse (km)"] = round(float(row["Distanse samme tid (km)"]) - ref_same_time_dist, 1)
                        rows.append(row)

                    season_cmp_df = pd.DataFrame(rows)
                    st.dataframe(season_cmp_df, use_container_width=True, hide_index=True)

                    if pd.notna(ref_cutoff_date):
                        st.caption(
                            f"Sammenligning ved samme punkt i treningssesongen som {reference_season}: "
                            f"til og med {pd.Timestamp(ref_cutoff_date).strftime('%d.%m.%Y')} "
                            f"(sesongdag {ref_cutoff_day + 1}). Positive differanser betyr mer volum enn referansesesongen på samme tidspunkt."
                        )

                    chart_metrics = [
                        "Økter",
                        "Timer",
                        "Distanse (km)",
                        "Snitt watt",
                        "Load",
                        "Økter samme tid",
                        "Timer samme tid",
                        "Distanse samme tid (km)",
                        "Diff timer vs referanse",
                        "Diff distanse vs referanse (km)",
                    ]
                    chart_choice = st.selectbox("Sammenlign metrikk", chart_metrics, index=6, key="season_compare_metric")
                    chart_df = season_cmp_df[["Sesong", chart_choice]].dropna(subset=[chart_choice]).copy()
                    chart_df[chart_choice] = pd.to_numeric(chart_df[chart_choice], errors="coerce")
                    chart_df = chart_df.dropna(subset=[chart_choice])
                    if len(chart_df) >= 2:
                        fig_sc = px.bar(
                            chart_df,
                            x="Sesong",
                            y=chart_choice,
                            text=chart_choice,
                            color="Sesong",
                        )
                        fig_sc.update_traces(texttemplate="%{text}", textposition="outside")
                        fig_sc.update_layout(
                            height=320,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="",
                            yaxis_title=chart_choice,
                            showlegend=False,
                        )
                        st.plotly_chart(fig_sc, use_container_width=True)

            st.markdown("#### Scatter: pace vs watt")
            tmp = filtered.dropna(subset=["pace_s_500", "avg_watts"])
            if not tmp.empty:
                fig = px.scatter(
                    tmp,
                    x="avg_watts",
                    y="pace_s_500",
                    color="workout_format",
                    size=tmp["work_distance_km"].fillna(0).clip(lower=0.1),
                    hover_data=["date", "description", "work_distance_km", "work_time_min", "avg_hr"],
                )
                y_vals = pd.to_numeric(tmp["pace_s_500"], errors="coerce").dropna()
                y_pad = (y_vals.max() - y_vals.min()) * 0.08 + 1
                fig.update_yaxes(range=[y_vals.max() + y_pad, y_vals.min() - y_pad])
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360, yaxis_title="sek/500m")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Mangler pace/watt for å lage scatter.")

            st.divider()
            st.markdown("#### Tid i pulssone")
            _hf_ov = _get_hf_maks()
            _zn_ov = _get_zones()
            _hr_ts_ov = filtered[["avg_hr", "work_time_s"]].dropna()
            if _hr_ts_ov.empty:
                st.info("Ingen pulsdata i det filtrerte datasettet.")
            else:
                _ov_pairs = list(zip(
                    _hr_ts_ov["avg_hr"].astype(float),
                    _hr_ts_ov["work_time_s"].astype(float),
                ))
                _render_zone_chart(
                    _compute_zone_times(_ov_pairs, _zn_ov, _hf_ov),
                    _zn_ov, _hf_ov,
                    source_label=(
                        f"snitt-puls per økt × varighet "
                        f"({len(_hr_ts_ov)} av {len(filtered)} økter har pulsdata)"
                    ),
                )

            st.divider()
            st.markdown("#### Formkurve")
            benchmark_options = ["Alle økter", *sorted(filtered["benchmark_name"].dropna().unique().tolist())]
            if filtered.empty:
                st.info("Ingen økter i det filtrerte datasettet ennå.")
            else:
                fc1, fc2, fc3 = st.columns([2, 2, 1])
                with fc1:
                    selected_benchmark = st.selectbox(
                        "Benchmark",
                        benchmark_options,
                        key="form_curve_benchmark",
                        help=(
                            "Velg hvilken økttype formkurven skal bygges fra. "
                            "Et spesifikt benchmark viser bare sammenlignbare benchmarkøkter, "
                            "mens 'Alle økter' bruker hele det filtrerte utvalget for flere datapunkter."
                        ),
                    )

                trend_specs = _all_session_trend_specs() if selected_benchmark == "Alle økter" else _benchmark_trend_specs(selected_benchmark)
                trend_labels = [spec["label"] for spec in trend_specs]
                default_metric = "Distanse" if selected_benchmark == "1 time" else "Tid"
                default_metric_index = trend_labels.index(default_metric) if default_metric in trend_labels else 0

                with fc2:
                    selected_trend_metric = st.selectbox(
                        "Metrikk",
                        trend_labels,
                        index=default_metric_index,
                        key="form_curve_metric",
                        help=(
                            "Velg hva som skal plottes på y-aksen. For Tid og Pace er lavere bedre, "
                            "for Distanse, Watt og W/kg er høyere bedre. Når metrikken ikke er Puls, "
                            "pulsjusteres enkeltpunktene mot medianpuls i utvalget som vises."
                        ),
                    )
                with fc3:
                    rolling_window = st.slider(
                        "Glidende snitt",
                        min_value=1,
                        max_value=8,
                        value=4,
                        step=1,
                        key="form_curve_window",
                        help=(
                            "Antall økter som inngår i glidende snitt-linjen. "
                            "Lav verdi gir en mer følsom kurve, høy verdi gir en jevnere trend."
                        ),
                    )

                selected_spec = next(spec for spec in trend_specs if spec["label"] == selected_trend_metric)
                trend_df = filtered.copy() if selected_benchmark == "Alle økter" else filtered[filtered["benchmark_name"] == selected_benchmark].copy()
                trend_df[selected_spec["col"]] = pd.to_numeric(trend_df[selected_spec["col"]], errors="coerce")
                trend_df = trend_df.dropna(subset=["date", selected_spec["col"]]).sort_values("date")

                if len(trend_df) < 2:
                    st.info("For få økter i utvalget til å vise utvikling over tid.")
                else:
                    reference_hr = pd.to_numeric(trend_df["avg_hr"], errors="coerce").dropna().median()
                    is_hr_adjusted = selected_trend_metric != "Puls" and pd.notna(reference_hr)
                    if is_hr_adjusted:
                        trend_df["plot_value"] = trend_df.apply(
                            lambda r: _normalize_response_to_target(
                                r.get(selected_spec["col"]),
                                selected_spec["direction"],
                                r.get("avg_hr"),
                                reference_hr,
                            ),
                            axis=1,
                        )
                    else:
                        trend_df["plot_value"] = trend_df[selected_spec["col"]]
                    trend_df["rolling_value"] = pd.to_numeric(trend_df["plot_value"], errors="coerce").rolling(rolling_window, min_periods=1).mean()
                    trend_df["hover_value"] = trend_df[selected_spec["col"]]
                    trend_df["plot_value_label"] = trend_df["plot_value"]
                    if selected_trend_metric == "Tid":
                        trend_df["value_label"] = trend_df["hover_value"].map(_format_duration)
                        trend_df["rolling_label"] = trend_df["rolling_value"].map(_format_duration)
                        trend_df["plot_value_label"] = trend_df["plot_value"].map(_format_duration)
                    elif selected_trend_metric == "Pace":
                        trend_df["value_label"] = trend_df["hover_value"].map(_format_pace)
                        trend_df["rolling_label"] = trend_df["rolling_value"].map(_format_pace)
                        trend_df["plot_value_label"] = trend_df["plot_value"].map(_format_pace)
                    elif selected_trend_metric == "Distanse":
                        trend_df["value_label"] = trend_df["hover_value"].map(lambda v: f"{float(v):.0f} m")
                        trend_df["rolling_label"] = trend_df["rolling_value"].map(lambda v: f"{float(v):.0f} m")
                        trend_df["plot_value_label"] = trend_df["plot_value"].map(lambda v: f"{float(v):.0f} m")
                    elif selected_trend_metric == "Puls":
                        trend_df["value_label"] = trend_df["hover_value"].map(lambda v: f"{float(v):.0f} bpm")
                        trend_df["rolling_label"] = trend_df["rolling_value"].map(lambda v: f"{float(v):.0f} bpm")
                        trend_df["plot_value_label"] = trend_df["plot_value"].map(lambda v: f"{float(v):.0f} bpm")
                    elif selected_trend_metric == "Watt":
                        trend_df["value_label"] = trend_df["hover_value"].map(lambda v: f"{float(v):.0f} W")
                        trend_df["rolling_label"] = trend_df["rolling_value"].map(lambda v: f"{float(v):.0f} W")
                        trend_df["plot_value_label"] = trend_df["plot_value"].map(lambda v: f"{float(v):.0f} W")
                    else:
                        trend_df["value_label"] = trend_df["hover_value"].map(lambda v: f"{float(v):.2f} W/kg")
                        trend_df["rolling_label"] = trend_df["rolling_value"].map(lambda v: f"{float(v):.2f} W/kg")
                        trend_df["plot_value_label"] = trend_df["plot_value"].map(lambda v: f"{float(v):.2f} W/kg")

                    raw_label_name = "Råverdi"
                    adjusted_label_name = "Pulsjustert"
                    if not is_hr_adjusted:
                        adjusted_label_name = selected_trend_metric

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_df["date"],
                        y=trend_df["plot_value"],
                        mode="lines+markers",
                        name=selected_trend_metric,
                        line=dict(color="rgba(79,124,255,0.35)", width=2),
                        marker=dict(size=8, color="rgba(79,124,255,0.9)"),
                        customdata=trend_df[["description", "value_label", "plot_value_label"]].values,
                        hovertemplate=f"%{{x|%d.%m.%Y}}<br>%{{customdata[0]}}<br>{raw_label_name}: %{{customdata[1]}}<br>{adjusted_label_name}: %{{customdata[2]}}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=trend_df["date"],
                        y=trend_df["rolling_value"],
                        mode="lines",
                        name=f"Glidende snitt ({rolling_window})",
                        line=dict(color="gold", width=3),
                        customdata=trend_df[["rolling_label"]].values,
                        hovertemplate="%{x|%d.%m.%Y}<br>%{customdata[0]}<extra></extra>",
                    ))
                    fig.update_layout(
                        height=340,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Dato",
                        yaxis_title=selected_spec["y_title"],
                        legend_title="",
                    )
                    if selected_spec["direction"] == "lower":
                        fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                    context_label = "alle filtrerte økter" if selected_benchmark == "Alle økter" else selected_benchmark
                    if is_hr_adjusted:
                        st.caption(
                            f"Viser {len(trend_df)} økter for {context_label}. Verdiene er pulsjustert mot medianpuls i utvalget på {reference_hr:.0f} bpm. "
                            f"Glidende snitt er beregnet over siste {rolling_window} økter."
                        )
                    else:
                        st.caption(
                            f"Viser {len(trend_df)} økter for {context_label}. "
                            f"Glidende snitt er beregnet over siste {rolling_window} økter."
                        )

            st.divider()
            st.markdown("#### Effektivitet ved lik belastning")
            efficiency_mode = st.selectbox(
                "Vis utvikling for",
                [
                    "Puls ved samme pace",
                    "Pace ved samme puls",
                    "Watt ved samme puls",
                ],
                key="efficiency_mode",
            )

            efficiency_specs = {
                "Puls ved samme pace": {
                    "target_col": "pace_s_500",
                    "target_label": "Målpace",
                    "target_step": 0.1,
                    "target_format": _format_pace,
                    "target_is_pace": True,
                    "response_col": "avg_hr",
                    "response_label": "Puls",
                    "response_title": "bpm",
                    "response_direction": "lower",
                    "tol_label": "Toleransepace",
                    "tol_default": 2.5,
                    "tol_step": 0.1,
                    "tol_is_pace": True,
                },
                "Pace ved samme puls": {
                    "target_col": "avg_hr",
                    "target_label": "Målpuls",
                    "target_step": 1.0,
                    "target_format": lambda v: f"{float(v):.0f} bpm",
                    "target_is_pace": False,
                    "response_col": "pace_s_500",
                    "response_label": "Pace",
                    "response_title": "sek/500m",
                    "response_direction": "lower",
                    "tol_label": "Toleranse puls (bpm)",
                    "tol_default": 3.0,
                    "tol_step": 1.0,
                    "tol_is_pace": False,
                },
                "Watt ved samme puls": {
                    "target_col": "avg_hr",
                    "target_label": "Målpuls",
                    "target_step": 1.0,
                    "target_format": lambda v: f"{float(v):.0f} bpm",
                    "target_is_pace": False,
                    "response_col": "avg_watts",
                    "response_label": "Watt",
                    "response_title": "W",
                    "response_direction": "higher",
                    "tol_label": "Toleranse puls (bpm)",
                    "tol_default": 3.0,
                    "tol_step": 1.0,
                    "tol_is_pace": False,
                },
            }
            eff_spec = efficiency_specs[efficiency_mode]
            eff_df = filtered.dropna(subset=["date", eff_spec["target_col"], eff_spec["response_col"]]).copy()
            eff_df[eff_spec["target_col"]] = pd.to_numeric(eff_df[eff_spec["target_col"]], errors="coerce")
            eff_df[eff_spec["response_col"]] = pd.to_numeric(eff_df[eff_spec["response_col"]], errors="coerce")
            eff_df = eff_df.dropna(subset=[eff_spec["target_col"], eff_spec["response_col"]]).sort_values("date")

            if len(eff_df) < 3:
                st.info("For få økter med relevante data til å vise effektivitetstrend.")
            else:
                target_lo = float(eff_df[eff_spec["target_col"]].min())
                target_hi = float(eff_df[eff_spec["target_col"]].max())
                target_default = float(eff_df[eff_spec["target_col"]].median())

                ec1, ec2 = st.columns([2, 1])
                with ec1:
                    if eff_spec["target_is_pace"]:
                        target_options = _numeric_slider_options(target_lo, target_hi, float(eff_spec["target_step"]))
                        target_value = st.select_slider(
                            eff_spec["target_label"],
                            options=target_options,
                            value=min(target_options, key=lambda v: abs(v - target_default)),
                            format_func=_format_pace,
                            key="efficiency_target_value",
                        )
                    else:
                        target_value = st.slider(
                            eff_spec["target_label"],
                            min_value=target_lo,
                            max_value=target_hi,
                            value=target_default,
                            step=float(eff_spec["target_step"]),
                            key="efficiency_target_value",
                        )
                with ec2:
                    tol_min = float(eff_spec["tol_step"])
                    tol_max = max(float(eff_spec["tol_default"] * 4), float(eff_spec["tol_step"]))
                    tol_default = float(eff_spec["tol_default"])
                    if eff_spec["tol_is_pace"]:
                        tolerance_options = _numeric_slider_options(tol_min, tol_max, float(eff_spec["tol_step"]))
                        tolerance_value = st.select_slider(
                            eff_spec["tol_label"],
                            options=tolerance_options,
                            value=min(tolerance_options, key=lambda v: abs(v - tol_default)),
                            format_func=_format_pace,
                            key="efficiency_tolerance_value",
                        )
                    else:
                        tolerance_value = st.slider(
                            eff_spec["tol_label"],
                            min_value=tol_min,
                            max_value=tol_max,
                            value=tol_default,
                            step=float(eff_spec["tol_step"]),
                            key="efficiency_tolerance_value",
                        )

                eff_slice = eff_df[
                    (eff_df[eff_spec["target_col"]] >= float(target_value) - float(tolerance_value))
                    & (eff_df[eff_spec["target_col"]] <= float(target_value) + float(tolerance_value))
                ].copy()

                if len(eff_slice) < 3:
                    st.info("For få økter innen valgt toleranse. Øk toleransen eller velg et annet mål.")
                else:
                    eff_slice["adjusted_response"] = eff_slice.apply(
                        lambda r: _normalize_response_to_target(
                            r.get(eff_spec["response_col"]),
                            eff_spec["response_direction"],
                            r.get(eff_spec["target_col"]),
                            target_value,
                        ),
                        axis=1,
                    )
                    eff_slice["rolling_value"] = pd.to_numeric(eff_slice["adjusted_response"], errors="coerce").rolling(4, min_periods=1).mean()
                    if eff_spec["response_label"] == "Pace":
                        eff_slice["response_label_text"] = eff_slice[eff_spec["response_col"]].map(_format_pace)
                        eff_slice["adjusted_label_text"] = eff_slice["adjusted_response"].map(_format_pace)
                        eff_slice["rolling_label_text"] = eff_slice["rolling_value"].map(_format_pace)
                    elif eff_spec["response_label"] == "Puls":
                        eff_slice["response_label_text"] = eff_slice[eff_spec["response_col"]].map(lambda v: f"{float(v):.0f} bpm")
                        eff_slice["adjusted_label_text"] = eff_slice["adjusted_response"].map(lambda v: f"{float(v):.0f} bpm")
                        eff_slice["rolling_label_text"] = eff_slice["rolling_value"].map(lambda v: f"{float(v):.0f} bpm")
                    else:
                        eff_slice["response_label_text"] = eff_slice[eff_spec["response_col"]].map(lambda v: f"{float(v):.0f} W")
                        eff_slice["adjusted_label_text"] = eff_slice["adjusted_response"].map(lambda v: f"{float(v):.0f} W")
                        eff_slice["rolling_label_text"] = eff_slice["rolling_value"].map(lambda v: f"{float(v):.0f} W")

                    fig_eff = go.Figure()
                    fig_eff.add_trace(go.Scatter(
                        x=eff_slice["date"],
                        y=eff_slice["adjusted_response"],
                        mode="lines+markers",
                        name=eff_spec["response_label"],
                        line=dict(color="rgba(16,185,129,0.35)", width=2),
                        marker=dict(size=8, color="rgba(16,185,129,0.9)"),
                        customdata=eff_slice[["description", "response_label_text", "adjusted_label_text"]].values,
                        hovertemplate="%{x|%d.%m.%Y}<br>%{customdata[0]}<br>Råverdi: %{customdata[1]}<br>Normalisert: %{customdata[2]}<extra></extra>",
                    ))
                    fig_eff.add_trace(go.Scatter(
                        x=eff_slice["date"],
                        y=eff_slice["rolling_value"],
                        mode="lines",
                        name="Glidende snitt (4)",
                        line=dict(color="#111827", width=3),
                        customdata=eff_slice[["rolling_label_text"]].values,
                        hovertemplate="%{x|%d.%m.%Y}<br>%{customdata[0]}<extra></extra>",
                    ))
                    fig_eff.update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Dato",
                        yaxis_title=eff_spec["response_title"],
                        legend_title="",
                    )
                    if eff_spec["response_direction"] == "lower":
                        fig_eff.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_eff, use_container_width=True)
                    tolerance_text = _format_pace(tolerance_value) if eff_spec["tol_is_pace"] else f"{tolerance_value:g}"
                    st.caption(
                        f"Viser {len(eff_slice)} økter innenfor {eff_spec['target_format'](target_value)} ± {tolerance_text}. "
                        f"Responsen er normalisert mot valgt målverdi for å korrigere små avvik i faktisk puls/pace."
                    )

    with tab_leaderboard:
        st.subheader("Leaderboard")
        st.caption("Beste resultater – basert på aktivt filter")

        def _lb_rank_time(data: pd.DataFrame, dist_lo: float, dist_hi: float) -> pd.DataFrame:
            """Rank a fixed-distance event by shortest work time."""
            cand = data[
                data["work_distance_m"].between(dist_lo, dist_hi)
                & data["work_time_s"].notna()
                & (data["workout_format"] != "Intervall")
            ].copy()
            return cand.sort_values("work_time_s")

        def _lb_rank_dist(data: pd.DataFrame, time_lo: float, time_hi: float) -> pd.DataFrame:
            """Rank a fixed-time event by longest distance."""
            cand = data[
                data["work_time_s"].between(time_lo, time_hi)
                & data["work_distance_m"].notna()
                & (data["workout_format"] != "Intervall")
            ].copy()
            return cand.sort_values("work_distance_m", ascending=False)

        def _lb_rank_10x500(data: pd.DataFrame) -> pd.DataFrame:
            """Rank 10×500 m interval sessions by best average pace."""
            desc_match = data["description"].fillna("").str.contains(
                r"10\s*[xX×]\s*500", regex=True, na=False
            )
            interval_fallback = (
                (data["workout_format"] == "Intervall")
                & data["work_distance_m"].between(4900, 5100)
            )
            cand = data[(desc_match | interval_fallback) & data["pace_s_500"].notna()].copy()
            return cand.sort_values("pace_s_500")

        def _lb_interval_protocols(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
            """Group interval sessions by their work/rest setup for the event selector."""
            candidates = data[
                (data["workout_format"] == "Intervall")
                & data["pace_s_500"].notna()
            ].copy()
            if candidates.empty:
                return []
            candidates["_interval_protocol"] = candidates.apply(_interval_protocol_label, axis=1)
            candidates = candidates.dropna(subset=["_interval_protocol"])
            grouped_protocols = []
            for protocol, protocol_rows in candidates.groupby("_interval_protocol", sort=True):
                grouped_protocols.append((protocol, protocol_rows))
            return grouped_protocols

        def _lb_render_table(ranked: pd.DataFrame, mtype: str, session_key: str) -> None:
            """Render a clickable leaderboard table with up to 15 visible rows."""
            if ranked.empty:
                st.caption("Ingen data")
                return
            rows_display = []
            for _rank, (_, _r) in enumerate(ranked.iterrows(), 1):
                _date_str = pd.Timestamp(_r["date"]).strftime("%d.%m.%Y") if pd.notna(_r.get("date")) else ""
                if mtype == "time":
                    _primary = _format_duration(_r.get("work_time_s"))
                    _secondary = _format_pace(_r.get("pace_s_500"))
                elif mtype == "dist":
                    _primary = f"{float(_r['work_distance_m']) / 1000:.3f} km"
                    _secondary = _format_pace(_r.get("pace_s_500"))
                else:
                    _primary = _format_pace(_r.get("pace_s_500"))
                    _secondary = _format_duration(_r.get("work_time_s"))
                rows_display.append({
                    "#": _rank,
                    "Dato": _date_str,
                    "Resultat": _primary,
                    "Pace": _secondary if mtype != "pace" else _primary,
                    "Watt": round(float(_r["avg_watts"]), 1) if pd.notna(_r.get("avg_watts")) else None,
                    "W/kg": round(float(_r["w_kg"]), 1) if pd.notna(_r.get("w_kg")) else None,
                    "Vekt (kg)": round(float(_r["vekt"]), 1) if pd.notna(_r.get("vekt")) else None,
                    "Vekt dato": str(_r.get("vekt_dato", "")),
                    "Puls": int(round(float(_r["avg_hr"]))) if pd.notna(_r.get("avg_hr")) else None,
                    "Laktat": _format_lactate(_r.get("interval_lactate_avg")),
                    "Beskrivelse": str(_r.get("description", ""))[:50],
                })
            tbl_df = pd.DataFrame(rows_display)
            table_height = min(len(tbl_df), 15) * 35 + 38
            sel = st.dataframe(
                tbl_df,
                use_container_width=True,
                hide_index=True,
                height=table_height,
                on_select="rerun",
                selection_mode="single-row",
                key=session_key,
            )
            sel_rows = sel.selection.rows
            if sel_rows:
                chosen_orig = ranked.iloc[sel_rows[0]]
                _det = (
                    filtered.loc[chosen_orig.name]
                    if chosen_orig.name in filtered.index
                    else chosen_orig
                )
                st.session_state["_detail_row"] = _det
                st.session_state["_detail_df"] = filtered
                st.rerun()

        lb_entries = [
            ("500 m", _lb_rank_time(filtered, 495, 505), "time"),
            ("1 000 m", _lb_rank_time(filtered, 990, 1010), "time"),
            ("2 000 m", _lb_rank_time(filtered, 1990, 2010), "time"),
            ("3 000 m", _lb_rank_time(filtered, 2990, 3010), "time"),
            ("5 000 m", _lb_rank_time(filtered, 4990, 5010), "time"),
            ("10 000 m", _lb_rank_time(filtered, 9990, 10010), "time"),
            ("30 minutter", _lb_rank_dist(filtered, 1790, 1810), "dist"),
            ("45 minutter", _lb_rank_dist(filtered, 2690, 2710), "dist"),
            ("1 time", _lb_rank_dist(filtered, 3590, 3610), "dist"),
            ("10 × 500 m (snitt pace)", _lb_rank_10x500(filtered), "pace"),
        ]
        for protocol_label, protocol_rows in _lb_interval_protocols(filtered):
            lb_entries.append((protocol_label, protocol_rows.sort_values("pace_s_500"), "pace"))

        lb_entries.sort(key=lambda entry: (-len(entry[1]), entry[0]))
        lb_specs = {}
        quick_labels = {}
        for event_label, ranked_rows, result_type in lb_entries:
            session_count = len(ranked_rows)
            count_label = "økt" if session_count == 1 else "økter"
            display_label = f"{event_label} ({session_count} {count_label})"
            lb_specs[display_label] = (lambda rows=ranked_rows: rows, result_type)
            quick_labels[event_label] = display_label

        quick_5000, quick_10x500 = st.columns(2)
        if quick_5000.button("5 000 m", key="lb_quick_5000", use_container_width=True):
            st.session_state["_leaderboard_event"] = quick_labels["5 000 m"]
        if quick_10x500.button("10 × 500 m", key="lb_quick_10x500", use_container_width=True):
            st.session_state["_leaderboard_event"] = quick_labels["10 × 500 m (snitt pace)"]

        if st.session_state.get("_leaderboard_event") not in lb_specs:
            st.session_state["_leaderboard_event"] = quick_labels["5 000 m"]
        selected_event = st.selectbox(
            "Velg økt",
            options=list(lb_specs),
            key="_leaderboard_event",
        )
        st.caption("Økter med flest treff i aktivt filter vises først. Tallet i parentes er antall økter.")
        result_factory, result_type = lb_specs[selected_event]
        st.markdown(f"##### {selected_event}")
        _lb_render_table(result_factory(), result_type, f"lb_{selected_event}")

    with tab_lactate:
        _render_lactate_analysis_tab(filtered)

    with tab_sessions:
        st.subheader("Økter")

        # Sort on real datetime first, then format for display
        display = filtered.dropna(subset=["date"]).sort_values("date", ascending=False).copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d %H:%M")
        display["work_time"] = display["work_time_s"].map(_format_duration)
        display["pace"] = display["pace_s_500"].map(_format_pace)
        display["avg_watts"] = pd.to_numeric(display["avg_watts"], errors="coerce").round(1)
        display["benchmark"] = display["benchmark_name"].fillna("")
        display["status"] = display["benchmark_status"].fillna("")

        session_search = st.text_input(
            "Søk i økter",
            placeholder="f.eks. terskel / 5 km / 20:00",
            help="Søker i blant annet beskrivelse, tagger, distanse, tid, pace, watt, puls, format og sesong. Skill flere krav med komma eller /.",
        )
        search_terms = [term.casefold().strip() for term in re.split(r"[,/]", session_search) if term.strip()]
        if search_terms:
            def session_search_text(session: pd.Series) -> str:
                distance_m = pd.to_numeric(pd.Series([session.get("work_distance_m")]), errors="coerce").iloc[0]
                distance_km = pd.to_numeric(pd.Series([session.get("work_distance_km")]), errors="coerce").iloc[0]
                work_time_s = pd.to_numeric(pd.Series([session.get("work_time_s")]), errors="coerce").iloc[0]
                values = [
                    session.get("description", ""),
                    session.get("tags", ""),
                    session.get("workout_format", ""),
                    session.get("benchmark", ""),
                    session.get("season", ""),
                    session.get("date", ""),
                    _format_duration(work_time_s),
                    _format_pace(session.get("pace_s_500")),
                    session.get("avg_watts", ""),
                    session.get("avg_hr", ""),
                    session.get("stroke_rate", ""),
                    session.get("drag_factor", ""),
                ]
                if pd.notna(distance_m):
                    values.extend([f"{float(distance_m):g}", f"{float(distance_m) / 1000:g} km"])
                if pd.notna(distance_km):
                    values.extend([f"{float(distance_km):g}", f"{float(distance_km):.3f} km"])
                return " ".join(str(value) for value in values).casefold()

            searchable = display.apply(session_search_text, axis=1)
            display = display[searchable.map(lambda text: all(term in text for term in search_terms))]

        st.caption(f"Viser {len(display):,} økter (etter filter og søk)")

        display["total_distance"] = display["work_distance_m"].map(_table_total_distance)
        display["interval_setup"] = display.apply(_interval_protocol_label, axis=1).fillna("")
        cols = [
            "date",
            "workout_format",
            "work_time",
            "total_distance",
            "interval_setup",
            "pace",
            "avg_watts",
            "avg_hr",
            "benchmark",
            "status",
            "vekt",
            "w_kg",
            "tags",
        ]
        cols = [c for c in cols if c in display.columns]

        table = display[cols].copy()
        table["_session_row"] = range(len(display))
        grid_builder = GridOptionsBuilder.from_dataframe(table)
        grid_builder.configure_selection("single", use_checkbox=False)
        grid_builder.configure_column("_session_row", hide=True)
        grid_builder.configure_grid_options(rowHeight=34, headerHeight=36, suppressCellFocus=True)
        grid_builder.configure_column("date", "Dato", width=118)
        grid_builder.configure_column("workout_format", "Type", width=88)
        grid_builder.configure_column("work_time", "Tid", width=82)
        grid_builder.configure_column("total_distance", "Tot. dist.", width=90)
        grid_builder.configure_column(
            "interval_setup",
            "Intervaller",
            minWidth=180,
            flex=1,
            wrapText=True,
            autoHeight=True,
            tooltipField="interval_setup",
        )
        grid_builder.configure_column("pace", "Pace", width=75)
        grid_builder.configure_column("avg_watts", "Snitt W", width=82)
        grid_builder.configure_column("avg_hr", "Puls", width=65)
        grid_builder.configure_column("benchmark", "Test", width=90)
        grid_builder.configure_column("status", "Status", width=78)
        grid_builder.configure_column("vekt", "Vekt", width=72)
        grid_builder.configure_column("w_kg", "W/kg", width=70)
        grid_builder.configure_column("tags", "Tagger", width=120, tooltipField="tags")
        grid_options = grid_builder.build()
        selection = AgGrid(
            _table_display(table),
            gridOptions=grid_options,
            height=560,
            update_on=["selectionChanged"],
            show_toolbar=False,
            show_search=False,
            key="sessions_grid",
        )

        selected_rows = selection.selected_rows
        if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
            selected_row = selected_rows.iloc[0]
            st.session_state["_detail_row"] = display.iloc[int(selected_row["_session_row"])]
            st.session_state["_detail_df"] = filtered
            st.rerun()

        csv_bytes = table.drop(columns="_session_row").to_csv(index=False).encode("utf-8")
        st.download_button(
            "Last ned filtrert CSV",
            data=csv_bytes,
            file_name="treningshistorikk_filtrert.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("Sammenlign økter")

        options = filtered.sort_values("date", ascending=False)["label"].dropna().tolist()
        selected = st.multiselect("Velg økter", options, default=options[:0])

        if selected:
            chosen = filtered[filtered["label"].isin(selected)].copy().sort_values("date")

            comp = pd.DataFrame(
                {
                    "Dato": chosen["date"].dt.strftime("%Y-%m-%d %H:%M"),
                    "Beskrivelse": chosen["description"],
                    "Format": chosen["workout_format"],
                    "Tid": chosen["work_time_s"].map(_format_duration),
                    "Distanse (km)": chosen["work_distance_km"].round(1),
                    "Pace": chosen["pace_s_500"].map(_format_pace),
                    "Watt": chosen["avg_watts"],
                    "Puls": chosen.get("avg_hr"),
                    "SPM": chosen.get("stroke_rate"),
                    "Drag": chosen.get("drag_factor"),
                    "Log ID": chosen["log_id"],
                }
            )
            st.dataframe(_table_display(comp), use_container_width=True, hide_index=True)

            m = chosen.copy()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Watt")
                w = m.dropna(subset=["avg_watts"])
                if not w.empty:
                    fig = px.bar(w, x="description", y="avg_watts", hover_data=["date", "work_distance_km", "work_time_min"])  # type: ignore[arg-type]
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="W")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Mangler watt på valgte økter.")

            with c2:
                st.markdown("#### Pace")
                p = m.dropna(subset=["pace_s_500"])
                if not p.empty:
                    fig = px.bar(p, x="description", y="pace_s_500", hover_data=["date", "avg_watts", "work_distance_km"])  # type: ignore[arg-type]
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="sek/500m")
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Mangler pace på valgte økter.")

    with tab_settings:
        _render_settings_tab()


if __name__ == "__main__":
    main()
