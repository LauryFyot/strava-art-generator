#!/usr/bin/env python3
"""Generate a Strava-style poster from a GPX track."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import gpxpy
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

TILE_SIZE = 256


@dataclass
class TrackPoint:
    lat: float
    lon: float
    ele: float
    time: dt.datetime | None


@dataclass
class Stats:
    distance_m: float
    elevation_gain_m: float
    duration_s: float
    avg_speed_kmh: float


DEFAULT_THEME: dict[str, Any] = {
    "poster": {
        "width": 1400,
        "height": 2000,
        "dpi": 300,
        "background": "#f7f5ef",
        "frame_color": "#b8b5ab",
        "frame_width": 3,
        "margin": 48,
    },
    "map": {
        "padding": 0.12,
        "route_color": "#d6511f",
        "route_width": 8,
        "route_glow_color": "#f7f5ef",
        "route_glow_width": 16,
        "start_color": "#2fbf5b",
        "end_color": "#111111",
        "point_outline_color": "#f7f5ef",
        "point_outline_width": 4,
        "point_radius": 10,
        "tile_url_template": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "tile_zoom": None,
        "style": {
            "desaturate": 0.35,
            "contrast": 1.18,
            "brightness": 1.03,
            "blur_radius": 0.6,
            "tint_color": "#dfe9d8",
            "tint_strength": 0.18,
            "vignette_strength": 0.15,
            "grain_strength": 0.045,
            "grain_seed": 42,
        },
    },
    "typography": {
        "font_path": None,
        "title_size": 96,
        "subtitle_size": 36,
        "metric_value_size": 40,
        "metric_label_size": 24,
        "small_size": 20,
        "title_color": "#3d2b8f",
        "text_color": "#312f2a",
        "muted_color": "#7a766d",
    },
    "layout": {
        "title_strip_width": 180,
        "map_height_ratio": 0.64,
        "stats_height": 160,
        "profile_height": 170,
        "section_gap": 30,
    },
    "profile": {
        "fill_color": "#b7adc9",
        "line_color": "#8a7cb0",
        "grid_color": "#d8d4ca",
    },
    "textbox": {
        "enabled": False,
        "text": "",
        "width_cm": 8.0,
        "height_cm": 4.0,
        "x_cm": None,
        "y_cm": None,
        "padding_cm": 0.25,
        "font_path": None,
        "min_font_size": 14,
        "max_font_size": 64,
        "align": "left",
        "bg_color": "#ffffff",
        "border_color": "#b8b5ab",
        "border_width": 2,
        "text_color": "#312f2a",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_theme(path: str | None) -> dict[str, Any]:
    if not path:
        return DEFAULT_THEME
    with open(path, "r", encoding="utf-8") as f:
        custom = json.load(f)
    return deep_merge(DEFAULT_THEME, custom)


def read_gpx_points(gpx_path: str) -> list[TrackPoint]:
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points: list[TrackPoint] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                points.append(
                    TrackPoint(
                        lat=float(p.latitude),
                        lon=float(p.longitude),
                        ele=float(p.elevation) if p.elevation is not None else 0.0,
                        time=p.time,
                    )
                )

    if not points:
        raise ValueError("No track points found in GPX file.")
    return points


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_stats(points: list[TrackPoint], avg_speed_kmh_fallback: float) -> Stats:
    distance_m = 0.0
    gain = 0.0

    for p1, p2 in zip(points[:-1], points[1:]):
        distance_m += haversine_m(p1.lat, p1.lon, p2.lat, p2.lon)
        delta_ele = p2.ele - p1.ele
        if delta_ele > 0:
            gain += delta_ele

    valid_times = [p.time for p in points if p.time is not None]
    if len(valid_times) >= 2:
        duration_s = max((valid_times[-1] - valid_times[0]).total_seconds(), 1.0)
        avg_speed_kmh = (distance_m / 1000.0) / (duration_s / 3600.0)
    else:
        avg_speed_kmh = avg_speed_kmh_fallback
        duration_s = ((distance_m / 1000.0) / max(avg_speed_kmh, 0.1)) * 3600.0

    return Stats(
        distance_m=distance_m,
        elevation_gain_m=gain,
        duration_s=duration_s,
        avg_speed_kmh=avg_speed_kmh,
    )


def lon_to_x(lon: float, z: int) -> float:
    return (lon + 180.0) / 360.0 * (2**z) * TILE_SIZE


def lat_to_y(lat: float, z: int) -> float:
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    return (
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * (2**z)
        * TILE_SIZE
    )


def choose_zoom(
    lats: np.ndarray,
    lons: np.ndarray,
    width: int,
    height: int,
    padding: float,
    forced_zoom: int | None,
) -> int:
    if forced_zoom is not None:
        return forced_zoom

    for z in range(16, 2, -1):
        xs = np.array([lon_to_x(lon, z) for lon in lons])
        ys = np.array([lat_to_y(lat, z) for lat in lats])
        span_x = max(xs.max() - xs.min(), 1.0)
        span_y = max(ys.max() - ys.min(), 1.0)
        needed_w = span_x * (1 + padding * 2)
        needed_h = span_y * (1 + padding * 2)
        if needed_w <= width and needed_h <= height:
            return z

    return 3


def fetch_tile(z: int, x: int, y: int, template: str) -> Image.Image:
    max_tile = 2**z
    if x < 0 or y < 0 or x >= max_tile or y >= max_tile:
        return Image.new("RGB", (TILE_SIZE, TILE_SIZE), "#e8e8e8")

    url = template.format(z=z, x=x, y=y)
    headers = {"User-Agent": "strava-gpx-poster-generator/1.0"}
    response = requests.get(url, timeout=10, headers=headers)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return (220, 230, 220)
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def cm_to_px(cm: float, dpi: int) -> int:
    return int(round((cm / 2.54) * dpi))


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    words = text.replace("\n", " \n ").split()
    lines: list[str] = []
    current = ""

    for word in words:
        if word == "\\n":
            if current:
                lines.append(current)
                current = ""
            elif not lines:
                lines.append("")
            continue

        candidate = word if not current else f"{current} {word}"
        candidate_w, _ = text_size(draw, candidate, font)
        if candidate_w <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        # Hard-wrap very long words if needed.
        chunk = ""
        for ch in word:
            trial = chunk + ch
            trial_w, _ = text_size(draw, trial, font)
            if trial_w <= max_width:
                chunk = trial
            else:
                if chunk:
                    lines.append(chunk)
                chunk = ch
        current = chunk

    if current:
        lines.append(current)

    return "\n".join(lines)


def fit_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str | None,
    max_w: int,
    max_h: int,
    min_size: int,
    max_size: int,
    align: str,
) -> tuple[ImageFont.ImageFont, str, int]:
    best_font = load_font(font_path, min_size)
    best_text = wrap_text_to_width(draw, text, best_font, max_w)
    best_spacing = max(2, int(min_size * 0.25))

    lo = min_size
    hi = max_size
    found = False

    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_font(font_path, mid)
        wrapped = wrap_text_to_width(draw, text, font, max_w)
        spacing = max(2, int(mid * 0.25))
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align=align)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if w <= max_w and h <= max_h and wrapped.strip():
            found = True
            best_font = font
            best_text = wrapped
            best_spacing = spacing
            lo = mid + 1
        else:
            hi = mid - 1

    if found:
        return best_font, best_text, best_spacing

    # Fallback: force fit at minimum size with ellipsis truncation.
    font = load_font(font_path, min_size)
    spacing = max(2, int(min_size * 0.25))
    raw = text.strip()
    if not raw:
        return font, "", spacing

    candidate = raw
    for _ in range(len(raw)):
        wrapped = wrap_text_to_width(draw, candidate, font, max_w)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align=align)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= max_w and h <= max_h:
            return font, wrapped, spacing
        candidate = candidate[:-1].rstrip()
        if not candidate:
            break
        candidate = candidate + "..."

    return font, "", spacing


def draw_textbox(
    poster: Image.Image,
    theme: dict[str, Any],
    textbox_opts: dict[str, Any],
    content_rect: tuple[int, int, int, int],
) -> None:
    text = str(textbox_opts.get("text", "")).strip()
    enabled = bool(textbox_opts.get("enabled", False)) or bool(text)
    if not enabled or not text:
        return

    draw = ImageDraw.Draw(poster)
    dpi = int(theme["poster"].get("dpi", 300))

    width_px = max(1, cm_to_px(float(textbox_opts.get("width_cm", 8.0)), dpi))
    height_px = max(1, cm_to_px(float(textbox_opts.get("height_cm", 4.0)), dpi))
    padding_px = max(0, cm_to_px(float(textbox_opts.get("padding_cm", 0.25)), dpi))

    cx0, cy0, cx1, cy1 = content_rect
    content_w = cx1 - cx0
    content_h = cy1 - cy0
    width_px = min(width_px, content_w)
    height_px = min(height_px, content_h)

    x_cm = textbox_opts.get("x_cm")
    y_cm = textbox_opts.get("y_cm")
    if x_cm is None:
        x0 = cx0 + (content_w - width_px) // 2
    else:
        x0 = cx0 + cm_to_px(float(x_cm), dpi)
    if y_cm is None:
        y0 = cy0 + (content_h - height_px) // 2
    else:
        y0 = cy0 + cm_to_px(float(y_cm), dpi)

    x0 = max(cx0, min(x0, cx1 - width_px))
    y0 = max(cy0, min(y0, cy1 - height_px))
    x1 = x0 + width_px
    y1 = y0 + height_px

    bg_color = textbox_opts.get("bg_color", "#ffffff")
    border_color = textbox_opts.get("border_color", "#b8b5ab")
    border_width = int(textbox_opts.get("border_width", 2))
    text_color = textbox_opts.get("text_color", "#312f2a")
    align = str(textbox_opts.get("align", "left")).lower()
    if align not in {"left", "center", "right"}:
        align = "left"

    draw.rectangle((x0, y0, x1, y1), fill=bg_color, outline=border_color, width=border_width)

    ix0 = x0 + padding_px
    iy0 = y0 + padding_px
    ix1 = x1 - padding_px
    iy1 = y1 - padding_px
    if ix1 <= ix0 or iy1 <= iy0:
        return

    font_path = textbox_opts.get("font_path")
    min_font_size = int(textbox_opts.get("min_font_size", 14))
    max_font_size = int(textbox_opts.get("max_font_size", 64))
    if max_font_size < min_font_size:
        max_font_size = min_font_size

    font, wrapped, spacing = fit_text_to_box(
        draw,
        text,
        str(font_path) if font_path else None,
        ix1 - ix0,
        iy1 - iy0,
        min_font_size,
        max_font_size,
        align,
    )
    if not wrapped:
        return

    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align=align)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if align == "center":
        tx = ix0 + (ix1 - ix0 - tw) // 2
    elif align == "right":
        tx = ix1 - tw
    else:
        tx = ix0
    ty = iy0 + (iy1 - iy0 - th) // 2

    draw.multiline_text((tx, ty), wrapped, font=font, fill=text_color, spacing=spacing, align=align)


def apply_map_styling(bg: Image.Image, theme: dict[str, Any]) -> Image.Image:
    style = theme["map"].get("style", {})
    result = bg.convert("RGB")

    desaturate = float(style.get("desaturate", 0.0))
    desaturate = min(max(desaturate, 0.0), 1.0)
    if desaturate > 0.0:
        gray = result.convert("L").convert("RGB")
        result = Image.blend(result, gray, desaturate)

    contrast = float(style.get("contrast", 1.0))
    if abs(contrast - 1.0) > 1e-3:
        result = ImageEnhance.Contrast(result).enhance(contrast)

    brightness = float(style.get("brightness", 1.0))
    if abs(brightness - 1.0) > 1e-3:
        result = ImageEnhance.Brightness(result).enhance(brightness)

    tint_strength = float(style.get("tint_strength", 0.0))
    tint_strength = min(max(tint_strength, 0.0), 1.0)
    if tint_strength > 0.0:
        tint_rgb = hex_to_rgb(str(style.get("tint_color", "#dfe9d8")))
        tint_layer = Image.new("RGB", result.size, tint_rgb)
        result = Image.blend(result, tint_layer, tint_strength)

    blur_radius = float(style.get("blur_radius", 0.0))
    if blur_radius > 0.0:
        result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    vignette_strength = float(style.get("vignette_strength", 0.0))
    vignette_strength = min(max(vignette_strength, 0.0), 1.0)
    if vignette_strength > 0.0:
        w, h = result.size
        yy, xx = np.ogrid[0:h, 0:w]
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        dist = np.sqrt(((xx - cx) / max(w, 1)) ** 2 + ((yy - cy) / max(h, 1)) ** 2)
        dist_norm = dist / max(dist.max(), 1e-6)
        factor = 1.0 - vignette_strength * (dist_norm**1.9)
        factor = np.clip(factor, 0.0, 1.0)
        arr = np.asarray(result).astype(np.float32)
        arr *= factor[..., None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        result = Image.fromarray(arr, mode="RGB")

    grain_strength = float(style.get("grain_strength", 0.0))
    grain_strength = min(max(grain_strength, 0.0), 1.0)
    if grain_strength > 0.0:
        seed = int(style.get("grain_seed", 42))
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 255.0 * grain_strength, size=(result.height, result.width, 1))
        arr = np.asarray(result).astype(np.float32)
        arr += noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        result = Image.fromarray(arr, mode="RGB")

    return result.convert("RGBA")


def render_map_background(
    points: list[TrackPoint],
    map_w: int,
    map_h: int,
    theme: dict[str, Any],
    disable_tiles: bool,
    tile_zoom_override: int | None,
) -> tuple[Image.Image, list[tuple[float, float]]]:
    lats = np.array([p.lat for p in points])
    lons = np.array([p.lon for p in points])

    padding = float(theme["map"]["padding"])
    forced_zoom = tile_zoom_override if tile_zoom_override is not None else theme["map"]["tile_zoom"]
    z = choose_zoom(lats, lons, map_w, map_h, padding, forced_zoom)

    xs = np.array([lon_to_x(lon, z) for lon in lons])
    ys = np.array([lat_to_y(lat, z) for lat in lats])

    span_x = max(xs.max() - xs.min(), 1.0)
    span_y = max(ys.max() - ys.min(), 1.0)
    min_x = xs.min() - span_x * padding
    max_x = xs.max() + span_x * padding
    min_y = ys.min() - span_y * padding
    max_y = ys.max() + span_y * padding

    left_tile = int(math.floor(min_x / TILE_SIZE))
    right_tile = int(math.floor(max_x / TILE_SIZE))
    top_tile = int(math.floor(min_y / TILE_SIZE))
    bottom_tile = int(math.floor(max_y / TILE_SIZE))

    canvas_w = (right_tile - left_tile + 1) * TILE_SIZE
    canvas_h = (bottom_tile - top_tile + 1) * TILE_SIZE

    if disable_tiles:
        tile_canvas = Image.new("RGB", (canvas_w, canvas_h), "#dbe8d5")
    else:
        tile_canvas = Image.new("RGB", (canvas_w, canvas_h), "#dbe8d5")
        template = str(theme["map"]["tile_url_template"])
        for ty in range(top_tile, bottom_tile + 1):
            for tx in range(left_tile, right_tile + 1):
                px = (tx - left_tile) * TILE_SIZE
                py = (ty - top_tile) * TILE_SIZE
                try:
                    tile = fetch_tile(z, tx, ty, template)
                except Exception:
                    tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), "#dbe8d5")
                tile_canvas.paste(tile, (px, py))

    crop_left = int(round(min_x - left_tile * TILE_SIZE))
    crop_top = int(round(min_y - top_tile * TILE_SIZE))
    crop_right = int(round(max_x - left_tile * TILE_SIZE))
    crop_bottom = int(round(max_y - top_tile * TILE_SIZE))

    cropped = tile_canvas.crop((crop_left, crop_top, crop_right, crop_bottom))
    bg = cropped.resize((map_w, map_h), resample=Image.Resampling.BICUBIC).convert("RGBA")
    bg = apply_map_styling(bg, theme)

    route_pixels: list[tuple[float, float]] = []
    denom_x = max(max_x - min_x, 1e-6)
    denom_y = max(max_y - min_y, 1e-6)
    for p in points:
        x = (lon_to_x(p.lon, z) - min_x) / denom_x * map_w
        y = (lat_to_y(p.lat, z) - min_y) / denom_y * map_h
        route_pixels.append((x, y))

    return bg, route_pixels


def draw_profile(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    points: list[TrackPoint],
    theme: dict[str, Any],
) -> None:
    x0, y0, x1, y1 = rect
    w = x1 - x0
    h = y1 - y0

    elevations = np.array([p.ele for p in points], dtype=float)
    if len(elevations) < 2:
        return

    min_ele = float(np.min(elevations))
    max_ele = float(np.max(elevations))
    span_ele = max(max_ele - min_ele, 1.0)

    for i in range(5):
        gy = y0 + int(i * h / 4)
        draw.line([(x0, gy), (x1, gy)], fill=theme["profile"]["grid_color"], width=1)

    poly: list[tuple[float, float]] = [(x0, y1)]
    n = len(elevations)
    for i, e in enumerate(elevations):
        px = x0 + (i / (n - 1)) * w
        py = y1 - ((e - min_ele) / span_ele) * h
        poly.append((px, py))
    poly.append((x1, y1))

    draw.polygon(poly, fill=theme["profile"]["fill_color"])
    draw.line(poly[1:-1], fill=theme["profile"]["line_color"], width=3)


def load_font(path: str | None, size: int) -> ImageFont.ImageFont:
    if path and os.path.exists(path):
        return ImageFont.truetype(path, size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        pass
    return ImageFont.load_default()


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_poster(
    points: list[TrackPoint],
    stats: Stats,
    title: str,
    subtitle: str,
    date_text: str,
    output_path: str,
    theme: dict[str, Any],
    disable_tiles: bool,
    tile_zoom_override: int | None,
    textbox_options: dict[str, Any] | None,
) -> None:
    pw = int(theme["poster"]["width"])
    ph = int(theme["poster"]["height"])
    margin = int(theme["poster"]["margin"])

    poster = Image.new("RGBA", (pw, ph), theme["poster"]["background"])
    draw = ImageDraw.Draw(poster)

    frame_w = int(theme["poster"]["frame_width"])
    draw.rectangle(
        (margin, margin, pw - margin, ph - margin),
        outline=theme["poster"]["frame_color"],
        width=frame_w,
    )

    layout = theme["layout"]
    strip_w = int(layout["title_strip_width"])
    section_gap = int(layout["section_gap"])

    content_x0 = margin + 24
    content_x1 = pw - margin - 24
    content_w = content_x1 - content_x0

    map_h = int((ph - 2 * margin) * float(layout["map_height_ratio"]))
    map_y0 = margin + 40
    map_y1 = map_y0 + map_h

    map_x0 = content_x0
    map_x1 = content_x1 - strip_w - 16
    map_w = map_x1 - map_x0

    map_img, route_px = render_map_background(
        points,
        map_w,
        map_h,
        theme,
        disable_tiles,
        tile_zoom_override,
    )

    map_draw = ImageDraw.Draw(map_img)
    glow_width = int(theme["map"].get("route_glow_width", 0))
    if glow_width > 0:
        map_draw.line(route_px, fill=theme["map"].get("route_glow_color", "#f7f5ef"), width=glow_width, joint="curve")
    map_draw.line(route_px, fill=theme["map"]["route_color"], width=int(theme["map"]["route_width"]), joint="curve")

    r = int(theme["map"]["point_radius"])
    point_outline_w = int(theme["map"].get("point_outline_width", 0))
    point_outline = theme["map"].get("point_outline_color", "#f7f5ef")
    sx, sy = route_px[0]
    ex, ey = route_px[-1]
    if point_outline_w > 0:
        map_draw.ellipse(
            (sx - r - point_outline_w, sy - r - point_outline_w, sx + r + point_outline_w, sy + r + point_outline_w),
            fill=point_outline,
        )
        map_draw.ellipse(
            (ex - r - point_outline_w, ey - r - point_outline_w, ex + r + point_outline_w, ey + r + point_outline_w),
            fill=point_outline,
        )
    map_draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=theme["map"]["start_color"])
    map_draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=theme["map"]["end_color"])

    poster.alpha_composite(map_img, (map_x0, map_y0))

    typo = theme["typography"]
    font_path = typo["font_path"]
    title_font = load_font(font_path, int(typo["title_size"]))
    subtitle_font = load_font(font_path, int(typo["subtitle_size"]))
    metric_value_font = load_font(font_path, int(typo["metric_value_size"]))
    metric_label_font = load_font(font_path, int(typo["metric_label_size"]))
    small_font = load_font(font_path, int(typo["small_size"]))

    strip_x0 = map_x1 + 16
    strip_y0 = map_y0
    strip_y1 = map_y1

    vertical_text = f"{title.upper()}"
    tmp = Image.new("RGBA", (strip_w, map_h), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0, 0), vertical_text, font=title_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (strip_w - tw) // 2
    ty = (map_h - th) // 2
    tmp_draw.text((tx, ty), vertical_text, font=title_font, fill=typo["title_color"])
    rotated = tmp.rotate(90, expand=True)
    rx = strip_x0 + (strip_w - rotated.width) // 2
    ry = strip_y0 + (map_h - rotated.height) // 2
    poster.alpha_composite(rotated, (rx, ry))

    stats_y0 = map_y1 + section_gap
    stats_h = int(layout["stats_height"])
    stats_y1 = stats_y0 + stats_h

    draw.line([(content_x0, stats_y0), (content_x1, stats_y0)], fill=theme["poster"]["frame_color"], width=2)
    draw.line([(content_x0, stats_y1), (content_x1, stats_y1)], fill=theme["poster"]["frame_color"], width=2)

    draw.text((content_x0, stats_y0 + 12), title, font=subtitle_font, fill=typo["title_color"])
    draw.text((content_x0, stats_y0 + 56), date_text, font=small_font, fill=typo["muted_color"])
    if subtitle:
        draw.text((content_x0, stats_y0 + 86), subtitle, font=small_font, fill=typo["muted_color"])

    col_start = content_x0 + int(content_w * 0.36)
    col_w = int((content_x1 - col_start) / 3)

    metrics = [
        (f"{stats.distance_m / 1000.0:.2f} km", "Distance"),
        (format_duration(stats.duration_s), "Temps"),
        (f"{stats.elevation_gain_m:.0f} m", "Denivele +"),
    ]

    for i, (value, label) in enumerate(metrics):
        x = col_start + i * col_w
        if i > 0:
            draw.line([(x, stats_y0 + 8), (x, stats_y1 - 8)], fill=theme["poster"]["frame_color"], width=1)
        draw.text((x + 20, stats_y0 + 30), value, font=metric_value_font, fill=typo["text_color"])
        draw.text((x + 20, stats_y0 + 78), label, font=metric_label_font, fill=typo["muted_color"])

    profile_y0 = stats_y1 + section_gap
    profile_h = int(layout["profile_height"])
    profile_y1 = profile_y0 + profile_h

    draw.rectangle(
        (content_x0, profile_y0, content_x1, profile_y1),
        outline=theme["poster"]["frame_color"],
        width=1,
    )
    draw_profile(
        draw,
        (content_x0 + 10, profile_y0 + 12, content_x1 - 10, profile_y1 - 10),
        points,
        theme,
    )
    draw.text((content_x0, profile_y0 - 28), "ELEVATION", font=small_font, fill=typo["muted_color"])

    merged_textbox = deep_merge(theme.get("textbox", {}), textbox_options or {})
    draw_textbox(
        poster,
        theme,
        merged_textbox,
        (content_x0, map_y0, content_x1, profile_y1),
    )

    dpi = int(theme["poster"].get("dpi", 300))
    poster.convert("RGB").save(output_path, format="PNG", dpi=(dpi, dpi))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Strava-like poster from GPX")
    parser.add_argument("--gpx", required=True, help="Path to input GPX file")
    parser.add_argument("--title", required=True, help="Poster main title")
    parser.add_argument("--subtitle", default="", help="Secondary subtitle")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Displayed date")
    parser.add_argument("--output", default="poster.png", help="Output PNG path")
    parser.add_argument("--theme", default=None, help="Path to theme JSON")
    parser.add_argument(
        "--avg-speed-kmh",
        type=float,
        default=15.0,
        help="Fallback average speed used if GPX has no timestamps",
    )
    parser.add_argument("--tile-zoom", type=int, default=None, help="Force slippy map zoom level")
    parser.add_argument("--no-tiles", action="store_true", help="Disable online tile fetching")
    parser.add_argument("--textbox-text", default="", help="Text content for an optional textbox")
    parser.add_argument("--textbox-width-cm", type=float, default=None, help="Textbox width in centimeters")
    parser.add_argument("--textbox-height-cm", type=float, default=None, help="Textbox height in centimeters")
    parser.add_argument(
        "--textbox-x-cm",
        type=float,
        default=None,
        help="Textbox x position in cm from inner frame left; centered if omitted",
    )
    parser.add_argument(
        "--textbox-y-cm",
        type=float,
        default=None,
        help="Textbox y position in cm from inner frame top; centered if omitted",
    )
    parser.add_argument("--textbox-font-path", default=None, help="Path to .ttf/.otf font for textbox")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    theme = load_theme(args.theme)
    points = read_gpx_points(args.gpx)
    stats = compute_stats(points, avg_speed_kmh_fallback=args.avg_speed_kmh)

    textbox_options: dict[str, Any] = {}
    if args.textbox_text:
        textbox_options["enabled"] = True
        textbox_options["text"] = args.textbox_text
    if args.textbox_width_cm is not None:
        textbox_options["width_cm"] = args.textbox_width_cm
    if args.textbox_height_cm is not None:
        textbox_options["height_cm"] = args.textbox_height_cm
    if args.textbox_x_cm is not None:
        textbox_options["x_cm"] = args.textbox_x_cm
    if args.textbox_y_cm is not None:
        textbox_options["y_cm"] = args.textbox_y_cm
    if args.textbox_font_path is not None:
        textbox_options["font_path"] = args.textbox_font_path

    build_poster(
        points=points,
        stats=stats,
        title=args.title,
        subtitle=args.subtitle,
        date_text=args.date,
        output_path=args.output,
        theme=theme,
        disable_tiles=args.no_tiles,
        tile_zoom_override=args.tile_zoom,
        textbox_options=textbox_options,
    )

    print(f"Poster generated: {args.output}")


if __name__ == "__main__":
    main()
