"""poster_toolbox.py — Fonctions utilitaires de composition d'affiche."""

from __future__ import annotations

import os
import math
import importlib
from pathlib import Path
from typing import Any, Literal
from urllib.request import Request, urlopen
from io import BytesIO

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageEnhance


DPI = 300
HorizontalAlign = Literal["left", "center", "right"]
VerticalAlign = Literal["top", "center", "bottom"]
TextAlign = Literal["left", "center", "right", "justify"]
TextVerticalAlign = Literal["top", "center", "bottom"]
ImageFitMode = Literal["stretch", "contain", "cover"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cm_to_px(cm: float, dpi: int = DPI) -> int:
    return int(round((cm / 2.54) * dpi))

def _load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size=size)

    # Try a list of common scalable fonts before falling back to bitmap default.
    common_fonts = [
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for fp in common_fonts:
        try:
            return ImageFont.truetype(fp, size=size)
        except OSError:
            continue

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

def _resolve_rgba(color: str | None, opacity: float) -> tuple[int, int, int, int] | None:
    if color is None:
        return None
    c = str(color).strip().lower()
    if c in {"transparent", "none", ""}:
        return None
    rgb = ImageColor.getrgb(str(color))
    alpha = int(round(255 * _clamp01(opacity)))
    if len(rgb) == 4:
        alpha = int(round(rgb[3] * _clamp01(opacity)))
        return (rgb[0], rgb[1], rgb[2], alpha)
    return (rgb[0], rgb[1], rgb[2], alpha)

def _load_rgba_image(source: str | Path | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGBA")
    with Image.open(source) as opened:
        return opened.convert("RGBA")

def _apply_image_opacity(image: Image.Image, opacity: float) -> Image.Image:
    alpha_factor = _clamp01(opacity)
    if alpha_factor >= 1.0:
        return image
    result = image.copy()
    alpha = result.getchannel("A")
    alpha = alpha.point(lambda value: int(round(value * alpha_factor)))
    result.putalpha(alpha)
    return result

def _resize_image_to_box(
    source: Image.Image,
    target_w: int,
    target_h: int,
    fit_mode: ImageFitMode,
    align: HorizontalAlign,
    valign: VerticalAlign,
) -> Image.Image:
    if target_w <= 0 or target_h <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    src_w, src_h = source.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))

    fit_mode = str(fit_mode).lower()
    if fit_mode not in {"stretch", "contain", "cover"}:
        raise ValueError("image_fit must be one of: stretch, contain, cover")

    if fit_mode == "stretch":
        return source.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    scale_x = target_w / src_w
    scale_y = target_h / src_h
    scale = min(scale_x, scale_y) if fit_mode == "contain" else max(scale_x, scale_y)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = source.resize((resized_w, resized_h), resample=Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    if align == "center":
        offset_x = (target_w - resized_w) // 2
    elif align == "right":
        offset_x = target_w - resized_w
    else:
        offset_x = 0

    if valign == "center":
        offset_y = (target_h - resized_h) // 2
    elif valign == "bottom":
        offset_y = target_h - resized_h
    else:
        offset_y = 0

    canvas.alpha_composite(resized, (offset_x, offset_y))
    return canvas

def _get_point_value(point: Any, name: str, index: int, default: float = 0.0) -> float:
    value: Any = default
    if hasattr(point, name):
        value = getattr(point, name)
    elif isinstance(point, dict):
        value = point.get(name, default)
    elif isinstance(point, (list, tuple)) and len(point) > index:
        value = point[index]
    if value is None:
        return float(default)
    return float(value)

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values[:]

    radius = max(0, window // 2)
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        chunk = values[start:end]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed

def _compute_box_rect(
    img: Image.Image,
    width_cm: float,
    height_cm: float,
    x_cm: float | HorizontalAlign | None,
    y_cm: float | VerticalAlign | None,
    margin_x_cm: float | None,
    margin_y_cm: float | None,
    margin_cm: float,
    dpi: int,
) -> tuple[int, int, int, int]:
    pw, ph = img.size
    box_w = min(_cm_to_px(width_cm, dpi), pw)
    box_h = min(_cm_to_px(height_cm, dpi), ph)
    mx_cm = margin_cm if margin_x_cm is None else margin_x_cm
    my_cm = margin_cm if margin_y_cm is None else margin_y_cm
    margin_x_px = _cm_to_px(mx_cm, dpi)
    margin_y_px = _cm_to_px(my_cm, dpi)

    if x_cm == "left" or x_cm is None:
        x0 = margin_x_px
    elif x_cm == "center":
        x0 = (pw - box_w) // 2
    elif x_cm == "right":
        x0 = pw - box_w - margin_x_px
    else:
        x0 = _cm_to_px(float(x_cm), dpi)

    if y_cm == "top" or y_cm is None:
        y0 = margin_y_px
    elif y_cm == "center":
        y0 = (ph - box_h) // 2
    elif y_cm == "bottom":
        y0 = ph - box_h - margin_y_px
    else:
        y0 = _cm_to_px(float(y_cm), dpi)

    x0 = max(margin_x_px, min(x0, pw - box_w - margin_x_px))
    y0 = max(margin_y_px, min(y0, ph - box_h - margin_y_px))
    return (x0, y0, x0 + box_w, y0 + box_h)

def _wrap_text(
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
            lines.append(current)
            current = ""
            continue

        candidate = word if not current else f"{current} {word}"
        w = draw.textbbox((0, 0), candidate, font=font)
        if (w[2] - w[0]) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        chunk = ""
        for ch in word:
            trial = chunk + ch
            tw = draw.textbbox((0, 0), trial, font=font)[2] - draw.textbbox((0, 0), trial, font=font)[0]
            if tw <= max_width:
                chunk = trial
            else:
                if chunk:
                    lines.append(chunk)
                chunk = ch
        current = chunk

    if current:
        lines.append(current)

    return "\n".join(lines)

def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str | None,
    max_w: int,
    max_h: int,
    min_size: int,
    max_size: int,
    align: str,
    wrap_text: bool = True,
) -> tuple[ImageFont.ImageFont, str, int]:
    best_font = _load_font(font_path, min_size)
    best_text = _wrap_text(draw, text, best_font, max_w) if wrap_text else " ".join(text.splitlines())
    best_spacing = max(2, int(min_size * 0.25))

    lo, hi = min_size, max_size
    found = False

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_path, mid)
        wrapped = _wrap_text(draw, text, font, max_w) if wrap_text else " ".join(text.splitlines())
        spacing = max(2, int(mid * 0.25))
        if wrap_text:
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align=align)
        else:
            bbox = draw.textbbox((0, 0), wrapped, font=font)
        if (bbox[2] - bbox[0]) <= max_w and (bbox[3] - bbox[1]) <= max_h and wrapped.strip():
            found = True
            best_font, best_text, best_spacing = font, wrapped, spacing
            lo = mid + 1
        else:
            hi = mid - 1

    if found:
        return best_font, best_text, best_spacing

    # Fallback: taille minimale + troncature avec "..."
    font = _load_font(font_path, min_size)
    spacing = max(2, int(min_size * 0.25))
    candidate = text.strip().replace("\n", " ") if not wrap_text else text.strip()
    while candidate:
        wrapped = _wrap_text(draw, candidate, font, max_w) if wrap_text else candidate
        if wrap_text:
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align=align)
        else:
            bbox = draw.textbbox((0, 0), wrapped, font=font)
        if (bbox[2] - bbox[0]) <= max_w and (bbox[3] - bbox[1]) <= max_h:
            return font, wrapped, spacing
        candidate = candidate[:-1].rstrip()
        if candidate:
            candidate = candidate + "..."

    return font, "", spacing

def _truncate_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_w: int,
    max_h: int,
    spacing: int,
    wrap_text: bool = True,
) -> str:
    candidate = text.strip().replace("\n", " ") if not wrap_text else text.strip()
    while candidate:
        wrapped = _wrap_text(draw, candidate, font, max_w) if wrap_text else candidate
        if wrap_text:
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align="left")
        else:
            bbox = draw.textbbox((0, 0), wrapped, font=font)
        if (bbox[2] - bbox[0]) <= max_w and (bbox[3] - bbox[1]) <= max_h:
            return wrapped
        candidate = candidate[:-1].rstrip()
        if candidate:
            candidate = candidate + "..."
    return ""

def _draw_multiline_justified(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    x: int,
    y: int,
    max_w: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int] | str,
    spacing: int,
) -> None:
    if not lines:
        return

    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] - draw.textbbox((0, 0), "Ag", font=font)[1]
    cy = y
    last_index = len(lines) - 1

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            cy += line_height + spacing
            continue

        # Last line is left-aligned to keep natural paragraph endings.
        if idx == last_index:
            draw.text((x, cy), line, font=font, fill=fill)
            cy += line_height + spacing
            continue

        words = line.split()
        if len(words) <= 1:
            draw.text((x, cy), line, font=font, fill=fill)
            cy += line_height + spacing
            continue

        word_widths = [draw.textbbox((0, 0), w, font=font)[2] - draw.textbbox((0, 0), w, font=font)[0] for w in words]
        total_words = sum(word_widths)
        gaps = len(words) - 1
        base_space = draw.textbbox((0, 0), " ", font=font)[2] - draw.textbbox((0, 0), " ", font=font)[0]
        extra_total = max(0, max_w - total_words - base_space * gaps)
        extra_per_gap = extra_total // gaps
        remainder = extra_total % gaps

        cx = x
        for i, w in enumerate(words):
            draw.text((cx, cy), w, font=font, fill=fill)
            cx += word_widths[i]
            if i < gaps:
                cx += base_space + extra_per_gap + (1 if i < remainder else 0)

        cy += line_height + spacing


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

def create_blank_poster(
    width_cm: float = 21.0,
    height_cm: float = 29.7,
    color: str = "#ffffff",
    dpi: int = DPI,
) -> Image.Image:
    """Cree une image RGBA vierge aux dimensions specifiees en cm.

    Args:
        width_cm:  Largeur en centimetres (defaut: 21 cm = A4 portrait).
        height_cm: Hauteur en centimetres (defaut: 29.7 cm = A4 portrait).
        color:     Couleur de fond initiale (hex ou nom CSS).
        dpi:       Resolution en DPI (defaut: 300).

    Returns:
        Image PIL en mode RGBA.
    """
    w_px = _cm_to_px(width_cm, dpi)
    h_px = _cm_to_px(height_cm, dpi)
    img = Image.new("RGBA", (w_px, h_px), color)
    img.info["dpi"] = dpi
    return img

def set_background(
    img: Image.Image,
    color: str = "#ffffff",
    keep_content: bool = False,
) -> Image.Image:
    """Applique une couleur de fond unie sur l'image.

    Args:
        img:   Image PIL.
        color: Couleur de fond (hex ou nom CSS).
        keep_content: Si True, conserve le contenu existant au-dessus du fond.

    Returns:
        Nouvelle image avec le fond appliqué.
    """
    bg = Image.new("RGBA", img.size, color)
    if keep_content:
        # Conserve le contenu existant au-dessus du nouveau fond.
        bg.alpha_composite(img.convert("RGBA"))
    return bg

def add_textbox(
    img: Image.Image,
    text: str,
    width_cm: float = 8.0,
    height_cm: float = 4.0,
    x_cm: float | HorizontalAlign | None = "center",
    y_cm: float | VerticalAlign | None = "center",
    margin_x_cm: float | None = None,
    margin_y_cm: float | None = None,
    margin_cm: float = 0.0,
    font_path: str | None = None,
    font_size: int | None = None,
    min_font_size: int = 14,
    max_font_size: int | None = None,
    text_align: TextAlign = "left",
    text_valign: TextVerticalAlign = "center",
    align: HorizontalAlign | None = None,
    padding_cm: float = 0.25,
    bg_color: str = "#ffffff",
    bg_opacity: float = 1.0,
    border_color: str | None = "#b8b5ab",
    border_opacity: float = 1.0,
    border_width: int = 2,
    text_color: str = "#312f2a",
    text_opacity: float = 1.0,
    wrap_text: bool = True,
    text_rotation: int = 0,
    dpi: int | None = None,
) -> Image.Image:
    """Ajoute une textbox sur l'image.

    Les dimensions et coordonnees sont en centimetres.
    La position x_cm/y_cm accepte:
    - un float en cm (coin haut-gauche de la box)
    - 'left' / 'center' / 'right'  pour x_cm
    - 'top' / 'center' / 'bottom'  pour y_cm

    La marge est configurable separement en X/Y via margin_x_cm et margin_y_cm.
    Si non fournis, margin_cm est utilise pour les deux axes.
    Le texte est automatiquement ajuste pour rentrer dans la box.

    Args:
        img:            Image PIL cible (modifiee en place).
        text:           Contenu textuel.
        width_cm:       Largeur de la box en cm.
        height_cm:      Hauteur de la box en cm.
        x_cm:           Position horizontale en cm, ou 'left'/'center'/'right'.
        y_cm:           Position verticale en cm, ou 'top'/'center'/'bottom'.
        margin_x_cm:    Marge horizontale entre la box et les bords (cm).
        margin_y_cm:    Marge verticale entre la box et les bords (cm).
        margin_cm:      Marge legacy appliquee aux deux axes si margin_x_cm/y_cm absents.
        font_path:      Chemin vers un .ttf/.otf (None = police systeme).
        font_size:      Si renseigne, force la taille de police (sinon auto-fit max).
        min_font_size:  Taille minimale de police.
        max_font_size:  Taille maximale de police (auto-ajustee pour tenir dans la box).
                Si None, la borne max est calculee depuis la taille de la box.
        text_align:     Alignement: 'left', 'center', 'right', 'justify'.
        text_valign:    Alignement vertical du texte: 'top', 'center', 'bottom'.
        align:          Alias backward-compatible de text_align (left/center/right).
        padding_cm:     Espace interne entre le bord de la box et le texte.
        bg_color:       Couleur de fond de la box.
        bg_opacity:     Opacite du fond entre 0.0 et 1.0.
        border_color:   Couleur de la bordure (None = pas de bordure).
        border_opacity: Opacite de la bordure entre 0.0 et 1.0.
        border_width:   Epaisseur de la bordure en pixels.
        text_color:     Couleur du texte.
        text_opacity:   Opacite du texte entre 0.0 et 1.0.
        wrap_text:      True = autorise retour a la ligne, False = force mono-ligne.
        text_rotation:  Rotation du texte en degres (0, 90, 180, 270).
        dpi:            DPI a utiliser pour la conversion cm->px (auto si None).

    Returns:
        La meme image modifiee.
    """
    resolved_dpi: int = dpi if dpi is not None else int(img.info.get("dpi", DPI))
    pw, ph = img.size

    box_w = min(_cm_to_px(width_cm, resolved_dpi), pw)
    box_h = min(_cm_to_px(height_cm, resolved_dpi), ph)
    mx_cm = margin_cm if margin_x_cm is None else margin_x_cm
    my_cm = margin_cm if margin_y_cm is None else margin_y_cm
    margin_x_px = _cm_to_px(mx_cm, resolved_dpi)
    margin_y_px = _cm_to_px(my_cm, resolved_dpi)

    # --- Position horizontale ---
    if x_cm == "left" or x_cm is None:
        x0 = margin_x_px
    elif x_cm == "center":
        x0 = (pw - box_w) // 2
    elif x_cm == "right":
        x0 = pw - box_w - margin_x_px
    else:
        x0 = _cm_to_px(float(x_cm), resolved_dpi)

    # --- Position verticale ---
    if y_cm == "top" or y_cm is None:
        y0 = margin_y_px
    elif y_cm == "center":
        y0 = (ph - box_h) // 2
    elif y_cm == "bottom":
        y0 = ph - box_h - margin_y_px
    else:
        y0 = _cm_to_px(float(y_cm), resolved_dpi)

    # Clamping dans les bornes
    x0 = max(margin_x_px, min(x0, pw - box_w - margin_x_px))
    y0 = max(margin_y_px, min(y0, ph - box_h - margin_y_px))
    x1 = x0 + box_w
    y1 = y0 + box_h

    draw = ImageDraw.Draw(img)

    # Backward compatibility alias.
    if align is not None and text_align == "left":
        text_align = align
    text_align = str(text_align).lower()
    if text_align not in {"left", "center", "right", "justify"}:
        text_align = "left"
    text_valign = str(text_valign).lower()
    if text_valign not in {"top", "center", "bottom"}:
        text_valign = "center"

    # Fond + bordure
    fill_rgba = _resolve_rgba(bg_color, bg_opacity)
    border_rgba = _resolve_rgba(border_color, border_opacity)
    if fill_rgba is not None:
        draw.rectangle((x0, y0, x1, y1), fill=fill_rgba)
    if border_rgba and border_width > 0:
        draw.rectangle((x0, y0, x1, y1), outline=border_rgba, width=border_width)

    # Zone interne (padding)
    padding_px = _cm_to_px(padding_cm, resolved_dpi)
    ix0, iy0 = x0 + padding_px, y0 + padding_px
    ix1, iy1 = x1 - padding_px, y1 - padding_px
    if ix1 <= ix0 or iy1 <= iy0:
        return img

    content_w = ix1 - ix0
    content_h = iy1 - iy0

    rotation = int(text_rotation) % 360
    if rotation not in {0, 90, 180, 270}:
        raise ValueError("text_rotation must be one of: 0, 90, 180, 270")

    # For vertical text, fitting constraints are swapped before rotation.
    fit_w, fit_h = (content_h, content_w) if rotation in {90, 270} else (content_w, content_h)

    layout_align = "left" if text_align == "justify" else text_align
    if font_size is not None:
        forced_size = max(1, int(font_size))
        font = _load_font(font_path, forced_size)
        spacing = max(2, int(forced_size * 0.25))
        wrapped = _truncate_text_to_box(draw, text, font, fit_w, fit_h, spacing, wrap_text=wrap_text)
    else:
        effective_max_font_size = (
            max(min_font_size, int(max(fit_w, fit_h)))
            if max_font_size is None
            else max(min_font_size, int(max_font_size))
        )
        font, wrapped, spacing = _fit_text(
            draw,
            text,
            font_path,
            fit_w,
            fit_h,
            min_font_size,
            effective_max_font_size,
            layout_align,
            wrap_text=wrap_text,
        )

    if not wrapped.strip():
        return img

    if wrap_text:
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align=layout_align)
    else:
        bbox = draw.textbbox((0, 0), wrapped, font=font)
    bx0, by0, bx1, by1 = bbox
    tw, th = bx1 - bx0, by1 - by0

    if layout_align == "center":
        tx = ix0 + (ix1 - ix0 - tw) // 2 - bx0
    elif layout_align == "right":
        tx = ix1 - tw - bx0
    else:
        tx = ix0 - bx0
    if text_valign == "top":
        ty = iy0 - by0
    elif text_valign == "bottom":
        ty = iy1 - th - by0
    else:
        ty = iy0 + (iy1 - iy0 - th) // 2 - by0

    text_rgba = _resolve_rgba(text_color, text_opacity)
    if text_rgba is None:
        return img

    # Render text into a sprite first, then rotate and place into the textbox.
    if text_align == "justify" and wrap_text:
        sprite_w = max(1, fit_w)
        sprite_h = max(1, th)
        sprite = Image.new("RGBA", (sprite_w, sprite_h), (0, 0, 0, 0))
        sprite_draw = ImageDraw.Draw(sprite)
        _draw_multiline_justified(
            draw=sprite_draw,
            lines=wrapped.split("\n"),
            x=0,
            y=-by0,
            max_w=sprite_w,
            font=font,
            fill=text_rgba,
            spacing=spacing,
        )
    else:
        sprite_w = max(1, tw)
        sprite_h = max(1, th)
        sprite = Image.new("RGBA", (sprite_w, sprite_h), (0, 0, 0, 0))
        sprite_draw = ImageDraw.Draw(sprite)
        if wrap_text:
            sprite_draw.multiline_text(
                (-bx0, -by0),
                wrapped,
                font=font,
                fill=text_rgba,
                spacing=spacing,
                align=layout_align,
            )
        else:
            sprite_draw.text((-bx0, -by0), wrapped, font=font, fill=text_rgba)

    if rotation:
        sprite = sprite.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

    rw, rh = sprite.size
    if layout_align == "center":
        px = ix0 + (content_w - rw) // 2
    elif layout_align == "right":
        px = ix1 - rw
    else:
        px = ix0

    if text_valign == "top":
        py = iy0
    elif text_valign == "bottom":
        py = iy1 - rh
    else:
        py = iy0 + (content_h - rh) // 2

    img.alpha_composite(sprite, (px, py))

    return img

def add_shape(
    img: Image.Image,
    width_cm: float = 8.0,
    height_cm: float = 4.0,
    x_cm: float | HorizontalAlign | None = "center",
    y_cm: float | VerticalAlign | None = "center",
    margin_x_cm: float | None = None,
    margin_y_cm: float | None = None,
    margin_cm: float = 0.0,
    bg_color: str = "#ffffff",
    bg_opacity: float = 1.0,
    border_color: str | None = "#b8b5ab",
    border_opacity: float = 1.0,
    border_width: int = 2,
    shape_rotation: int = 0,
    dpi: int | None = None,
) -> Image.Image:
    """Ajoute une forme rectangulaire (carré/rectangle) sur l'image.

    Les dimensions et coordonnees sont en centimetres.
    La position x_cm/y_cm accepte:
    - un float en cm (coin haut-gauche de la forme)
    - 'left' / 'center' / 'right'  pour x_cm
    - 'top' / 'center' / 'bottom'  pour y_cm

    La marge est configurable separement en X/Y via margin_x_cm et margin_y_cm.
    Si non fournis, margin_cm est utilise pour les deux axes.

    Args:
        img:            Image PIL cible (modifiee en place).
        width_cm:       Largeur de la forme en cm.
        height_cm:      Hauteur de la forme en cm.
        x_cm:           Position horizontale en cm, ou 'left'/'center'/'right'.
        y_cm:           Position verticale en cm, ou 'top'/'center'/'bottom'.
        margin_x_cm:    Marge horizontale entre la forme et les bords (cm).
        margin_y_cm:    Marge verticale entre la forme et les bords (cm).
        margin_cm:      Marge legacy appliquee aux deux axes si margin_x_cm/y_cm absents.
        bg_color:       Couleur de remplissage de la forme.
        bg_opacity:     Opacite du remplissage entre 0.0 et 1.0.
        border_color:   Couleur de la bordure (None = pas de bordure).
        border_opacity: Opacite de la bordure entre 0.0 et 1.0.
        border_width:   Epaisseur de la bordure en pixels.
        shape_rotation: Rotation de la forme en degres (0, 90, 180, 270).
        dpi:            DPI a utiliser pour la conversion cm->px (auto si None).

    Returns:
        La meme image modifiee.
    """
    resolved_dpi: int = dpi if dpi is not None else int(img.info.get("dpi", DPI))
    pw, ph = img.size

    shape_w = min(_cm_to_px(width_cm, resolved_dpi), pw)
    shape_h = min(_cm_to_px(height_cm, resolved_dpi), ph)
    mx_cm = margin_cm if margin_x_cm is None else margin_x_cm
    my_cm = margin_cm if margin_y_cm is None else margin_y_cm
    margin_x_px = _cm_to_px(mx_cm, resolved_dpi)
    margin_y_px = _cm_to_px(my_cm, resolved_dpi)

    # --- Position horizontale ---
    if x_cm == "left" or x_cm is None:
        x0 = margin_x_px
    elif x_cm == "center":
        x0 = (pw - shape_w) // 2
    elif x_cm == "right":
        x0 = pw - shape_w - margin_x_px
    else:
        x0 = _cm_to_px(float(x_cm), resolved_dpi)

    # --- Position verticale ---
    if y_cm == "top" or y_cm is None:
        y0 = margin_y_px
    elif y_cm == "center":
        y0 = (ph - shape_h) // 2
    elif y_cm == "bottom":
        y0 = ph - shape_h - margin_y_px
    else:
        y0 = _cm_to_px(float(y_cm), resolved_dpi)

    # Clamping dans les bornes
    x0 = max(margin_x_px, min(x0, pw - shape_w - margin_x_px))
    y0 = max(margin_y_px, min(y0, ph - shape_h - margin_y_px))
    x1 = x0 + shape_w
    y1 = y0 + shape_h

    # Validation de la rotation
    rotation = int(shape_rotation) % 360
    if rotation not in {0, 90, 180, 270}:
        raise ValueError("shape_rotation must be one of: 0, 90, 180, 270")

    # Si pas de rotation, dessine directement
    if rotation == 0:
        draw = ImageDraw.Draw(img)
        fill_rgba = _resolve_rgba(bg_color, bg_opacity)
        border_rgba = _resolve_rgba(border_color, border_opacity)
        
        if fill_rgba is not None:
            draw.rectangle((x0, y0, x1, y1), fill=fill_rgba)
        if border_rgba and border_width > 0:
            draw.rectangle((x0, y0, x1, y1), outline=border_rgba, width=border_width)
        
        return img

    # Sinon, cree un sprite et le rotate
    fill_rgba = _resolve_rgba(bg_color, bg_opacity)
    border_rgba = _resolve_rgba(border_color, border_opacity)
    
    # Cree le sprite de la forme
    sprite = Image.new("RGBA", (shape_w, shape_h), (0, 0, 0, 0))
    sprite_draw = ImageDraw.Draw(sprite)
    
    # Dessine la forme dans le sprite
    if fill_rgba is not None:
        sprite_draw.rectangle((0, 0, shape_w - 1, shape_h - 1), fill=fill_rgba)
    if border_rgba and border_width > 0:
        sprite_draw.rectangle((0, 0, shape_w - 1, shape_h - 1), outline=border_rgba, width=border_width)
    
    # Rotate le sprite
    sprite = sprite.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    
    # Reposition apres rotation pour garder le centre
    rw, rh = sprite.size
    px = x0 + (shape_w - rw) // 2
    py = y0 + (shape_h - rh) // 2
    
    # Composite sur l'image
    img.alpha_composite(sprite, (px, py))
    
    return img

def add_image_shape(
    img: Image.Image,
    photo: str | Path | Image.Image,
    width_cm: float = 8.0,
    height_cm: float = 4.0,
    x_cm: float | HorizontalAlign | None = "center",
    y_cm: float | VerticalAlign | None = "center",
    margin_x_cm: float | None = None,
    margin_y_cm: float | None = None,
    margin_cm: float = 0.0,
    padding_cm: float = 0.0,
    bg_color: str = "transparent",
    bg_opacity: float = 1.0,
    border_color: str | None = "#b8b5ab",
    border_opacity: float = 1.0,
    border_width: int = 2,
    image_fit: ImageFitMode = "cover",
    image_align: HorizontalAlign = "center",
    image_valign: VerticalAlign = "center",
    image_opacity: float = 1.0,
    shape_rotation: int = 0,
    dpi: int | None = None,
) -> Image.Image:
    """Ajoute une forme rectangulaire remplie par une image.

    Les dimensions et coordonnees sont en centimetres.
    La forme est rectangulaire, avec les memes regles de placement que add_shape.

    Le contenu image supporte trois modes:
    - stretch: deforme l'image pour remplir exactement la zone
    - contain: fait rentrer toute l'image sans deformation, avec marge visible si besoin
    - cover: remplit toute la zone sans deformation, avec crop si besoin

    Args:
        img:            Image PIL cible (modifiee en place).
        photo:          Chemin vers une image ou instance PIL.Image.Image.
        width_cm:       Largeur de la forme en cm.
        height_cm:      Hauteur de la forme en cm.
        x_cm:           Position horizontale en cm, ou 'left'/'center'/'right'.
        y_cm:           Position verticale en cm, ou 'top'/'center'/'bottom'.
        margin_x_cm:    Marge horizontale entre la forme et les bords (cm).
        margin_y_cm:    Marge verticale entre la forme et les bords (cm).
        margin_cm:      Marge legacy appliquee aux deux axes si margin_x_cm/y_cm absents.
        padding_cm:     Espace interne entre la bordure et l'image.
        bg_color:       Couleur de fond visible autour de l'image en mode contain.
        bg_opacity:     Opacite du fond entre 0.0 et 1.0.
        border_color:   Couleur de la bordure (None = pas de bordure).
        border_opacity: Opacite de la bordure entre 0.0 et 1.0.
        border_width:   Epaisseur de la bordure en pixels.
        image_fit:      'stretch', 'contain' ou 'cover'.
        image_align:    Alignement horizontal de l'image: 'left', 'center', 'right'.
        image_valign:   Alignement vertical de l'image: 'top', 'center', 'bottom'.
        image_opacity:  Opacite appliquee a l'image entre 0.0 et 1.0.
        shape_rotation: Rotation de la forme en degres (0, 90, 180, 270).
        dpi:            DPI a utiliser pour la conversion cm->px (auto si None).

    Returns:
        La meme image modifiee.
    """
    resolved_dpi: int = dpi if dpi is not None else int(img.info.get("dpi", DPI))
    pw, ph = img.size

    shape_w = min(_cm_to_px(width_cm, resolved_dpi), pw)
    shape_h = min(_cm_to_px(height_cm, resolved_dpi), ph)
    mx_cm = margin_cm if margin_x_cm is None else margin_x_cm
    my_cm = margin_cm if margin_y_cm is None else margin_y_cm
    margin_x_px = _cm_to_px(mx_cm, resolved_dpi)
    margin_y_px = _cm_to_px(my_cm, resolved_dpi)

    if x_cm == "left" or x_cm is None:
        x0 = margin_x_px
    elif x_cm == "center":
        x0 = (pw - shape_w) // 2
    elif x_cm == "right":
        x0 = pw - shape_w - margin_x_px
    else:
        x0 = _cm_to_px(float(x_cm), resolved_dpi)

    if y_cm == "top" or y_cm is None:
        y0 = margin_y_px
    elif y_cm == "center":
        y0 = (ph - shape_h) // 2
    elif y_cm == "bottom":
        y0 = ph - shape_h - margin_y_px
    else:
        y0 = _cm_to_px(float(y_cm), resolved_dpi)

    x0 = max(margin_x_px, min(x0, pw - shape_w - margin_x_px))
    y0 = max(margin_y_px, min(y0, ph - shape_h - margin_y_px))

    rotation = int(shape_rotation) % 360
    if rotation not in {0, 90, 180, 270}:
        raise ValueError("shape_rotation must be one of: 0, 90, 180, 270")

    fill_rgba = _resolve_rgba(bg_color, bg_opacity)
    border_rgba = _resolve_rgba(border_color, border_opacity)
    padding_px = _cm_to_px(padding_cm, resolved_dpi)

    sprite = Image.new("RGBA", (shape_w, shape_h), (0, 0, 0, 0))
    sprite_draw = ImageDraw.Draw(sprite)

    if fill_rgba is not None:
        sprite_draw.rectangle((0, 0, shape_w - 1, shape_h - 1), fill=fill_rgba)

    inner_x0 = padding_px
    inner_y0 = padding_px
    inner_x1 = shape_w - padding_px
    inner_y1 = shape_h - padding_px
    inner_w = inner_x1 - inner_x0
    inner_h = inner_y1 - inner_y0

    if inner_w > 0 and inner_h > 0:
        source = _load_rgba_image(photo)
        source = _apply_image_opacity(source, image_opacity)
        placed = _resize_image_to_box(source, inner_w, inner_h, image_fit, image_align, image_valign)
        sprite.alpha_composite(placed, (inner_x0, inner_y0))

    if border_rgba and border_width > 0:
        sprite_draw.rectangle((0, 0, shape_w - 1, shape_h - 1), outline=border_rgba, width=border_width)

    if rotation:
        sprite = sprite.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

    rw, rh = sprite.size
    px = x0 + (shape_w - rw) // 2
    py = y0 + (shape_h - rh) // 2
    img.alpha_composite(sprite, (px, py))

    return img

def elevation_v1(
    img: Image.Image,
    points: list[Any],
    width_cm: float = 10.0,
    height_cm: float = 4.0,
    x_cm: float | HorizontalAlign | None = "center",
    y_cm: float | VerticalAlign | None = "center",
    margin_x_cm: float | None = None,
    margin_y_cm: float | None = None,
    margin_cm: float = 0.0,
    padding_cm: float = 0.25,
    title: str | None = None,
    subtitle: str | None = None,
    font_path: str | None = None,
    title_color: str = "#7e77aa",
    text_color: str = "#7a766d",
    axis_text_color: str | None = None,
    x_text_size: int | None = None,
    y_text_size: int | None = None,
    bg_color: str = "#f5f1e6",
    bg_opacity: float = 1.0,
    border_color: str | None = "#b8b5ab",
    border_opacity: float = 1.0,
    border_width: int = 2,
    grid_color: str = "#d8d4ca",
    grid_line_color: str | None = None,
    grid_opacity: float = 1.0,
    grid_rows: int = 4,
    grid_cols: int = 6,
    skip_min_x_label: bool = True,
    skip_min_y_label: bool = True,
    x_label_gap_cm: float = 0.1,
    y_label_gap_cm: float = 0.1,
    fill_color: str = "#b7adc9",
    fill_opacity: float = 0.9,
    elevation_opacity: float | None = None,
    fill_top_color: str | None = None,
    fill_top_opacity: float | None = None,
    line_color: str = "#8a7cb0",
    line_opacity: float = 1.0,
    line_width: int = 3,
    smooth_window: int = 9,
    dpi: int | None = None,
) -> Image.Image:
    """Dessine un profil d'elevation dans une boite style affiche.

    La fonction attend une liste de points GPX avec au minimum un attribut `ele`.
    Si les points contiennent aussi `lat` et `lon`, l'axe X est calcule en distance reelle.
    Sinon, la progression se fait simplement par index.
    """
    resolved_dpi: int = dpi if dpi is not None else int(img.info.get("dpi", DPI))
    x0, y0, x1, y1 = _compute_box_rect(
        img,
        width_cm,
        height_cm,
        x_cm,
        y_cm,
        margin_x_cm,
        margin_y_cm,
        margin_cm,
        resolved_dpi,
    )

    draw = ImageDraw.Draw(img)
    fill_rgba = _resolve_rgba(bg_color, bg_opacity)
    border_rgba = _resolve_rgba(border_color, border_opacity)
    if fill_rgba is not None:
        draw.rectangle((x0, y0, x1, y1), fill=fill_rgba)
    if border_rgba and border_width > 0:
        draw.rectangle((x0, y0, x1, y1), outline=border_rgba, width=border_width)

    if len(points) < 2:
        return img

    padding_px = _cm_to_px(padding_cm, resolved_dpi)
    content_x0 = x0 + padding_px
    content_y0 = y0 + padding_px
    content_x1 = x1 - padding_px
    content_y1 = y1 - padding_px
    if content_x1 <= content_x0 or content_y1 <= content_y0:
        return img

    box_h = y1 - y0
    title_size = max(14, int(round(box_h * 0.12)))
    base_label_size = max(11, int(round(box_h * 0.08)))
    resolved_x_text_size = base_label_size if x_text_size is None else max(8, int(x_text_size))
    resolved_y_text_size = base_label_size if y_text_size is None else max(8, int(y_text_size))
    title_font = _load_font(font_path, title_size)
    x_label_font = _load_font(font_path, resolved_x_text_size)
    y_label_font = _load_font(font_path, resolved_y_text_size)
    subtitle_font = _load_font(font_path, max(resolved_x_text_size, resolved_y_text_size))

    effective_grid_color = grid_line_color if grid_line_color is not None else grid_color
    effective_fill_opacity = fill_opacity if elevation_opacity is None else elevation_opacity
    effective_text_color = axis_text_color if axis_text_color is not None else text_color

    title_rgba = _resolve_rgba(title_color, 1.0)
    text_rgba = _resolve_rgba(effective_text_color, 1.0)
    grid_rgba = _resolve_rgba(effective_grid_color, grid_opacity)
    fill_profile_rgba = _resolve_rgba(fill_color, effective_fill_opacity)
    fill_top_rgba = _resolve_rgba(fill_top_color, fill_top_opacity if fill_top_opacity is not None else effective_fill_opacity)
    line_rgba = _resolve_rgba(line_color, line_opacity)

    title_h = 0
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_h = title_bbox[3] - title_bbox[1]
        if title_rgba is not None:
            draw.text((content_x0, content_y0), title, font=title_font, fill=title_rgba)

    if subtitle and text_rgba is not None:
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_w = subtitle_bbox[2] - subtitle_bbox[0]
        draw.text((content_x1 - subtitle_w, content_y0 + 1), subtitle, font=subtitle_font, fill=text_rgba)

    y_label_sample = draw.textbbox((0, 0), "0000m", font=y_label_font)
    x_label_sample = draw.textbbox((0, 0), "00.0km", font=x_label_font)
    y_label_w = (y_label_sample[2] - y_label_sample[0]) + max(8, int(round(box_h * 0.02)))
    x_label_h = (x_label_sample[3] - x_label_sample[1]) + max(8, int(round(box_h * 0.03)))
    plot_x0 = content_x0 + y_label_w
    plot_y0 = content_y0 + title_h + max(10, int(round(box_h * 0.04)))
    plot_x1 = content_x1
    plot_y1 = content_y1 - x_label_h
    plot_w = plot_x1 - plot_x0
    plot_h = plot_y1 - plot_y0
    if plot_w <= 4 or plot_h <= 4:
        return img

    elevations = [_get_point_value(point, "ele", 2, 0.0) for point in points]
    smoothed_elevations = _moving_average(elevations, smooth_window)

    distances_m: list[float] = [0.0]
    can_use_geo = all(
        hasattr(point, "lat") and hasattr(point, "lon")
        or (isinstance(point, dict) and "lat" in point and "lon" in point)
        or (isinstance(point, (list, tuple)) and len(point) >= 2)
        for point in points
    )
    if can_use_geo:
        total = 0.0
        for p1, p2 in zip(points[:-1], points[1:]):
            lat1 = _get_point_value(p1, "lat", 0, 0.0)
            lon1 = _get_point_value(p1, "lon", 1, 0.0)
            lat2 = _get_point_value(p2, "lat", 0, 0.0)
            lon2 = _get_point_value(p2, "lon", 1, 0.0)
            total += _haversine_m(lat1, lon1, lat2, lon2)
            distances_m.append(total)
    else:
        distances_m = [float(index) for index in range(len(points))]

    min_ele = min(smoothed_elevations)
    max_ele = max(smoothed_elevations)
    if math.isclose(min_ele, max_ele):
        min_ele -= 1.0
        max_ele += 1.0
    span_ele = max_ele - min_ele
    total_distance = max(distances_m[-1], 1.0)

    if grid_rgba is not None:
        for row in range(grid_rows + 1):
            gy = plot_y0 + int(round((row / max(grid_rows, 1)) * plot_h))
            draw.line([(plot_x0, gy), (plot_x1, gy)], fill=grid_rgba, width=1)
        for col in range(grid_cols + 1):
            gx = plot_x0 + int(round((col / max(grid_cols, 1)) * plot_w))
            draw.line([(gx, plot_y0), (gx, plot_y1)], fill=grid_rgba, width=1)

    profile_points: list[tuple[int, int]] = []
    for distance_m, elevation_m in zip(distances_m, smoothed_elevations):
        px = plot_x0 + int(round((distance_m / total_distance) * plot_w))
        py = plot_y1 - int(round(((elevation_m - min_ele) / span_ele) * plot_h))
        profile_points.append((px, py))

    if len(profile_points) >= 2:
        polygon = [(plot_x0, plot_y1)] + profile_points + [(plot_x1, plot_y1)]
        if fill_profile_rgba is not None:
            if fill_top_rgba is None:
                draw.polygon(polygon, fill=fill_profile_rgba)
            else:
                local_w = max(1, plot_w + 1)
                local_h = max(1, plot_h + 1)
                gradient_layer = Image.new("RGBA", (local_w, local_h), (0, 0, 0, 0))
                gradient_draw = ImageDraw.Draw(gradient_layer)
                for y in range(local_h):
                    ratio = y / max(local_h - 1, 1)
                    r = int(round(fill_top_rgba[0] + (fill_profile_rgba[0] - fill_top_rgba[0]) * ratio))
                    g = int(round(fill_top_rgba[1] + (fill_profile_rgba[1] - fill_top_rgba[1]) * ratio))
                    b = int(round(fill_top_rgba[2] + (fill_profile_rgba[2] - fill_top_rgba[2]) * ratio))
                    a = int(round(fill_top_rgba[3] + (fill_profile_rgba[3] - fill_top_rgba[3]) * ratio))
                    gradient_draw.line([(0, y), (local_w, y)], fill=(r, g, b, a), width=1)

                local_polygon = [(px - plot_x0, py - plot_y0) for px, py in polygon]
                polygon_mask = Image.new("L", (local_w, local_h), 0)
                mask_draw = ImageDraw.Draw(polygon_mask)
                mask_draw.polygon(local_polygon, fill=255)

                clipped = Image.new("RGBA", (local_w, local_h), (0, 0, 0, 0))
                clipped.paste(gradient_layer, (0, 0), polygon_mask)
                img.alpha_composite(clipped, (plot_x0, plot_y0))
        if line_rgba is not None and line_width > 0:
            draw.line(profile_points, fill=line_rgba, width=line_width)

    if text_rgba is not None:
        y_label_gap_px = int(round(_cm_to_px(y_label_gap_cm, resolved_dpi)))
        x_label_gap_px = int(round(_cm_to_px(x_label_gap_cm, resolved_dpi)))
        
        for row in range(grid_rows + 1):
            if skip_min_y_label and row == grid_rows:
                continue
            ratio = 1.0 - (row / max(grid_rows, 1))
            value = min_ele + ratio * span_ele
            label = f"{int(round(value))}m"
            bbox = draw.textbbox((0, 0), label, font=y_label_font)
            label_w = bbox[2] - bbox[0]
            label_h = bbox[3] - bbox[1]
            gy = plot_y0 + int(round((row / max(grid_rows, 1)) * plot_h)) - (label_h // 2)
            draw.text((plot_x0 - label_w - y_label_gap_px, gy), label, font=y_label_font, fill=text_rgba)

        for col in range(grid_cols + 1):
            if skip_min_x_label and col == 0:
                continue
            ratio = col / max(grid_cols, 1)
            value_km = (ratio * total_distance) / 1000.0
            if total_distance >= 10000:
                label = f"{value_km:.0f}km"
            else:
                label = f"{value_km:.1f}km"
            bbox = draw.textbbox((0, 0), label, font=x_label_font)
            label_w = bbox[2] - bbox[0]
            gx = plot_x0 + int(round(ratio * plot_w)) - (label_w // 2)
            draw.text((gx, plot_y1 + x_label_gap_px), label, font=x_label_font, fill=text_rgba)

    return img


# ---------------------------------------------------------------------------
# Cartographie
# ---------------------------------------------------------------------------

def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convertit lat/lon en index de tuile OSM."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0 ** zoom
    xtile = math.floor((lon + 180.0) / 360.0 * n)
    ytile = math.floor(
        (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi)
        / 2.0
        * n
    )
    return int(xtile), int(ytile)


def _lat_lon_to_world_px(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Convertit lat/lon en coordonnees pixel monde Web Mercator."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0 ** zoom
    x = ((lon + 180.0) / 360.0) * n * 256.0
    y = (
        (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi)
        / 2.0
        * n
        * 256.0
    )
    return x, y


def _download_osm_tile(
    x: int,
    y: int,
    zoom: int,
    tile_server: str = "a.tile.openstreetmap.org",
) -> Image.Image | None:
    """Telecharge une tuile OSM 256x256 avec User-Agent."""
    url = f"https://{tile_server}/{zoom}/{x}/{y}.png"
    try:
        req = Request(url, headers={"User-Agent": "strava-generator/1.0 (+local dev)"})
        with urlopen(req, timeout=8) as response:
            return Image.open(BytesIO(response.read())).convert("RGBA")
    except Exception:
        return None


def add_map(
    img: Image.Image,
    points: list[Any],
    width_cm: float = 10.0,
    height_cm: float = 8.0,
    x_cm: float | HorizontalAlign | None = "center",
    y_cm: float | VerticalAlign | None = "center",
    margin_x_cm: float | None = None,
    margin_y_cm: float | None = None,
    margin_cm: float = 0.0,
    padding_cm: float = 0.25,
    zoom: int | None = None,
    bg_color: str = "#e6e3d6",
    bg_opacity: float = 1.0,
    border_color: str | None = "#999999",
    border_opacity: float = 1.0,
    border_width: int = 1,
    trace_color: str = "#d62828",
    trace_opacity: float = 1.0,
    trace_width: int = 3,
    style_contrast: float = 1.0,
    style_brightness: float = 1.0,
    dpi: int | None = None,
) -> Image.Image:
    """Ajoute une carte OSM stylisee avec le trace GPX."""
    resolved_dpi = dpi if dpi is not None else int(img.info.get("dpi", DPI))
    x0, y0, x1, y1 = _compute_box_rect(
        img, width_cm, height_cm, x_cm, y_cm, margin_x_cm, margin_y_cm, margin_cm, resolved_dpi
    )

    box_w = x1 - x0
    box_h = y1 - y0
    if box_w <= 2 or box_h <= 2:
        return img

    geo_points: list[tuple[float, float]] = []
    for p in points:
        lat = _get_point_value(p, "lat", 0, 0.0)
        lon = _get_point_value(p, "lon", 1, 0.0)
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and not (lat == 0.0 and lon == 0.0):
            geo_points.append((lat, lon))

    if len(geo_points) < 2:
        return img

    lats = [lat for lat, _ in geo_points]
    lons = [lon for _, lon in geo_points]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    lat_pad = max((lat_max - lat_min) * 0.08, 0.002)
    lon_pad = max((lon_max - lon_min) * 0.08, 0.002)
    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad

    # Auto-zoom: evite de trop agrandir une petite zone (source de flou).
    if zoom is None:
        resolved_zoom = 13
        for candidate_zoom in range(18, 2, -1):
            tlx, tly = _lat_lon_to_world_px(lat_max, lon_min, candidate_zoom)
            brx, bry = _lat_lon_to_world_px(lat_min, lon_max, candidate_zoom)
            span_x = abs(brx - tlx)
            span_y = abs(bry - tly)
            if span_x >= box_w and span_y >= box_h:
                resolved_zoom = candidate_zoom
                break
    else:
        resolved_zoom = max(1, min(int(zoom), 19))

    top_left_world = _lat_lon_to_world_px(lat_max, lon_min, resolved_zoom)
    bottom_right_world = _lat_lon_to_world_px(lat_min, lon_max, resolved_zoom)

    min_x = min(top_left_world[0], bottom_right_world[0])
    max_x = max(top_left_world[0], bottom_right_world[0])
    min_y = min(top_left_world[1], bottom_right_world[1])
    max_y = max(top_left_world[1], bottom_right_world[1])

    if max_x - min_x < 2 or max_y - min_y < 2:
        return img

    tile_min_x = int(math.floor(min_x / 256.0))
    tile_max_x = int(math.floor(max_x / 256.0))
    tile_min_y = int(math.floor(min_y / 256.0))
    tile_max_y = int(math.floor(max_y / 256.0))

    canvas_w = (tile_max_x - tile_min_x + 1) * 256
    canvas_h = (tile_max_y - tile_min_y + 1) * 256
    tile_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    servers = ["a.tile.openstreetmap.org", "b.tile.openstreetmap.org", "c.tile.openstreetmap.org"]
    any_tile = False

    for tx in range(tile_min_x, tile_max_x + 1):
        for ty in range(tile_min_y, tile_max_y + 1):
            tile = None
            for server in servers:
                tile = _download_osm_tile(tx, ty, resolved_zoom, server)
                if tile is not None:
                    break
            if tile is not None:
                any_tile = True
                paste_x = (tx - tile_min_x) * 256
                paste_y = (ty - tile_min_y) * 256
                tile_canvas.paste(tile, (paste_x, paste_y))

    crop_left = int(round(min_x - tile_min_x * 256.0))
    crop_top = int(round(min_y - tile_min_y * 256.0))
    crop_right = int(round(max_x - tile_min_x * 256.0))
    crop_bottom = int(round(max_y - tile_min_y * 256.0))

    crop_left = max(0, min(crop_left, canvas_w - 1))
    crop_top = max(0, min(crop_top, canvas_h - 1))
    crop_right = max(crop_left + 1, min(crop_right, canvas_w))
    crop_bottom = max(crop_top + 1, min(crop_bottom, canvas_h))

    map_crop = tile_canvas.crop((crop_left, crop_top, crop_right, crop_bottom))
    map_resized = map_crop.resize((box_w, box_h), Image.Resampling.LANCZOS)

    bg_rgba = _resolve_rgba(bg_color, bg_opacity)
    if bg_rgba is None:
        bg_rgba = (230, 227, 214, 255)
    styled_bg = Image.new("RGBA", (box_w, box_h), bg_rgba)

    if any_tile:
        if style_contrast != 1.0:
            map_resized = ImageEnhance.Contrast(map_resized).enhance(style_contrast)
        if style_brightness != 1.0:
            map_resized = ImageEnhance.Brightness(map_resized).enhance(style_brightness)
        map_layer = Image.alpha_composite(styled_bg, map_resized.convert("RGBA"))
    else:
        map_layer = styled_bg
        bg_draw = ImageDraw.Draw(map_layer)
        step = max(24, int(round(min(box_w, box_h) * 0.06)))
        line_col = (190, 186, 173, 140)
        for xx in range(0, box_w, step):
            bg_draw.line([(xx, 0), (xx, box_h)], fill=line_col, width=1)
        for yy in range(0, box_h, step):
            bg_draw.line([(0, yy), (box_w, yy)], fill=line_col, width=1)

    draw = ImageDraw.Draw(map_layer)
    trace_rgba = _resolve_rgba(trace_color, trace_opacity)
    if trace_rgba is not None:
        world_w = max_x - min_x
        world_h = max_y - min_y
        trace_px: list[tuple[int, int]] = []
        for lat, lon in geo_points:
            wx, wy = _lat_lon_to_world_px(lat, lon, resolved_zoom)
            px = int(round(((wx - min_x) / world_w) * (box_w - 1)))
            py = int(round(((wy - min_y) / world_h) * (box_h - 1)))
            trace_px.append((px, py))
        if len(trace_px) > 1:
            draw.line(trace_px, fill=trace_rgba, width=max(1, int(trace_width)), joint="curve")

    border_rgba = _resolve_rgba(border_color, border_opacity)
    if border_rgba is not None and border_width > 0:
        draw.rectangle((0, 0, box_w - 1, box_h - 1), outline=border_rgba, width=border_width)

    img.alpha_composite(map_layer, (x0, y0))
    return img


def _render_osmnx_graph_image(
    graph: Any,
    width_px: int,
    height_px: int,
    dpi: int,
    bgcolor: str,
    edge_colors: list[str],
    edge_widths: list[float],
    edge_alpha: float = 1.0,
    trace_points: list[tuple[float, float]] | None = None,
    trace_color: str | None = None,
    trace_opacity: float = 1.0,
    trace_width: float = 3.0,
) -> tuple[Image.Image, float, float, float, float]:
    """Rend un graph OSMnx en image PIL RGBA sans fenetre interactive.

    Retourne (image, west, east, south, north) — les limites reelles des axes matplotlib.
    Si `trace_points` est fourni, le trace est dessine dans le meme axe matplotlib
    que la carte pour eviter tout decalage de projection.
    """
    try:
        ox = importlib.import_module("osmnx")
    except ImportError as exc:
        raise ImportError("osmnx est requis pour add_map_v1/add_map_v2") from exc

    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError("matplotlib est requis pour add_map_v1/add_map_v2") from exc

    fig_w = max(1.0, float(width_px) / max(dpi, 1))
    fig_h = max(1.0, float(height_px) / max(dpi, 1))

    fig, ax = ox.plot_graph(
        graph,
        node_size=0,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        bgcolor=bgcolor,
        show=False,
        save=False,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        edge_alpha=edge_alpha,
    )

    # Capturer les limites reelles AVANT ajout de la trace.
    render_west, render_east = ax.get_xlim()   # lon
    render_south, render_north = ax.get_ylim() # lat

    if trace_points and trace_color:
        trace_lon = [lon for lat, lon in trace_points]
        trace_lat = [lat for lat, lon in trace_points]
        if len(trace_lon) >= 2:
            ax.plot(
                trace_lon,
                trace_lat,
                color=trace_color,
                alpha=_clamp01(trace_opacity),
                linewidth=max(0.1, float(trace_width)),
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=5,
            )
            # Evite tout autoscale apres ajout de la trace.
            ax.set_xlim(render_west, render_east)
            ax.set_ylim(render_south, render_north)

    # Evite les recadrages de sauvegarde qui peuvent creer 1-2 px d'offset.
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        facecolor=fig.get_facecolor(),
        transparent=False,
    )
    plt.close(fig)

    buffer.seek(0)
    with Image.open(buffer) as rendered:
        image = rendered.convert("RGBA")
    if image.size != (width_px, height_px):
        image = image.resize((width_px, height_px), Image.Resampling.LANCZOS)
    return image, render_west, render_east, render_south, render_north


def add_map_v1(
    img: Image.Image,
    points: list[Any] | None = None,
    center_point: tuple[float, float] | None = None,
    dist_m: int | None = None,
    map_padding: float = 0.15,
    network_type: str = "all",
    width_cm: float = 10.0,
    height_cm: float = 8.0,
    x_cm: float | HorizontalAlign | None = "center",
    y_cm: float | VerticalAlign | None = "center",
    margin_x_cm: float | None = None,
    margin_y_cm: float | None = None,
    margin_cm: float = 0.0,
    bg_color: str = "#061529",
    color_short: str = "#a6a6a6",
    color_medium: str = "#676767",
    color_long: str = "#454545",
    color_xlong: str = "#d5d5d5",
    color_xxlong: str = "#ededed",
    width_short: float = 0.10,
    width_medium: float = 0.15,
    width_long: float = 0.25,
    width_xlong: float = 0.35,
    width_xxlong: float = 0.45,
    len_short_m: float = 100.0,
    len_medium_m: float = 200.0,
    len_long_m: float = 400.0,
    len_xlong_m: float = 800.0,
    trace_color: str | None = "#d62828",
    trace_opacity: float = 1.0,
    trace_width: int = 3,
    trace_points: list[Any] | None = None,
    border_color: str | None = None,
    border_opacity: float = 1.0,
    border_width: int = 0,
    dpi: int | None = None,
) -> Image.Image:
    """Ajoute une carte type routes (style beautifulMaps/createMapRoads.py).

    Passe soit ``points`` (liste de points GPX — le centre est calcule automatiquement),
    soit ``center_point`` (tuple (lat, lon) explicite).
    """
    resolved_dpi = dpi if dpi is not None else int(img.info.get("dpi", DPI))
    x0, y0, x1, y1 = _compute_box_rect(
        img, width_cm, height_cm, x_cm, y_cm, margin_x_cm, margin_y_cm, margin_cm, resolved_dpi
    )
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w <= 2 or box_h <= 2:
        return img

    if center_point is not None:
        resolved_center = center_point
        geo = []
    elif points:
        geo = [
            (_get_point_value(p, "lat", 0, 0.0), _get_point_value(p, "lon", 1, 0.0))
            for p in points
            if not (_get_point_value(p, "lat", 0, 0.0) == 0.0 and _get_point_value(p, "lon", 1, 0.0) == 0.0)
        ]
        if not geo:
            return img
        resolved_center = (
            (min(lat for lat, _ in geo) + max(lat for lat, _ in geo)) / 2.0,
            (min(lon for _, lon in geo) + max(lon for _, lon in geo)) / 2.0,
        )
    else:
        raise ValueError("add_map_v1: passe 'points' (GPX) ou 'center_point' (lat, lon).")

    # Auto-calcul du dist_m depuis l'etendue de la trace si non specifie
    if dist_m is None:
        if geo:
            lat_span_m = (max(lat for lat, _ in geo) - min(lat for lat, _ in geo)) * 111320
            lon_span_m = (
                (max(lon for _, lon in geo) - min(lon for _, lon in geo))
                * 111320
                * math.cos(math.radians(resolved_center[0]))
            )
            half_diag = math.sqrt(lat_span_m ** 2 + lon_span_m ** 2) / 2.0
            resolved_dist_m = max(500, int(half_diag * (1.0 + map_padding * 2)))
        else:
            resolved_dist_m = 5000
    else:
        resolved_dist_m = max(100, int(dist_m))

    try:
        ox = importlib.import_module("osmnx")
    except ImportError as exc:
        raise ImportError("osmnx est requis pour add_map_v1") from exc

    graph = ox.graph_from_point(
        resolved_center,
        dist=resolved_dist_m,
        retain_all=True,
        simplify=True,
        network_type=network_type,
    )

    edge_colors: list[str] = []
    edge_widths: list[float] = []
    for _, _, _, attrs in graph.edges(keys=True, data=True):
        length = float(attrs.get("length", 0.0))
        if length <= len_short_m:
            edge_colors.append(color_short)
            edge_widths.append(width_short)
        elif length <= len_medium_m:
            edge_colors.append(color_medium)
            edge_widths.append(width_medium)
        elif length <= len_long_m:
            edge_colors.append(color_long)
            edge_widths.append(width_long)
        elif length <= len_xlong_m:
            edge_colors.append(color_xlong)
            edge_widths.append(width_xlong)
        else:
            edge_colors.append(color_xxlong)
            edge_widths.append(width_xxlong)

    # Trace GPX eventuel: dessine directement dans l'axe matplotlib d'OSMnx.
    geo_to_draw: list[tuple[float, float]] = [
        (_get_point_value(p, "lat", 0, 0.0), _get_point_value(p, "lon", 1, 0.0))
        for p in (trace_points if trace_points is not None else (points or []))
        if not (_get_point_value(p, "lat", 0, 0.0) == 0.0 and _get_point_value(p, "lon", 1, 0.0) == 0.0)
    ]

    map_layer, _, _, _, _ = _render_osmnx_graph_image(
        graph=graph,
        width_px=box_w,
        height_px=box_h,
        dpi=resolved_dpi,
        bgcolor=bg_color,
        edge_colors=edge_colors,
        edge_widths=edge_widths,
        edge_alpha=1.0,
        trace_points=geo_to_draw,
        trace_color=trace_color,
        trace_opacity=trace_opacity,
        trace_width=max(1, int(trace_width)),
    )

    border_rgba = _resolve_rgba(border_color, border_opacity)
    if border_rgba is not None and border_width > 0:
        draw = ImageDraw.Draw(map_layer)
        draw.rectangle((0, 0, box_w - 1, box_h - 1), outline=border_rgba, width=border_width)

    img.alpha_composite(map_layer, (x0, y0))
    return img


def add_map_v2(
    img: Image.Image,
    points: list[Any] | None = None,
    center_point: tuple[float, float] | None = None,
    dist_m: int = 15000,
    width_cm: float = 10.0,
    height_cm: float = 8.0,
    x_cm: float | HorizontalAlign | None = "center",
    y_cm: float | VerticalAlign | None = "center",
    margin_x_cm: float | None = None,
    margin_y_cm: float | None = None,
    margin_cm: float = 0.0,
    bg_color: str = "#f6f8fa",
    water_color: str = "#72b1b1",
    major_water_color: str | None = None,
    minor_width: float = 0.5,
    major_width: float = 2.0,
    major_len_m: float = 400.0,
    border_color: str | None = None,
    border_opacity: float = 1.0,
    border_width: int = 0,
    dpi: int | None = None,
) -> Image.Image:
    """Ajoute une carte type rivieres/eaux (style beautifulMaps/createWaterMap.py).

    Passe soit ``points`` (liste de points GPX — le centre est calcule automatiquement),
    soit ``center_point`` (tuple (lat, lon) explicite).
    """
    resolved_dpi = dpi if dpi is not None else int(img.info.get("dpi", DPI))
    x0, y0, x1, y1 = _compute_box_rect(
        img, width_cm, height_cm, x_cm, y_cm, margin_x_cm, margin_y_cm, margin_cm, resolved_dpi
    )
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w <= 2 or box_h <= 2:
        return img

    if center_point is not None:
        resolved_center = center_point
    elif points:
        geo = [
            (_get_point_value(p, "lat", 0, 0.0), _get_point_value(p, "lon", 1, 0.0))
            for p in points
            if not (_get_point_value(p, "lat", 0, 0.0) == 0.0 and _get_point_value(p, "lon", 1, 0.0) == 0.0)
        ]
        if not geo:
            return img
        resolved_center = (
            (min(lat for lat, _ in geo) + max(lat for lat, _ in geo)) / 2.0,
            (min(lon for _, lon in geo) + max(lon for _, lon in geo)) / 2.0,
        )
    else:
        raise ValueError("add_map_v2: passe 'points' (GPX) ou 'center_point' (lat, lon).")

    try:
        nx = importlib.import_module("networkx")
        ox = importlib.import_module("osmnx")
    except ImportError as exc:
        raise ImportError("osmnx et networkx sont requis pour add_map_v2") from exc

    g1 = ox.graph_from_point(
        resolved_center,
        dist=max(100, int(dist_m)),
        dist_type="bbox",
        network_type="all",
        custom_filter='["natural"~"water"]',
    )
    g2 = ox.graph_from_point(
        resolved_center,
        dist=max(100, int(dist_m)),
        dist_type="bbox",
        network_type="all",
        custom_filter='["waterway"~"river"]',
    )
    graph = nx.compose(g1, g2)

    strong_color = major_water_color if major_water_color is not None else water_color
    edge_colors: list[str] = []
    edge_widths: list[float] = []
    for _, _, _, attrs in graph.edges(keys=True, data=True):
        length = float(attrs.get("length", 0.0))
        if length > major_len_m:
            edge_colors.append(strong_color)
            edge_widths.append(major_width)
        else:
            edge_colors.append(water_color)
            edge_widths.append(minor_width)

    map_layer = _render_osmnx_graph_image(
        graph=graph,
        width_px=box_w,
        height_px=box_h,
        dpi=resolved_dpi,
        bgcolor=bg_color,
        edge_colors=edge_colors,
        edge_widths=edge_widths,
        edge_alpha=1.0,
    )

    border_rgba = _resolve_rgba(border_color, border_opacity)
    if border_rgba is not None and border_width > 0:
        draw = ImageDraw.Draw(map_layer)
        draw.rectangle((0, 0, box_w - 1, box_h - 1), outline=border_rgba, width=border_width)

    img.alpha_composite(map_layer, (x0, y0))
    return img


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_poster(img: Image.Image, path: str | Path, dpi: int | None = None) -> Path:
    """Sauvegarde l'image en PNG avec les metadonnees DPI.

    Args:
        img:  Image PIL.
        path: Chemin de sortie (.png).
        dpi:  DPI pour les metadonnees (lit img.info['dpi'] si None).

    Returns:
        Chemin du fichier ecrit.
    """
    resolved_dpi = dpi if dpi is not None else int(img.info.get("dpi", DPI))
    out = Path(path)
    img.convert("RGB").save(out, format="PNG", dpi=(resolved_dpi, resolved_dpi))
    return out
