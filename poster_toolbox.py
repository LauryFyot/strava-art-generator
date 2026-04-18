"""poster_toolbox.py — Fonctions utilitaires de composition d'affiche."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from PIL import Image, ImageColor, ImageDraw, ImageFont


DPI = 300
HorizontalAlign = Literal["left", "center", "right"]
VerticalAlign = Literal["top", "center", "bottom"]
TextAlign = Literal["left", "center", "right", "justify"]
TextVerticalAlign = Literal["top", "center", "bottom"]


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
# 1. Creer un poster vierge
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


# ---------------------------------------------------------------------------
# 2. Changer la couleur de fond
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. Ajouter une textbox
# ---------------------------------------------------------------------------

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
