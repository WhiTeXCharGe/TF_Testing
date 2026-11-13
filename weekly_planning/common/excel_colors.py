"""
Cell colours of the SU_Others workbook: grey = personal business, red = unavailable (theme colours and tints included).
"""
from __future__ import annotations

import colorsys

from openpyxl.styles.colors import COLOR_INDEX


GREY_RGB_LAST6 = {"A6A6A6", "BFBFBF", "D9D9D9", "808080"}


RED_RGB_LAST6 = {"FF0000"}


GREY_INDEXED = {15, 22, 23, 24, 25, 26, 27, 28, 29}


RED_INDEXED = {10}


IGNORE_WHITE_TEXT = {"FI", "FO"}


def _color_to_rgb6(color):
    if color is None:
        return None
    ctype = getattr(color, "type", None)
    if ctype == "rgb":
        rgb = getattr(color, "rgb", None)
        return str(rgb).upper()[-6:] if rgb else None
    if ctype == "indexed":
        idx = getattr(color, "indexed", None)
        if idx is None:
            return None
        try:
            rgb = COLOR_INDEX[idx]
            return str(rgb).upper()[-6:] if rgb else None
        except Exception:
            return None
    return None


def _cell_fill_rgb6(cell):
    fill = getattr(cell, "fill", None)
    if fill is None:
        return None
    candidates = [getattr(fill, "fgColor", None), getattr(fill, "start_color", None),
                  getattr(fill, "bgColor", None), getattr(fill, "end_color", None)]
    for col in candidates:
        rgb6 = _color_to_rgb6(col)
        if rgb6:
            return rgb6
    return None


def _cell_fill_indexed(cell):
    fill = getattr(cell, "fill", None)
    if fill is None:
        return None
    for col in [getattr(fill, "fgColor", None), getattr(fill, "start_color", None),
                getattr(fill, "bgColor", None), getattr(fill, "end_color", None)]:
        if col is None:
            continue
        if getattr(col, "type", None) == "indexed":
            return getattr(col, "indexed", None)
    return None


def _is_theme_grey(color) -> bool:
    if color is None:
        return False
    if getattr(color, "type", None) != "theme":
        return False
    tint = getattr(color, "tint", None)
    theme = getattr(color, "theme", None)
    if tint is not None:
        try:
            t = float(tint)
            if -0.6 <= t <= 0.6:
                return True
        except Exception:
            pass
    if theme is not None:
        return True
    return False


def is_red_cell(cell) -> bool:
    rgb6 = _cell_fill_rgb6(cell)
    if rgb6 and rgb6 in RED_RGB_LAST6:
        return True
    idx = _cell_fill_indexed(cell)
    if idx is not None and idx in RED_INDEXED:
        return True
    return False


def is_grey_cell(cell) -> bool:
    rgb6 = _cell_fill_rgb6(cell)
    if rgb6 and rgb6 in GREY_RGB_LAST6:
        return True
    idx = _cell_fill_indexed(cell)
    if idx is not None and idx in GREY_INDEXED:
        return True
    fill = getattr(cell, "fill", None)
    if fill is not None:
        for col in [getattr(fill, "fgColor", None), getattr(fill, "start_color", None),
                    getattr(fill, "bgColor", None), getattr(fill, "end_color", None)]:
            if _is_theme_grey(col):
                return True
    return False


# Excel theme index -> clrScheme element (lt/dk pairs are swapped vs. XML order)
_THEME_INDEX_TO_SLOT = ["lt1", "dk1", "lt2", "dk2", "accent1", "accent2", "accent3",
                        "accent4", "accent5", "accent6", "hlink", "folHlink"]


def load_theme_palette(wb):
    """{theme_index: "RRGGBB"} from the workbook's theme1.xml (empty if unreadable)."""
    raw = getattr(wb, "loaded_theme", None)
    if not raw:
        return {}
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw)
        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        scheme = root.find(".//a:clrScheme", ns)
        if scheme is None:
            return {}
        slot_rgb = {}
        for child in scheme:
            slot = child.tag.split("}")[-1]
            srgb = child.find("a:srgbClr", ns)
            sys_c = child.find("a:sysClr", ns)
            if srgb is not None:
                slot_rgb[slot] = srgb.get("val", "").upper()
            elif sys_c is not None:
                slot_rgb[slot] = sys_c.get("lastClr", "").upper()
        return {i: slot_rgb[s] for i, s in enumerate(_THEME_INDEX_TO_SLOT) if slot_rgb.get(s)}
    except Exception:
        return {}


def _apply_tint(rgb6: str, tint) -> str:
    try:
        t = float(tint or 0.0)
    except Exception:
        t = 0.0
    if not t:
        return rgb6
    import colorsys
    r, g, b = (int(rgb6[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l * (1.0 + t) if t < 0 else l * (1.0 - t) + t
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l)), s)
    return "".join(f"{round(c * 255):02X}" for c in (r, g, b))


def cell_color_hex(cell, theme_palette):
    """Solid fill colour of a cell as "RRGGBB", or None when it has no fill."""
    fill = getattr(cell, "fill", None)
    if fill is None or getattr(fill, "fill_type", None) in (None, "none"):
        return None
    col = getattr(fill, "fgColor", None)
    ctype = getattr(col, "type", None)
    if ctype == "rgb":
        rgb = str(getattr(col, "rgb", "") or "").upper()
        return rgb[-6:] if len(rgb) >= 6 and all(ch in "0123456789ABCDEF" for ch in rgb[-6:]) else None
    if ctype == "indexed":
        return _color_to_rgb6(col)
    if ctype == "theme":
        base = theme_palette.get(getattr(col, "theme", None))
        return _apply_tint(base, getattr(col, "tint", 0.0)) if base else None
    return None
