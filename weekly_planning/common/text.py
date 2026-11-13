"""
Text normalisation: worker-name keys, module (tool) codes and the cleaning of Excel cell text.
"""
from __future__ import annotations

import re
import unicodedata


_WS_RE = re.compile(r"\s+")


_ZERO_WIDTH = {"​", "‌", "‍", "﻿"}


TOOLCODE_RE = re.compile(r"\d{3}[A-Z0-9]\d{5}A")


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("　", " ")
    for z in _ZERO_WIDTH:
        s = s.replace(z, "")
    return s


def norm_name(s: str) -> str:
    s = clean_text(s)
    s = _WS_RE.sub("", s).strip()
    return s


def name_key(s) -> str:
    """Stable key of a worker name: NFKC, no whitespace (worker ids are renumbered by the decoder, names are not)."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(s or "")))


def tool_code(text):
    m = TOOLCODE_RE.search(unicodedata.normalize("NFKC", str(text or "")).upper())
    return m.group(0) if m else None


def extract_tool_code(s: str):
    if not isinstance(s, str):
        return None
    m = TOOLCODE_RE.search(s)
    return m.group(0) if m else None


def normalize_tool_code_in_text(s: str) -> str:
    r"""Rewrite a full-width tool code (Python's \d also matches "５") to its
    half-width form in place, so "５３０N00000A_..." and "530N00000A_..." are
    the same module everywhere downstream (cut rules replace the code inside
    the text, so the text itself has to carry the normalised code)."""
    m = TOOLCODE_RE.search(s) if isinstance(s, str) else None
    if not m:
        return s
    norm = unicodedata.normalize("NFKC", m.group(0))
    return s if norm == m.group(0) else s[:m.start()] + norm + s[m.end():]
