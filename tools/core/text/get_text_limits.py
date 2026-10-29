"""
This module supplies display limits and classifies text entries by original ROM offset.

The tables describe known text categories, default and overridden line widths,
and visible row limits. classify_text_window() identifies a system-text category;
is_combat_caption() identifies entries using the single-line battle strip.
Layout validation and screenshot checks use these classifications.

estimate_japanese_line_width() estimates source-line width from character counts
and a fixed given-name reservation. It does not use ROM font bitmaps. The module
contains lookup data and helpers; it does not wrap text or modify ROM bytes.
"""

from __future__ import annotations

# Maximum unheadered image size supported by the standard LoROM mapping.
ROM_CAPACITY = 0x400000

# System-text windows classified by original entry offset. Each (name, lo, hi)
# covers a half-open ROM file-offset range; its category selects layout limits.
CATS = [
    ("item descriptions", 0x196B5, 0x19A60),
    ("weapon descriptions", 0x1A8D0, 0x1AE60),
    ("job board", 0x3CA00, 0x3CCE0),
    ("news", 0x3CCE0, 0x3DAEE),
    ("rumors", 0x3DAEE, 0x3E742),
    ("profiles", 0x3E742, 0x3EEF4),
    ("encyclopedia", 0x3EF56, 0x3F610),
    ("chapter titles", 0x916A3, 0x91720),
    ("chapter summaries", 0x91739, 0x919C7),
    ("armor descriptions", 0x9B376, 0x9B420),
    ("map labels", 0xF3978, 0xF3C40),
    ("terminal counters", 0x38F00, 0x3A980),
    ("garage labels", 0x9B2A0, 0x9B330),
]
MAXW = 228  # Default usable line width in pixels; category/entry limits override it below.

# Category-specific line widths in pixels, overriding MAXW. Measured or derived window widths:
# weapon descriptions share the 224 px item description box; chapter titles go in the one-line
# mission banner: "Mission " (45 px), the title, then "Complete" (53 px), each a separate draw that
# starts on a 12 px cell. The title (with its quotes and a trailing space) starts at 48, its end is
# rounded up to a cell, and "Complete" must end by 240, so a title may be at most 132 px
OVERRIDE = {
    "weapon descriptions": 224,
    "item descriptions": 224,
    "chapter titles": 132,
    "garage labels": 48,
    "chapter summaries": 188,
}

# Category-specific visible row limits for non-paginated prose.
# Mission results retain the original three-line layout.
LINE_LIMIT = {"chapter summaries": 3}

# Per-entry line widths in pixels, overriding category limits. Keys are hex ROM offsets. Job board
# client names follow "Client:" at x=120 in the job detail box (edge at x=248, seen on screen)
ENTRY_W = {
    i: 126 for i in ("03CA16", "03CA86", "03CADC", "03CB36", "03CB88", "03CBEA", "03CC46", "03CC9B")
}


def estimate_japanese_line_width(line: str) -> int:
    """
    Estimate line width, treating {名} as five ASCII-width characters.

    After that substitution, count eight pixels per ASCII character and twelve
    per non-ASCII character. Other markup is counted as literal characters.

    The {名} given-name token reserves five ASCII characters.

    Args:
        line (str): Decoded Japanese text without a newline; {名} reserves five
            ASCII characters.

    Returns:
        int: Estimated pixel width, including the five-character reservation for
            the dynamic given-name token.
    """
    return sum(8 if c.isascii() else 12 for c in line.replace("{名}", "XXXXX"))


def _validate_rom_offset(offset: int) -> None:
    """
    Require a byte offset within standard LoROM capacity.

    Args:
        offset (int): Zero-based file offset in an unheadered ROM.

    Raises:
        TypeError: offset is not an integer or is a boolean.
        ValueError: offset is negative or at least ROM_CAPACITY.
    """
    if type(offset) is not int:
        raise TypeError("Text offset must be an integer; booleans are not offsets")
    if not 0 <= offset < ROM_CAPACITY:
        raise ValueError("Text offset must lie within the 4 MiB LoROM address range")


def classify_text_window(a: int) -> str | None:
    """
    Find the known system-text category for a ROM file offset.

    Args:
        a (int): Original text-entry file offset from 0 through 0x3FFFFF.
            Booleans and other nonintegers are rejected.

    Returns:
        str | None: Known system-text window category, or None outside
            categorized ranges.

    Raises:
        TypeError: a is not an integer or is a boolean.
        ValueError: a is outside standard LoROM capacity.
    """
    _validate_rom_offset(a)
    for name, lo, hi in CATS:
        if lo <= a < hi:
            return name

    return None


def is_combat_caption(offset: int) -> bool:
    """
    Check whether an original slot uses the combat caption strip.

    Args:
        offset (int): Zero-based byte offset from 0 through 0x3FFFFF in the
            unheadered ROM. Booleans and other nonintegers are rejected.

    Returns:
        bool: Whether the slot uses the single-line combat caption strip.

    Raises:
        TypeError: offset is not an integer or is a boolean.
        ValueError: offset is outside standard LoROM capacity.
    """
    _validate_rom_offset(offset)
    return (
        0x085CBF <= offset <= 0x0863C1
        or 0x019B79 <= offset <= 0x019C0C
        or 0x08646D <= offset <= 0x086494
        or offset in (0x080366, 0x086715, 0x086733)
    )
