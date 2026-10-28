"""
This module supplies text display limits and checks that battle captions fit on one line.

The tables map original-ROM text offsets to categories such as item descriptions,
job reports, and chapter headings, with width and row limits for their displays.
Layout validation uses these limits when checking English translations.

validate_combat_captions() takes ROM font bytes and translation insertion
records, measures and wraps each battle caption, and raises ValueError if any
caption exceeds its width or contains a line or page break. It returns the
number of captions checked and does not change the ROM or inspect screenshots.
"""

from __future__ import annotations

from typing import Iterable

from core.text import layout_text, measure_text
from core.text.insert_text import CompleteInsertionRecord

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


def classify_text_window(a: int) -> str | None:
    """
    Find the known system-text category for a ROM file offset.

    Args:
        a (int): Original text-entry file offset.

    Returns:
        str | None: Known system-text window category, or None outside
            categorized ranges.
    """
    for name, lo, hi in CATS:
        if lo <= a < hi:
            return name

    return None


def is_combat_caption(offset: int) -> bool:
    """
    Check whether an original slot uses the combat caption strip.

    Args:
        offset (int): Zero-based byte offset in the unheadered ROM.

    Returns:
        bool: Whether the slot uses the single-line combat caption strip.
    """
    return (
        0x085CBF <= offset <= 0x0863C1
        or 0x019B79 <= offset <= 0x019C0C
        or 0x08646D <= offset <= 0x086494
        or offset in (0x080366, 0x086715, 0x086733)
    )


def validate_combat_captions(
    rom: bytes | bytearray, messages: Iterable[CompleteInsertionRecord]
) -> int:
    """
    Reject wrapping and page breaks before writing an English ROM.

    Measures runtime insertions against the candidate font and preserves the
    four-pixel margin within the 240-pixel strip.

    Args:
        rom (bytes | bytearray): Candidate ROM containing the installed
            caption font.
        messages (Iterable[CompleteInsertionRecord]): Compiler insertion records, with start
            offset and English text.

    Returns:
        int: Number of combat-caption entries checked successfully.

    Raises:
        ValueError: A combat caption wraps, contains a page break, or exceeds
            its strip allowance.
    """
    errors: list[str] = []
    checked = 0

    for start, _, text, _, width, _, _ in messages:
        if not is_combat_caption(start):
            continue

        checked += 1

        # The combat renderer has no second caption row. Enforce both the entry
        # width and the global strip width before rejecting any resulting line/page break.
        limit = min(width if width is not None else layout_text.MAX_W, layout_text.MAX_W)
        try:
            laid = layout_text.wrap_text(rom, text, limit, paginate=False)
        except AssertionError as error:
            errors.append(f"{start:06X}: {error}")
            continue

        if any(token in (b"\x00", b"\xfd") for token, _ in measure_text.iter_text_tokens(laid)):
            errors.append(f"{start:06X}: combat caption must fit one {limit}px line: {text!r}")

    if errors:
        raise ValueError("Combat caption layout failed:\n" + "\n".join(errors))

    return checked
