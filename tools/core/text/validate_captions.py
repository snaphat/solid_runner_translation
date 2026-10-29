"""
This module validates translated battle captions against the single-line caption strip.

validate_combat_captions() selects caption records by their original ROM offsets,
wraps their English text using the candidate font, and rejects line breaks,
page breaks, embedded stream terminators, and text wider than the strip. Each
record's width can impose a smaller limit. The function returns the number of captions checked or raises
ValueError with collected layout failures. It does not modify the ROM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from core.text import layout_text, measure_text
from core.text.get_text_limits import is_combat_caption

if TYPE_CHECKING:
    from core.text.insert_text import CompleteInsertionRecord


def validate_combat_captions(
    rom: bytes | bytearray, messages: Iterable[CompleteInsertionRecord]
) -> int:
    """
    Reject multiline captions and embedded terminators before writing an English ROM.

    Measures runtime insertions against the candidate font and preserves the
    four-pixel margin within the 240-pixel strip.

    Args:
        rom (bytes | bytearray): Candidate ROM containing the installed
            caption font.
        messages (Iterable[CompleteInsertionRecord]): Compiler insertion records
            containing original start/end offsets, English text, mode, optional
            width, pointers, and address operands. Start offsets must be integers
            within 4 MiB LoROM capacity; booleans are rejected. Non-caption
            records are skipped. Caption width must be None or a positive
            integer; booleans and other nonintegers are rejected. None uses the
            strip limit, and larger integer widths are capped at that limit.

    Returns:
        int: Number of combat-caption entries checked successfully.

    Raises:
        TypeError: A record's start offset is not an integer or is a boolean.
        ValueError: A start offset is outside LoROM capacity, a caption width
            is neither None nor a positive integer, or a combat caption wraps,
            contains a page break or an FD/FE/FF terminator, or exceeds its
            strip allowance.
    """
    errors: list[str] = []
    checked = 0

    for start, _, text, _, width, _, _ in messages:
        if not is_combat_caption(start):
            continue

        # Validate before capping the width: min() could otherwise turn an
        # oversized fractional value or infinity into a valid integer limit.
        if width is not None and (type(width) is not int or width <= 0):
            raise ValueError(f"{start:06X}: caption width must be None or a positive integer")

        checked += 1

        # The combat renderer has no second caption row. Enforce both the entry
        # width and the global strip width before rejecting any resulting line/page break.
        limit = min(width if width is not None else layout_text.MAX_W, layout_text.MAX_W)
        try:
            laid = layout_text.wrap_text(rom, text, limit, paginate=False)
        except AssertionError as error:
            errors.append(f"{start:06X}: {error}")
            continue

        # Inspect complete tokens so terminator-valued command operands and
        # extended glyph bytes remain valid payload data.
        if any(
            token in (b"\x00", b"\xfd", b"\xfe", b"\xff")
            for token, _ in measure_text.iter_text_tokens(laid)
        ):
            errors.append(
                f"{start:06X}: combat caption must fit one {limit}px line "
                f"without embedded terminators: {text!r}"
            )

    if errors:
        raise ValueError("Combat caption layout failed:\n" + "\n".join(errors))

    return checked
