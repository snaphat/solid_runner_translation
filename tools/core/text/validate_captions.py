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
    rom: bytes | bytearray,
    messages: Iterable[CompleteInsertionRecord],
    *,
    name_width: int | None = None,
) -> int:
    """
    Reject multiline captions and embedded terminators before writing an English ROM.

    Battle captions have one visible row. Measure each caption using the font
    installed in rom and allow at most 236 pixels of the 240-pixel strip.
    A record can request a smaller limit. Captions outside the known battle
    text offsets are skipped.

    A caption can contain a command that inserts an item or opponent name.
    The game chooses that name later. Supply the longest possible name's pixel
    width through name_width so this check reserves room for that choice.

    Args:
        rom (bytes | bytearray): Candidate ROM containing the installed
            caption font.
        messages (Iterable[CompleteInsertionRecord]): Translation records in
            (start, end, text, mode, width, pointers, address_operands) order.
            start identifies which entries are battle captions. text supplies
            their English wording. width is None or a positive integer pixel
            limit; None uses 236 pixels and larger values are capped at 236.
            start must be an integer from 0 through 0x3FFFFF. Booleans are
            rejected for offsets and widths. Other fields are retained in the
            record for compilation and are not used by this check.
        name_width (int | None, optional): Keyword-only pixel reservation for
            each F8/12 or F8/18 item/opponent-name command in a caption. Use
            measure_text.measure_name_width() with the same ROM font and name
            list used for compilation. The value must be a nonnegative integer.
            None is allowed when the checked captions contain no such commands.

    Returns:
        int: Number of combat-caption entries checked successfully.

    Raises:
        TypeError: A record's start offset is not an integer or is a boolean.
        ValueError: A start offset is outside LoROM capacity, a caption width
            is neither None nor a positive integer, or a combat caption wraps,
            contains a page break or an FD/FE/FF terminator, or exceeds its
            strip allowance, or a required system-name width is absent or invalid.

    Examples:
        With rom containing the installed font and messages holding the
        translation records, load and measure the name candidates once:

        from core.text import measure_text, read_translation_files

        names = read_translation_files.load_names_for_dialogue_layout()
        width = measure_text.measure_name_width(rom, names)
        validate_combat_captions(rom, messages, name_width=width)
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

        # Check the same inserted-name reservation that compilation will use.
        # The chosen item/opponent name must fit without creating a second row.
        # The record's width can reduce, but cannot enlarge, the caption strip.
        limit = min(width if width is not None else layout_text.MAX_W, layout_text.MAX_W)
        try:
            laid = layout_text.wrap_text(rom, text, limit, paginate=False, name_width=name_width)
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
