"""
This module calculates pixel advances from ROM font bitmaps and supplied name widths.

measure_text_width() sums glyph advances and reserves space for runtime
insertions. Player names and numbers use fixed maximum lengths; callers supply
the widest active system-name width for item/opponent insertions. Obtain that
width with measure_name_width() using names already loaded by the caller.

The functions do not read translation files or modify the ROM. A measurement
cache belongs to one set of ROM/font inputs. System-name cache entries are
refreshed from the supplied width so changing it cannot reuse an old value.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Literal, cast

from core.graphics.font_tables import MAIN_FONT, VWF_FONT
from core.text.encode_text import encode_text
from core.text.read_text_tokens import decode_glyph_index, parse_encoded_text

GAP, SPACEW = 1, 4  # Pixels: inter-glyph gap and space advance; match asm/vwf.asm.


def iter_text_tokens(b: bytes) -> Iterator[tuple[bytes, int | Literal["name"] | None]]:
    """
    Separate glyphs and commands so layout can measure text without splitting operands.

    One-byte and two-byte glyph codes produce font indices. Insertion commands
    produce a name marker for runtime-width reservation; other commands and
    terminators have no glyph index. Every yielded byte string includes the
    complete command operands.

    Args:
        b (bytes): Encoded game text, including any control-code operands.

    Yields:
        tuple[bytes, int | Literal['name'] | None]: Pair (token bytes, glyph index); the second item
            is a font index (or -1 for newline), "name" for a dynamic insertion,
            or None for another control or terminator.
    """
    for token in parse_encoded_text(b):
        if token[0] == 0xF8:
            # These commands insert text whose width is known only at runtime.
            glyph: int | Literal["name"] | None = (
                "name" if token[1] in (0x0C, 0x12, 0x18, 0x04) else None
            )
        elif token[0] >= 0xFD:
            # Retain page/record boundaries for layout checks without treating
            # them as drawable characters.
            glyph = None
        else:
            glyph = decode_glyph_index(token, 0)

        yield token, glyph


def measure_glyph_advance(rom: bytes | bytearray, idx: int) -> int:
    """
    Measure a glyph advance using the appropriate ROM font table.

    Args:
        rom (bytes | bytearray): Font tables used to measure the selected
            glyph.
        idx (int): Zero-based font glyph index; indices below 256 select the
            relocated VWF table. Booleans and other nonintegers are rejected.

    Returns:
        int: Ink width plus one shadow pixel and the inter-glyph gap. Empty
            VWF glyphs advance SPACEW pixels; empty native glyphs advance one.

    Raises:
        TypeError: idx is not an integer or is a boolean.
        ValueError: The glyph index is negative or its font record lies outside the ROM.
    """
    if type(idx) is not int:
        raise TypeError("Glyph index must be an integer; booleans are not indices")
    base = VWF_FONT if idx < 256 else MAIN_FONT
    if idx < 0 or base + (idx + 1) * 22 > len(rom):
        raise ValueError("Glyph index does not identify a complete ROM font record")
    # Combine all eleven bitmap rows to find the glyph's leftmost and rightmost ink.
    acc = 0
    for r in range(11):
        o = base + idx * 22 + r * 2
        acc |= rom[o] | rom[o + 1] << 8

    if not acc:
        return SPACEW if idx < 256 else 1

    # The font uses the high bits for leftmost pixels; trim blank columns on
    # both sides, then add one shadow pixel and the renderer's inter-glyph gap.
    lz = 16 - acc.bit_length()
    width = 16 - ((acc & -acc).bit_length() - 1)
    return width - lz + 1 + GAP


WIDE = 13  # Pixel advance reserved per player-entered name glyph, including gap.
DIGIT_W = 9  # Maximum digit advance in pixels, including gap; reserves number width.


def _validate_name_width(name_width: int | None) -> None:
    """
    Reject invalid explicit system-name widths.

    Args:
        name_width (int | None): Nonnegative pixel width, or None when absent.

    Raises:
        ValueError: name_width is neither None nor a nonnegative integer;
            booleans are rejected.
    """
    # bool is an int subclass; True must not become a one-pixel reservation.
    if name_width is not None and (type(name_width) is not int or name_width < 0):
        raise ValueError("System-name width must be a nonnegative integer")


def measure_insertion_width(code: bytes, *, name_width: int | None = None) -> int:
    """
    Measure the maximum width reserved for a runtime name or number insertion.

    Player-entered names and numeric values use the native maximum lengths. Other
    name insertions use the caller's widest active system-name measurement.
    Word wrapping uses the result before the actual inserted value is known.

    Args:
        code (bytes): Complete F8 insertion token, including opcode and
            operands.
        name_width (int | None, optional): Pixel width of the longest translated
            item/opponent name. F8/12 and F8/18 use this value as their reserved
            advance. Supply a nonnegative integer measured by measure_name_width().
            None is allowed for player-name and numeric insertion commands.

    Returns:
        int: Horizontal text advance in pixels.

    Raises:
        ValueError: code is not one supported insertion command, name_width is
            invalid, or an item/opponent insertion has no supplied name width.
    """
    _validate_name_width(name_width)
    if (
        len(parse_encoded_text(code)) != 1
        or code[0] != 0xF8
        or code[1] not in (0x04, 0x0C, 0x12, 0x18)
    ):
        raise ValueError("Width measurement requires one complete name or number insertion")
    op = code[1]
    if op == 0x0C:  # player-entered names: 5 chars, Runner name 6
        # Saved name contents are unknown during layout. Reserve the buffer's
        # full character capacity using the maximum name-glyph advance.
        return WIDE * (6 if code[2] == 0x02 else 5)

    if op == 0x04:  # number: arg 1 = value size in bytes
        # Unsigned 1/2/3-byte values require at most 3/5/8 decimal digits.
        # The fallback reserves ten digits for the larger numeric case.
        return DIGIT_W * {1: 3, 2: 5, 3: 8}.get(code[2], 10)

    # F8/12 and F8/18 select names during gameplay. Their width comes from the
    # caller's complete translated-name list, measured using this build's font.
    if name_width is None:
        raise ValueError("Item/opponent insertions require an explicit system-name width")
    return name_width


def measure_text_width(
    rom: bytes | bytearray,
    b: bytes,
    insert_sizes: dict[bytes, int] | None = None,
    *,
    name_width: int | None = None,
) -> int:
    """
    Measure text with worst-case dynamic insertion widths.

    Args:
        rom (bytes | bytearray): Font tables used to measure the encoded
            text.
        b (bytes): Encoded game text, including any control-code operands.
        insert_sizes (dict[bytes, int] | None, optional): Mutable token-to-width
            cache for repeated measurements in one layout. Player-name and
            numeric reservations are reused. Item/opponent reservations are
            refreshed from name_width on each call. None creates a fresh cache.
        name_width (int | None, optional): Pixel advance added for each F8/12 or
            F8/18 item/opponent-name command in b. Obtain the longest name's
            advance with measure_name_width() using the same ROM font. The value
            must be a nonnegative integer. None is allowed if b contains no
            item/opponent-name commands.

    Returns:
        int: Horizontal text advance in pixels.

    Raises:
        ValueError: name_width is invalid or an item/opponent insertion has no
            supplied name width, or an encoded token/font record is invalid.
    """
    _validate_name_width(name_width)
    # A layout may measure many candidate lines. Cache their insertions only
    # within that layout, so another ROM, font or script edit is measured afresh.
    insert_sizes = {} if insert_sizes is None else insert_sizes
    width = 0

    for token, glyph in iter_text_tokens(b):
        # Runtime insertions have no single font glyph. Reserve their computed
        # maximum width and cache it across the candidate lines of this layout.
        if glyph == "name":
            # System-name reservations depend on this call's explicit width.
            # Refresh them even if the caller retained a cache from another width.
            if token[:2] in (b"\xf8\x12", b"\xf8\x18") or token not in insert_sizes:
                insert_sizes[token] = measure_insertion_width(token, name_width=name_width)

            width += insert_sizes[token]
        # Commands and terminators contribute no glyph advance; literal glyphs
        # use the same proportional or fixed-font measurement as compilation.
        elif isinstance(glyph, int) and glyph >= 0:
            width += measure_glyph_advance(rom, glyph)

    return width


def measure_name_width(rom: bytes | bytearray, names: Iterable[str]) -> int:
    """
    Measure the widest translated name that dialogue may insert during gameplay.

    Dialogue can contain commands that substitute an item or opponent name.
    Before the game chooses that name, wrapping reserves the largest possible
    width. This function measures every supplied name using the candidate ROM's
    font and returns the largest advance. Load the strings with
    read_translation_files.load_names_for_dialogue_layout() before calling it.

    Args:
        rom (bytes | bytearray): Font tables used to measure literal glyphs.
        names (Iterable[str]): Nonempty collection of translated names. Each
            name must contain only literal glyphs; controls and newlines are
            rejected. Supply a collection, not a single string.

    Returns:
        int: Maximum summed glyph advance among the supplied names.

    Raises:
        TypeError: names is not an iterable collection of strings, or is a
            string/byte buffer instead of a collection.
        KeyError: A name contains a character with no game encoding.
        ValueError: No names are supplied, a name has control markup or a
            newline, or a font record lies outside the ROM.
    """
    # A bare string would iterate character by character and accidentally
    # measure the widest letter instead of the widest complete name.
    if isinstance(cast(object, names), (str, bytes, bytearray)):
        raise TypeError("System names must be supplied as a collection of strings")
    # None distinguishes no records from a supplied name with zero advance.
    widest: int | None = None
    for name in names:
        if not isinstance(cast(object, name), str):
            raise TypeError("Each system name must be a string")
        width = 0
        for _, glyph in iter_text_tokens(encode_text(name)):
            # Controls could insert more text or move the cursor. Reject them
            # so each candidate contributes the width of one literal name.
            if not isinstance(glyph, int) or glyph < 0:
                raise ValueError("System names must contain only literal glyphs")
            width += measure_glyph_advance(rom, glyph)
        # The game can choose any candidate; reserve the largest whole-name
        # advance, including each glyph's shadow and spacing.
        widest = width if widest is None else max(widest, width)
    if widest is None:
        raise ValueError("At least one active system name is required")
    return widest
