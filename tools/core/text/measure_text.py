"""
This module calculates pixel advances for encoded text using the ROM's font bitmaps.

iter_text_tokens() distinguishes drawable glyphs, control commands, and runtime
insertions. measure_glyph_advance() measures one glyph; measure_text_width()
sums advances and reserves worst-case widths for inserted names and numbers.
It returns a total advance, rather than the maximum width of separate lines.
Layout and validation callers use these measurements before drawing the text.

Player and Runner names use fixed maximum lengths. Item and opponent insertions
use the widest active name loaded from the system translation source. Optional
insertion-width caches belong to one set of ROM, font, and translation inputs.
The measurement functions do not modify the ROM or translation records.
"""

from __future__ import annotations

from typing import Iterator, Literal

from core.text.encode_text import encode_text
from core.text.read_text_tokens import decode_glyph_index, parse_encoded_text

FONT = 0x170000  # Native font file offset; each glyph has eleven 16-bit rows.
VWF_FONT = 0x3F0000  # glyphs 0-255 as the VWF draws them
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
    base = VWF_FONT if idx < 256 else FONT
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


def measure_insertion_width(rom: bytes | bytearray, code: bytes) -> int:
    """
    Measure the maximum width reserved for a runtime name or number insertion.

    Player-entered names and numeric values use the native maximum lengths. Other
    name insertions use the widest active system-name entry. Word wrapping uses
    the result before the actual inserted value is known.

    Args:
        rom (bytes | bytearray): Font tables used to reserve space for
            inserted text.
        code (bytes): Complete F8 insertion token, including opcode and
            operands.

    Returns:
        int: Horizontal text advance in pixels.

    Raises:
        ValueError: code is not one complete supported insertion command, or no
            active system names are available for an item/opponent insertion.
    """
    if (
        len(parse_encoded_text(code)) != 1
        or code[0] != 0xF8
        or code[1] not in (0x04, 0x0C, 0x12, 0x18)
    ):
        raise ValueError("Width measurement requires one complete name or number insertion")
    op = code[1]
    if op == 0x0C:  # player-entered names: 5 chars, Runner name 6
        return WIDE * (6 if code[2] == 0x02 else 5)

    if op == 0x04:  # number: arg 1 = value size in bytes
        return DIGIT_W * {1: 3, 2: 5, 3: 8}.get(code[2], 10)

    from core.text import read_translation_files as script_files

    return max(
        sum(
            measure_glyph_advance(rom, g)
            for _, g in iter_text_tokens(encode_text(e["en"]))
            if isinstance(g, int)
        )
        for e in script_files.load("system")
        if e.get("mode") == "n" and e.get("en") and not e.get("skip")
    )  # opponent name / item name


def measure_text_width(
    rom: bytes | bytearray, b: bytes, insert_sizes: dict[bytes, int] | None = None
) -> int:
    """
    Measure text with worst-case dynamic insertion widths.

    Args:
        rom (bytes | bytearray): Font tables used to measure the encoded
            text.
        b (bytes): Encoded game text, including any control-code operands.
        insert_sizes (dict[bytes, int] | None, optional): Mutable cache for
            this layout. Discard it after changing ROM, font or translation
            inputs; None creates a fresh cache.

    Returns:
        int: Horizontal text advance in pixels.
    """
    # A layout may measure many candidate lines. Cache their insertions only
    # within that layout, so another ROM, font or script edit is measured afresh.
    insert_sizes = {} if insert_sizes is None else insert_sizes
    width = 0

    for token, glyph in iter_text_tokens(b):
        # Runtime insertions have no single font glyph. Reserve their computed
        # maximum width and cache it across the candidate lines of this layout.
        if glyph == "name":
            if token not in insert_sizes:
                insert_sizes[token] = measure_insertion_width(rom, token)

            width += insert_sizes[token]
        # Commands and terminators contribute no glyph advance; literal glyphs
        # use the same proportional or fixed-font measurement as compilation.
        elif isinstance(glyph, int) and glyph >= 0:
            width += measure_glyph_advance(rom, glyph)

    return width
