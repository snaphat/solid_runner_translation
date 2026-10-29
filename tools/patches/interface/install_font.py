"""
This module writes the editable Latin glyph artwork into the ROM font tables using the character codes expected by the text encoder.

Editable artwork is encoded into eleven-row glyph records. The relocated
variable-width table receives Latin text, while selected punctuation and naming
ligatures also update the native fixed-width table. Slot lookup uses the same
character codes as the text encoder.
"""

from __future__ import annotations

from typing import Mapping, Sequence, cast

from core.files.validate_binary_buffers import require_byte_buffer
from core.graphics.font_tables import MAIN_FONT, VWF_FONT
from core.graphics.load_font_artwork import MAIN_PUNCT, G
from core.text.encode_text import M


def find_glyph_slots() -> dict[str, int]:
    """
    Return artwork characters mapped to zero-based encoded font slots.

    Returns:
        dict[str, int]: Artwork characters or ligature tokens mapped to zero-based
            native font slots, one less than their encoded glyph codes.

    Raises:
        KeyError: A registered key in G has no encoding in M.
    """
    return {ch: M[ch] - 1 for ch in G}


def patch_font(rom: bytearray) -> None:
    """
    Install Latin glyphs in the proportional and native font tables.

    Build both 256-record tables in private buffers. The proportional table uses
    G artwork; the native table receives the three name ligatures and MAIN_PUNCT.
    Only after both preparations pass are their 256 * 22 bytes copied into rom.
    Records without replacement artwork are copied from the native font table;
    existing bytes in the relocated table are replaced, not preserved.

    Args:
        rom (bytearray): Candidate ROM containing the native and relocated font
            allocations; both tables are updated in place.

    Raises:
        TypeError: rom is not a mutable bytearray.
        ValueError: The ROM does not contain the complete relocated font allocation,
            or glyph/punctuation artwork has an invalid key or shape.
        KeyError: Required ligature artwork is absent, or a registered key has
            no entry in the encoder mapping.
    """
    require_byte_buffer(rom, (bytearray,))
    if len(rom) < VWF_FONT + 256 * 22:
        raise ValueError("ROM does not contain the complete relocated font allocation")
    # Keep a separate font copy for proportional rendering; fixed-width screens
    # continue to use the native font plus their required ligatures.
    font_bytes = 256 * 22
    native = bytearray(rom[MAIN_FONT : MAIN_FONT + font_bytes])
    proportional = native.copy()
    write_glyphs(proportional, 0, G)
    fixed_glyphs = {k: G[k] for k in ("{ki}", "{ne}", "{il}")}

    # Fixed-width choices and menus use the main font. English apostrophe, period
    # and comma codes reuse its Japanese ・ 。 、 slots, so install Latin artwork
    # there too, with the fixed-width letters' two-pixel weight and row-9 baseline.
    fixed_glyphs.update({ch: art.strip("\n").split("\n") for ch, art in MAIN_PUNCT.items()})
    write_glyphs(native, 0, fixed_glyphs)

    # Both font tables are ready before either is installed. Invalid artwork
    # cannot leave a partial font or spill a punctuation row into the next glyph.
    rom[VWF_FONT : VWF_FONT + font_bytes] = proportional
    rom[MAIN_FONT : MAIN_FONT + font_bytes] = native


def write_glyphs(rom: bytearray, base: int, glyphs: Mapping[str, Sequence[str]]) -> None:
    """
    Install supplied bitmap artwork into recognized font slots.

    Validate all supplied rows and destinations before writing. Slot numbers come
    from the current G registry and encoder map, not from the ordering of glyphs.
    Only # sets an ink bit; every other row character is background.

    Args:
        rom (bytearray): Candidate ROM whose font table is updated in place.
        base (int): Nonnegative integer file offset of the destination font table;
            booleans and noninteger values are rejected.
        glyphs (Mapping[str, Sequence[str]]): Character/ligature keys mapped to
            eleven single-line bitmap row strings, at most 16 columns wide; # is ink.
            Rows must not contain newline or carriage-return characters, and the
            row collection must not be a single string.

    Raises:
        TypeError: rom is not a mutable bytearray.
        ValueError: An artwork key, bitmap shape or destination range is invalid.
        KeyError: A key in the registered G artwork has no encoder mapping.
    """
    require_byte_buffer(rom, (bytearray,))
    # Validate the offset before arithmetic: bool would otherwise act as 0 or 1.
    if type(base) is not int or base < 0:
        raise ValueError("Font table base must be a nonnegative integer; booleans are invalid")

    slots = find_glyph_slots()
    # Validate every bitmap and destination before the first write. An extra row
    # would overwrite the next glyph rather than being clipped by the renderer.
    for ch, rows in glyphs.items():
        if ch not in slots:
            raise ValueError(f"Unknown font artwork key: {ch!r}")
        if (
            isinstance(rows, str)
            or len(rows) != 11
            or any(
                not isinstance(row, str) or len(row) > 16 or "\n" in row or "\r" in row
                for row in cast(Sequence[object], rows)
            )
        ):
            raise ValueError(
                "Glyph artwork requires eleven single-line strings of at most sixteen pixels"
            )
        start = base + slots[ch] * 22
        if not 0 <= start <= len(rom) - 22:
            raise ValueError("Font glyph destination lies outside the ROM")

    # Use the encoder-assigned font slots and leave every unspecified glyph
    # untouched, including artwork still needed by Japanese or fixed-font text.
    for ch, idx in slots.items():
        if ch not in glyphs:
            continue

        rows = glyphs[ch]
        for r, line in enumerate(rows):
            w = 0

            for x, c in enumerate(line):
                if c == "#":
                    w |= 1 << (15 - x)

            # Each glyph has eleven little-endian 16-bit rows, with leftmost ink in bit 15.
            o = base + idx * 22 + r * 2
            rom[o] = w & 0xFF
            rom[o + 1] = w >> 8
