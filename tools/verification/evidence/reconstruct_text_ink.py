"""
This module calculates a traced glyph's expected ink pixels and overwrite span from ROM font data and drawing coordinates.

These calculations are independent of the game's renderer. Dialogue and screen
checks share the native font and cursor rules while applying their own clipping,
overwrite and tilemap-placement policies to the reconstructed pixels.
The trace does not record the renderer's overwrite flag ($18B9), so this module
cannot determine whether a glyph actually erased earlier pixels.
"""

from core.graphics.font_tables import MAIN_FONT as MAIN_FONT
from core.graphics.font_tables import VWF_FONT as VWF_FONT
from verification.evidence.instrument_rom import GlyphRecord


def reconstruct_glyph_ink(
    rom: bytes | bytearray, g: GlyphRecord
) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]]:
    """
    Place one fixed or proportional glyph in the native text-buffer coordinates.

    Args:
        rom (bytes | bytearray): ROM containing the native and proportional font tables.
        g (GlyphRecord): Trace containing the font offset, before/after cursor,
            proportional-mode bit and native row/strip coordinates.

    Returns:
        tuple[set[tuple[int, int]], tuple[int, int, int, int]]: Glyph ink coordinates
            and its half-open overwrite rectangle (left, top, right, bottom).
            This is the span used for erasure in overwrite mode, not evidence
            that erasure occurred. GlyphRecord omits the overwrite flag ($18B9);
            callers must decide whether to apply the rectangle. Coordinates
            remain unclipped so each verifier can enforce its own drawing bounds.

    Raises:
        ValueError: The trace selects a negative or incomplete ROM font record.
    """
    vwf = bool(g["after_col"] & 0x80)
    font = VWF_FONT if vwf and g["font_offset"] // 22 < 256 else MAIN_FONT
    if g["font_offset"] < 0 or font + g["font_offset"] + 22 > len(rom):
        raise ValueError("Glyph trace points outside the ROM font data")
    rows = [
        rom[font + g["font_offset"] + 2 * y] | rom[font + g["font_offset"] + 2 * y + 1] << 8
        for y in range(11)
    ]

    # Row bit 1 selects the second 120-pixel strip; the remaining bits locate
    # the 16-pixel text row. A proportional cursor can continue within a strip.
    if vwf:
        x0 = g["before"] if g["col"] & 0x80 else g["col"] * 12 + (120 if g["row"] & 2 else 0)
        y0 = (g["row"] & 0xFC) * 4 + 3
        bits = 0
        for row in rows:
            bits |= row
        left = 16 - bits.bit_length() if bits else 0
        columns, end = 16, g["after"]
    else:
        x0 = g["after_col"] * 12 + (120 if g["after_row"] & 2 else 0)
        y0 = (g["after_row"] & 0xFC) * 4 + 3
        left, columns, end = 0, 12, x0 + 12

    pixels = {
        (x0 + x - left, y0 + y)
        for y, row in enumerate(rows)
        for x in range(columns)
        if row & (1 << (15 - x))
    }
    return pixels, (x0, y0, end, y0 + 11)
