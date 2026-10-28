"""
This module renders decoded SNES tiles as grayscale PNG preview sheets.

write_tile_sheet() reads 2bpp or 4bpp tile bytes, arranges them in a grid,
and enlarges each pixel to a 2x2 square. Palette indices determine grayscale
brightness; the image does not use the game's color palettes. The caller
supplies decoded tile bytes and a destination path.
"""

from __future__ import annotations

import os
from typing import Literal, cast

from core.graphics import convert_tiles
from core.graphics import write_png as png


def write_tile_sheet(
    data: bytes | bytearray, bpp: Literal[2, 4], path: str | os.PathLike[str], cols: int = 16
) -> None:
    """
    Write 8x8 tiles as a grayscale PNG sheet, drawn at twice their size.

    Empty data produces no file. Unused cells in the final row are black.
    Palette indices are mapped to evenly spaced grayscale values, and each
    source pixel is repeated twice horizontally and vertically.

    Args:
        data (bytes | bytearray): Planar tile bytes to render; not
            modified.
        bpp (Literal[2, 4]): Bits per pixel in the source planar tiles.
        path (str | os.PathLike[str]): Destination PNG path; an existing file is
            overwritten. The parent directory must already exist.
        cols (int, optional): Positive number of tile columns in the exported
            sheet.

    Raises:
        TypeError: data is not bytes or bytearray.
        ValueError: bpp or cols is invalid, or data ends with an incomplete tile.
            Both options require integers; booleans are rejected.
        OSError: The destination file cannot be written.
    """
    if not isinstance(cast(object, data), (bytes, bytearray)):
        raise TypeError("Expected bytes or bytearray data")
    if type(bpp) is not int or type(cols) is not int or bpp not in (2, 4) or cols <= 0:
        raise ValueError("Tile sheets require 2/4bpp and a positive column count")
    if len(data) % (8 * bpp):
        raise ValueError("Tile sheet data must contain complete tiles")
    # An 8x8 tile uses eight bytes per bitplane. Decode each tile separately
    # then place it in the preview grid without changing its native pixel order.
    tb = 8 * bpp
    n = len(data) // tb
    if not n:
        return

    rows = (n + cols - 1) // cols
    pixels = [bytearray(cols * 8) for _ in range(rows * 8)]
    for t in range(n):
        tile = convert_tiles.get_pixels(data, t, bpp)
        for y, row in enumerate(tile):
            for x, color in enumerate(row):
                pixels[(t // cols) * 8 + y][(t % cols) * 8 + x] = color * (255 // ((1 << bpp) - 1))

    # Enlarge pixels by two in both dimensions without filtering palette edges.
    doubled = [bytes(v for v in row for _ in (0, 1)) for row in pixels for _ in (0, 1)]
    png.write_png(path, doubled, 1)
