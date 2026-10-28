"""
This module converts SNES tile bytes into 8x8 palette-index arrays and back.

get_pixels() reads one 2bpp or 4bpp tile from a byte buffer. put_pixels()
validates an 8x8 array of palette indices and writes one 4bpp tile into a
mutable buffer. Tile indices start at zero; neither operation changes the
buffer length.
"""

from __future__ import annotations

from typing import Literal, Sequence, cast


def get_pixels(data: bytes | bytearray, t: int, bpp: Literal[2, 4] = 4) -> list[list[int]]:
    """
    Decode one native planar tile into an 8x8 palette-index matrix.

    Args:
        data (bytes | bytearray): Packed native planar tile bytes to read.
        t (int): Zero-based tile number in the supplied buffer.
        bpp (Literal[2, 4], optional): Bitplanes per pixel; four selects the
            game's usual resource tiles, two selects its lower-color previews.

    Returns:
        list[list[int]]: 8 by 8 row-major palette-index matrix.

    Raises:
        TypeError: data is not bytes or bytearray.
        ValueError: bpp is not integer 2 or 4, or t is not an integer identifying
            a complete tile. Booleans are rejected.
    """
    if not isinstance(cast(object, data), (bytes, bytearray)):
        raise TypeError("Expected bytes or bytearray data")
    if type(bpp) is not int or bpp not in (2, 4):
        raise ValueError("Tile depth must be 2 or 4 bits per pixel")
    tile_bytes = 8 * bpp
    if type(t) is not int or t < 0 or (t + 1) * tile_bytes > len(data):
        raise ValueError("Tile index must identify a complete tile in the buffer")
    tile = data[t * tile_bytes : (t + 1) * tile_bytes]

    # SNES tiles interleave pairs of bitplanes in consecutive 16-byte groups.
    planes = [
        [tile[16 * (plane // 2) + y * 2 + plane % 2] for plane in range(bpp)] for y in range(8)
    ]
    return [
        [
            sum(((value >> (7 - x)) & 1) << plane for plane, value in enumerate(row))
            for x in range(8)
        ]
        for row in planes
    ]


def put_pixels(data: bytearray, t: int, px: Sequence[Sequence[int]]) -> None:
    """
    Write an 8x8 palette-index matrix into 4bpp tile t of a mutable buffer.

    Args:
        data (bytearray): Mutable packed 4bpp SNES tile buffer, updated in
            place.
        t (int): Zero-based tile number in a packed 4bpp SNES tile buffer.
        px (Sequence[Sequence[int]]): 8 by 8 palette-index matrix; each pixel
            is an integer in the range 0-15; booleans are not palette indices.

    Raises:
        TypeError: data is not a mutable bytearray.
        ValueError: The tile index, matrix dimensions or palette indices are invalid.
            The tile index must be an integer; booleans are rejected.
    """
    # Validate the whole matrix before the first write; a malformed late row
    # must not leave the destination tile partly replaced.
    if not isinstance(cast(object, data), bytearray):
        raise TypeError("Expected bytearray data")
    if type(t) is not int or t < 0 or (t + 1) * 32 > len(data):
        raise ValueError("Tile index must identify a complete 4bpp tile")
    if len(px) != 8 or any(len(row) != 8 for row in px):
        raise ValueError("A tile must contain exactly eight rows of eight pixels")
    if any(type(pixel) is not int or not 0 <= pixel <= 15 for row in px for pixel in row):
        raise ValueError("4bpp palette indices must be integers in 0-15")
    # Split each palette index into four bitplanes, most-significant pixel first.
    # The first two planes occupy the first 16 bytes; planes 2-3 occupy the next 16.
    for y in range(8):
        planes = [0, 0, 0, 0]
        for x in range(8):
            for k in range(4):
                planes[k] |= ((px[y][x] >> k) & 1) << (7 - x)

        data[t * 32 + y * 2], data[t * 32 + y * 2 + 1] = planes[0], planes[1]
        data[t * 32 + 16 + y * 2], data[t * 32 + 17 + y * 2] = planes[2], planes[3]
