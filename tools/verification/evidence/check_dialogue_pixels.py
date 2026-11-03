"""
This module reconstructs dialogue pixels from ROM font data and traced glyph positions and counts pixels missing from a captured frame.

The pixel reconstruction uses font bytes and the glyph log rather than calling the game's VWF
drawing code. The comparison counts expected pixels that fall outside the frame or fail the
supplied brightness threshold. It depends on the trace identifying the glyphs and drawing
coordinates correctly.
"""

from __future__ import annotations

from typing import Iterable, Sequence, cast

from core.validation.validate_integers import is_integer
from verification.evidence.instrument_rom import GlyphRecord
from verification.evidence.reconstruct_text_ink import (
    MAIN_FONT as MAIN_FONT,
)
from verification.evidence.reconstruct_text_ink import (
    VWF_FONT as VWF_FONT,
)
from verification.evidence.reconstruct_text_ink import (
    reconstruct_glyph_ink,
)

WIDTH = 240  # Dialogue text-plane width in pixels (thirty 8-pixel tiles).


def reconstruct_dialogue_ink(
    rom: bytes | bytearray, glyphs: Iterable[GlyphRecord], height: int = 48
) -> tuple[set[tuple[int, int]], int]:
    """
    This function reconstructs dialogue pixels within the text buffer and counts glyph pixels drawn outside it.

    Glyphs are processed in draw order, clearing each glyph's rectangle before
    adding its ink. This models overwrite mode. The trace does not include the
    renderer flag at $18B9, so the caller must establish that overwrite mode applies.

    The outside count includes every draw beyond the buffer's 240-pixel width or
    supplied height, including repeated draws at the same position. Later glyphs
    do not reduce that count.

    Args:
        rom (bytes | bytearray): Font data used to redraw the traced
            dialogue glyphs.
        glyphs (Iterable[GlyphRecord]): Instrumented glyph records containing font offsets
            and text-buffer drawing coordinates.
        height (int, optional): Nonnegative integer text-plane height in pixels;
            integer subclasses are accepted and booleans are rejected.
            Its width is fixed at 240; zero clips all ink.
            Defaults to 48.

    Returns:
        tuple[set[tuple[int, int]], int]: Expected (x, y) ink coordinates within
            the 240-by-height text buffer, and the count of glyph pixels drawn
            outside that buffer.

    Raises:
        ValueError: If height is a boolean, noninteger or negative, a glyph cursor
            field or font offset is a boolean or noninteger, or a glyph
            selects a negative or incomplete ROM font record.
    """
    if (not is_integer(height)) or height < 0:
        raise ValueError("Text-plane height must be a nonnegative integer")

    ink: set[tuple[int, int]] = set()
    outside = 0

    for g in glyphs:
        pixels, (x1, y1, x2, y2) = reconstruct_glyph_ink(rom, g)
        # Apply the overwrite-mode assumption; the trace does not record erasure.
        ink = {(x, y) for x, y in ink if not (x1 <= x < x2 and y1 <= y < y2)}
        for x, y in pixels:
            if 0 <= x < WIDTH and 0 <= y < height:
                ink.add((x, y))
            else:
                outside += 1

    return ink, outside


def count_missing_dialogue_pixels(
    rows: Sequence[bytes],
    ink: set[tuple[int, int]],
    dx: int,
    dy: int,
    threshold: tuple[int, int, int] = (220, 220, 180),
) -> tuple[int, int]:
    """
    Count expected ink pixels missing from or outside the frame.

    Origins, ink coordinates and thresholds must be integers. Integer subclasses are accepted;
    booleans are rejected.

    Args:
        rows (Sequence[bytes]): Frame scanlines of packed 8-bit RGB pixels.
            Every sampled row must contain 256 complete RGB pixels; this helper
            uses that fixed frame width rather than deriving it from row length.
        ink (set[tuple[int, int]]): Expected text-plane ink-pixel coordinates.
        dx (int): Text-plane horizontal origin on the frame, in pixels.
        dy (int): Text-plane vertical origin on the frame, in pixels.
        threshold (tuple[int, int, int], optional): Per-channel RGB brightness
            cutoffs for expected text ink: exactly three integers in 0-255.
            Booleans are rejected. A pixel counts as bright only when
            every channel is strictly greater than its cutoff.
            Defaults to (220, 220, 180).

    Returns:
        tuple[int, int]: Missing bright ink-pixel count and off-frame ink-pixel
            count.

    Raises:
        ValueError: If an origin or ink coordinate is a boolean or noninteger, or
            threshold is not a three-element tuple of integer RGB cutoffs in 0-255
            excluding booleans, or a sampled pixel lacks three RGB bytes.
    """
    if not is_integer(dx) or not is_integer(dy):
        raise ValueError("dx, dy must be integers, not booleans")
    if any(len(point) != 2 or any(not is_integer(value) for value in point) for point in ink):
        raise ValueError("Ink coordinates must be pairs of integers, not booleans")
    # Reject malformed RGB tuples before sampling, including when ink is empty.
    if (
        not isinstance(cast(object, threshold), tuple)
        or len(threshold) != 3
        or any((not is_integer(channel)) or not 0 <= channel <= 255 for channel in threshold)
    ):
        raise ValueError("RGB brightness thresholds must be three integers in 0-255")
    # Count pixels outside the captured frame separately from in-frame pixels
    # whose RGB channels do not reach the text-color thresholds.
    missing = off = 0

    for x, y in ink:
        sx, sy = x + dx, y + dy
        if not (0 <= sx < 256 and 0 <= sy < len(rows)):
            off += 1
            continue

        r, g, b = rows[sy][sx * 3 : sx * 3 + 3]
        if not (r > threshold[0] and g > threshold[1] and b > threshold[2]):
            missing += 1

    return missing, off
