"""
This module writes rows of grayscale or RGB pixel bytes to a PNG file at a supplied path.

write_png() takes equal-length rows and a channel count: one byte per pixel
for grayscale or three bytes per pixel for RGB. It derives width and height,
checks row lengths, constructs the PNG header and chunks, compresses the pixel
bytes, and writes the file. The destination is overwritten if it exists; its
parent directory must already exist. All encoding uses the standard library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    import os
import struct
import zlib
from pathlib import Path


def write_png(path: str | os.PathLike[str], rows: Sequence[bytes], channels: Literal[1, 3]) -> None:
    """
    Write grayscale or RGB scanlines to a PNG file.

    The destination is overwritten after the complete PNG is encoded. Invalid
    row dimensions are rejected before opening it; an I/O failure while writing
    can still leave a partial file. This writer does not create parent directories.

    Args:
        path (str | os.PathLike[str]): Destination PNG path; its parent
            directory must exist.
        rows (Sequence[bytes]): Nonempty sequence of equal-length packed
            scanlines without PNG filter bytes. Each row must contain a
            positive whole number of pixels.
        channels (Literal[1, 3]): Bytes per pixel: one for grayscale or three
            for RGB. Booleans and other noninteger values are rejected.

    Raises:
        ValueError: The channel count, row lengths or image dimensions are invalid.
        OSError: The destination cannot be opened or written.
    """
    # Validate before opening the destination; a malformed image must not
    # replace an existing screenshot with an invalid PNG stream.
    if type(channels) is not int or channels not in (1, 3):
        raise ValueError("PNG channels must be 1 (grayscale) or 3 (RGB)")
    if not rows or not rows[0] or len(rows[0]) % channels:
        raise ValueError("PNG rows must contain a positive whole number of pixels")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("PNG scanlines must have equal byte lengths")

    def encode_png_chunk(kind: bytes, data: bytes) -> bytes:
        """
        Serialize a PNG chunk with its length and CRC.

        Args:
            kind (bytes): Four-byte PNG chunk type code.
            data (bytes): Chunk payload; both type and payload are included in the
                CRC.

        Returns:
            bytes: Big-endian length, type, payload and CRC concatenated in PNG
                format.
        """
        # PNG lengths and CRCs are big-endian; the CRC covers chunk type and payload.
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    width, height = len(rows[0]) // channels, len(rows)

    # PNG color types: 0 is grayscale, 2 is RGB; each sample is eight bits.
    color = {1: 0, 3: 2}[channels]

    # Each scanline starts with filter byte 00, meaning its bytes are unfiltered.
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + encode_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0))
        + encode_png_chunk(b"IDAT", zlib.compress(b"".join(b"\0" + r for r in rows), 9))
        + encode_png_chunk(b"IEND", b"")
    )
