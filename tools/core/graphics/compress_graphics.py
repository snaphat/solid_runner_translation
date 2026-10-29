"""
This module encodes and decodes the game's compressed resource streams.

The decoder takes ROM bytes and a resource offset and returns decompressed
bytes plus the position after the compressed stream. The two encoders produce
game-format compressed bytes. One immediately emits the longest available
backreference at each position; the other finds a minimum-size encoding.
ROM pointer-table lookup is provided by core.graphics.read_resources;
core.graphics.repack_resources installs recompressed resource groups.

Grayscale PNG previews are provided by core.graphics.write_tile_sheet.
The command-line interface is in tools/graphics/inspect_resources.py.

In the compressed format, 00 ends a stream, 01-7F copies that many literal
bytes, and 80-FF copies (control & 7F) + 3 bytes from previously decoded output.
The following byte specifies the backward distance minus one.
"""

from __future__ import annotations

import collections
import functools

from core.files.validate_binary_buffers import require_byte_buffer


def decompress(
    off: int, limit: int = 0x10000, *, rom: bytes | bytearray
) -> tuple[bytes | None, int]:
    """
    Decode one compressed resource while scanning a supplied ROM.

    Resource-table and corruption checks use this bounded decoder to distinguish a
    terminated resource from an invalid backreference or an overlong scan. The next
    source offset allows adjacent compressed blocks to be checked for contiguity.

    Args:
        off (int): Zero-based byte offset in the unheadered ROM.
        limit (int, optional): Decoded-byte threshold checked before each
            command. A command may exceed it; the next iteration returns a
            failure. Zero allows only an immediately empty stream.
        rom (bytes | bytearray): ROM containing the compressed resource
            at off.

    Returns:
        tuple[bytes | None, int]: Decoded bytes and next source offset; decoded data is None on a
            bounded-scan, truncated-command or invalid-backreference failure. On
            success the offset is just after the 00 terminator. On failure it
            identifies the next unprocessed command, which may lie beyond the buffer.

    Raises:
        TypeError: rom is not bytes or bytearray.
        ValueError: off or limit is not a nonnegative integer; booleans are rejected.
    """
    require_byte_buffer(rom, (bytes, bytearray))
    if type(off) is not int or type(limit) is not int or off < 0 or limit < 0:
        raise ValueError("Resource offset and decoded-byte limit must be nonnegative integers")
    out = bytearray()
    i = off
    while True:
        if i >= len(rom) or len(out) > limit:
            return None, i

        c = rom[i]
        if c == 0:
            return bytes(out), i + 1

        # 01-7F gives the count of literal bytes following the control byte.
        if c < 0x80:
            if i + 1 + c > len(rom):
                return None, i
            out += rom[i + 1 : i + 1 + c]
            i += 1 + c
        else:
            # The low seven bits encode length minus three. The following byte
            # encodes distance minus one, allowing distances from 1 through 256.
            n = (c & 0x7F) + 3
            if i + 1 >= len(rom):
                return None, i
            d = rom[i + 1] + 1
            if d > len(out):
                return None, i

            # Copy one byte at a time: a backreference may overlap its own output.
            for _ in range(n):
                out.append(out[-d])

            i += 2


def compress(data: bytes | bytearray) -> bytes:
    """
    Compress a resource by immediately taking the longest available backreference.

    Backreferences reach 256 bytes behind the cursor and copy 3–130 bytes.
    Equal-length matches use the nearest distance. Bytes without a three-byte
    match are collected into literal runs of at most 127 bytes. This local choice
    does not guarantee a minimum-size stream; use compress_optimal() when fitting
    a replacement into a fixed ROM allocation.

    Args:
        data (bytes | bytearray): Uncompressed resource bytes; read
            without modification.

    Returns:
        bytes: Compressed stream including its terminating zero byte.

    Raises:
        TypeError: data is not bytes or bytearray.
    """
    require_byte_buffer(data, (bytes, bytearray))
    out = bytearray()
    lit = bytearray()

    def flush() -> None:
        """
        Emit pending literal bytes in runs of at most 127 bytes.

        The compressed output is extended and the shared literal buffer is emptied
        before a backreference or the final terminator is emitted.
        """
        while lit:
            out.append(min(len(lit), 0x7F))
            out.extend(lit[:0x7F])
            del lit[:0x7F]

    i = 0
    while i < len(data):
        best, bd = 0, 0

        # Try the distances representable by the one-byte distance-minus-one field.
        for d in range(1, min(256, i) + 1):
            n = 0
            while n < 130 and i + n < len(data) and data[i + n] == data[i + n - d]:
                n += 1

            if n > best:
                best, bd = n, d

        # A match needs at least three bytes; shorter matches stay in the literal run.
        if best >= 3:
            flush()
            out += bytes([0x80 | (best - 3), bd - 1])
            i += best
        else:
            lit.append(data[i])
            i += 1

    # Finish the last literal run before the zero end-of-resource marker.
    flush()
    out.append(0)
    return bytes(out)


def compress_optimal(data: bytes) -> bytes:
    """
    Produce a smallest resource encoding with dynamic programming.

    Computes minimum suffix costs from right to left over literal runs and
    backreferences. Equal costs prefer a literal run over a backreference.
    Among backreferences, ties prefer the shortest copy, then the nearest match.
    Encoded results are cached by complete input bytes for the life of the process,
    with no size limit. Repeated inputs reuse the encoded stream.

    Args:
        data (bytes): Uncompressed resource bytes; immutable input is required
            by the optimal encoder cache.

    Returns:
        bytes: Compressed stream including its terminating zero byte.

    Raises:
        TypeError: data is not immutable bytes.
    """
    # Validate before looking up the cache: a hashable memoryview compares equal
    # to bytes and could otherwise hit an existing entry without being checked.
    # Conversion then gives the cache an immutable, ordinary bytes key.
    require_byte_buffer(data, (bytes,))
    return _compress_optimal_cached(bytes(data))


@functools.lru_cache(maxsize=None)
def _compress_optimal_cached(data: bytes) -> bytes:
    """
    Cache the dynamic-programming encoder for validated immutable resources.

    Args:
        data (bytes): Ordinary bytes already validated by compress_optimal().

    Returns:
        bytes: Minimum-size compressed stream including its terminating zero byte.
    """
    n = len(data)

    # cost[i] is the smallest encoded byte count for data[i:], including
    # the resource terminator. choice[i] records (length, backreference
    # distance); distance zero selects a literal run.
    cost = [0] * (n + 1)
    cost[n] = 1  # terminator
    choice: list[tuple[int, int] | None] = [None] * n
    matches: dict[int, int] = {}  # distance -> current match length (nonzero only)
    window: collections.deque[int] = collections.deque()  # j in (i, i+127], increasing j + cost[j]

    # Solve each suffix after its possible destinations, so every candidate
    # includes the cheapest encoding of the bytes that follow it.
    for i in range(n - 1, -1, -1):
        value = data[i]

        # Extend matching distances from the next suffix, capped at the
        # 130-byte command limit. Only the preceding 256 bytes are addressable.
        matches = {
            d: min(matches.get(d, 0) + 1, 130)
            for d in range(1, min(i, 256) + 1)
            if data[i - d] == value
        }
        j = i + 1  # literal run of length j - i costs 1 + (j - i) + cost[j]

        # Keep literal-run endpoints ordered by endpoint + suffix cost. The front
        # is the cheapest run still within the 127-byte literal limit.
        while window and window[-1] + cost[window[-1]] >= j + cost[j]:
            window.pop()

        window.append(j)

        if window[0] > i + 127:
            window.popleft()

        run = window[0] - i
        best, pick = 1 + run + cost[i + run], (run, 0)

        # A backreference costs two bytes regardless of its length. Compare
        # that cost plus the remaining suffix against the best literal run.
        longest = max(matches.values(), default=0)
        if longest >= 3:
            size = min(range(i + 3, i + longest + 1), key=cost.__getitem__) - i
            c = 2 + cost[i + size]
            if c < best:
                best, pick = c, (size, min(d for d, length in matches.items() if length >= size))

        cost[i], choice[i] = best, pick

    out = bytearray()
    i = 0

    # Replay the chosen commands; distance zero denotes a literal run.
    while i < n:
        pick = choice[i]
        if pick is None:
            raise AssertionError("Compression choice has not been initialized")

        size, d = pick
        if d:
            out += bytes([0x80 | (size - 3), d - 1])
        else:
            out.append(size)
            out += data[i : i + size]

        i += size

    out.append(0)
    return bytes(out)


def decompress_bytes(src: bytes | bytearray, *, require_complete: bool = False) -> bytes:
    """
    Decode the first compressed resource in a byte buffer.

    Args:
        src (bytes | bytearray): Compressed resource including its zero-byte terminator;
            following bytes are ignored unless require_complete is true.
        require_complete (bool, optional): Require the terminator to be the last
            byte. Use for isolated packed blocks, rather than slices of a ROM.

    Returns:
        bytes: Decoded resource bytes.

    Raises:
        TypeError: src is not bytes or bytearray, or require_complete is not a boolean.
        IndexError: A compressed operand or backreference is truncated or
            invalid.
        ValueError: require_complete is true and bytes follow the terminator.
    """
    # Even a stream made entirely of backreferences cannot expand beyond this
    # bound. The bounded scanner remains the single compression-format decoder.
    if type(require_complete) is not bool:
        raise TypeError("require_complete must be a boolean")
    require_byte_buffer(src, (bytes, bytearray))
    decoded, end = decompress(0, limit=len(src) * 130, rom=src)
    if decoded is None:
        raise IndexError("Compressed stream is unterminated or has an invalid backreference")
    if require_complete and end != len(src):
        raise ValueError("Compressed block contains bytes after its terminator")
    return decoded
