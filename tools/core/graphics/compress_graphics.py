"""
This module compresses, decompresses, and repacks game resources and exports tile-sheet previews.

The decoder takes ROM bytes and a resource offset and returns decompressed
bytes plus the position after the compressed stream. The two encoders produce
game-format compressed bytes. One immediately emits the longest available
backreference at each position; the other finds a minimum-size encoding.
Resource-group helpers recompress a changed resource together with adjacent
resources, then check and write the compressed bytes and pointers
into a caller-supplied ROM buffer.

The preview writer exports grayscale PNG tile sheets using the tile decoder
in core.graphics.convert_tiles. That module also provides the 4bpp tile writer.
The command-line interface is in tools/graphics/inspect_resources.py.

In the compressed format, 00 ends a stream, 01-7F copies that many literal
bytes, and 80-FF copies (control & 7F) + 3 bytes from previously decoded output.
The following byte specifies the backward distance minus one. Resource tables
store 16-bit pointers at the beginning of each 32 KiB bank.
"""

from __future__ import annotations

import collections
import functools
import os
from typing import Iterator, Literal, Sequence

from core.graphics import convert_tiles
from core.rom.convert_addresses import convert_file_offset_to_snes, resolve_bank_pointer


def _require_byte_buffer(data: object, accepted: tuple[type[bytes] | type[bytearray], ...]) -> None:
    """
    Reject values outside a reader or writer's supported byte-buffer types.

    Args:
        data (object): Resource, tile or destination buffer supplied by the caller.
        accepted (tuple[type[bytes] | type[bytearray], ...]): Allowed buffer classes;
            bytes is immutable and bytearray supports in-place writes.

    Raises:
        TypeError: data is not an instance of an accepted buffer class.
    """
    if not isinstance(data, accepted):
        raise TypeError("Expected " + " or ".join(kind.__name__ for kind in accepted) + " data")


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
            failure.
        rom (bytes | bytearray): ROM containing the compressed resource
            at off.

    Returns:
        tuple[bytes | None, int]: Decoded bytes and next source offset; decoded data is None on a
            bounded-scan, truncated-command or invalid-backreference failure.

    Raises:
        TypeError: rom is not bytes or bytearray.
        ValueError: off or limit is not a nonnegative integer; booleans are rejected.
    """
    _require_byte_buffer(rom, (bytes, bytearray))
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
    Compress a resource using greedy backreference selection.

    Backreferences reach 256 bytes behind the cursor and copy 3–130 bytes.

    Args:
        data (bytes | bytearray): Uncompressed resource bytes; read
            without modification.

    Returns:
        bytes: Compressed stream including its terminating zero byte.

    Raises:
        TypeError: data is not bytes or bytearray.
    """
    _require_byte_buffer(data, (bytes, bytearray))
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
    _require_byte_buffer(data, (bytes,))
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
    _require_byte_buffer(src, (bytes, bytearray))
    decoded, end = decompress(0, limit=len(src) * 130, rom=src)
    if decoded is None:
        raise IndexError("Compressed stream is unterminated or has an invalid backreference")
    if require_complete and end != len(src):
        raise ValueError("Compressed block contains bytes after its terminator")
    return decoded


def read_bank_resource_table(bank: int, *, rom: bytes | bytearray) -> list[int] | None:
    """
    Read a plausible resource pointer table from a 32 KiB ROM bank.

    Table words are at least $8000. The table ends where its earliest resource
    begins.

    Args:
        bank (int): Zero-based 32 KiB ROM bank index, not an SNES bus-bank byte.
        rom (bytes | bytearray): ROM containing the bank-local resource
            pointers.

    Returns:
        list[int] | None: Resource file offsets, or None if the bank does
            not have a plausible complete table.

    Raises:
        TypeError: rom is not bytes or bytearray.
        ValueError: bank is not an integer identifying a complete standard LoROM
            bank. Booleans are rejected.
    """
    _require_byte_buffer(rom, (bytes, bytearray))
    if type(bank) is not int or not 0 <= bank < 128 or (bank + 1) * 0x8000 > len(rom):
        raise ValueError("Resource bank must identify a complete 32 KiB LoROM bank")
    b0 = bank * 0x8000
    ptrs: list[int] = []
    first = 0x10000
    k = 0

    # Pointers are little-endian CPU addresses within this 32 KiB bank. The
    # first resource also marks the end of the pointer table.
    while b0 + k * 2 < b0 + 0x8000 and 0x8000 + k * 2 < first:
        p = rom[b0 + k * 2] | rom[b0 + k * 2 + 1] << 8
        if p < 0x8000:
            break

        ptrs.append(p)
        first = min(first, p)
        k += 1

    if not ptrs or first != 0x8000 + len(ptrs) * 2:
        return None

    return [resolve_bank_pointer(p, b0) for p in ptrs]


def iter_resource_blocks(rom: bytes | bytearray) -> Iterator[tuple[int, list[tuple[int, bytes]]]]:
    """
    Yield candidate graphics banks and their decoded resource blocks.

    Scan complete 32 KiB banks only. A bank is accepted only when its pointer
    table is plausible and every listed resource decodes to at least 32 bytes,
    the size of one 4bpp tile. Smaller resources exclude the entire bank.

    Args:
        rom (bytes | bytearray): ROM whose banks are scanned for resource
            tables.

    Yields:
        tuple[int, list[tuple[int, bytes]]]: Bank index and a list of
            (resource file offset, decompressed bytes) pairs in table order.
    """
    for bank in range(len(rom) // 0x8000):
        t = read_bank_resource_table(bank, rom=rom)
        if not t:
            continue

        res: list[tuple[int, bytes]] = []
        for off in t:
            data, _ = decompress(off, rom=rom)
            if data is None or len(data) < 32:
                break

            res.append((off, data))
        else:
            # Yield a bank only if every candidate resource decoded successfully;
            # a break above rejects the entire candidate table, not just one block.
            if res:
                yield bank, res


def write_tile_sheet(
    data: bytes | bytearray, bpp: Literal[2, 4], path: str | os.PathLike[str], cols: int = 16
) -> None:
    """
    Write 8x8 tiles as a grayscale PNG sheet, drawn at twice their size.

    Args:
        data (bytes | bytearray): Planar tile bytes to render; not
            modified.
        bpp (Literal[2, 4]): Bits per pixel in the source planar tiles.
        path (str | os.PathLike[str]): Destination PNG tile-sheet path.
        cols (int, optional): Positive number of tile columns in the exported
            sheet.

    Raises:
        TypeError: data is not bytes or bytearray.
        ValueError: bpp or cols is invalid, or data ends with an incomplete tile.
            Both options require integers; booleans are rejected.
    """
    from core.graphics import write_png as png

    _require_byte_buffer(data, (bytes, bytearray))
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


def read_resource_block(rom: bytes | bytearray, bank: int, index: int) -> bytes:
    """
    Read one decompressed resource through its native bank-local table.

    Args:
        rom (bytes | bytearray): Source ROM containing the resource and pointer table.
        bank (int): Zero-based 32 KiB file-bank index.
        index (int): Zero-based resource index within that bank's table.

    Returns:
        bytes: Decompressed resource contents.

    Raises:
        AssertionError: The bank table or compressed resource is invalid.
        IndexError: The resource index is not an integer or is out of bounds;
            booleans are rejected.
        ValueError: bank does not identify a complete standard LoROM bank.
    """
    # A resource index selects a bank-local table entry, not a direct ROM offset.
    table = read_bank_resource_table(bank, rom=rom)
    if table is None:
        raise AssertionError("Native resource table is invalid")
    if type(index) is not int or not 0 <= index < len(table):
        raise IndexError("Resource index is outside the native table")
    raw, _ = decompress(table[index], rom=rom)
    if raw is None:
        raise AssertionError("Native resource could not be decompressed")
    return raw


def pack_resource_group(
    source: bytes | bytearray, bank: int, first: int, replacement: bytes
) -> list[bytes]:
    """
    Recompress a replacement and adjacent resources into their combined allocation.

    Args:
        source (bytes | bytearray): Original ROM supplying the resource table and neighbors.
        bank (int): Zero-based 32 KiB file-bank index.
        first (int): Table index of the resource being replaced.
        replacement (bytes): Uncompressed replacement for the first resource.

    Returns:
        list[bytes]: Packed blocks in table order, stopping at the first fit.
            The following untouched resource remains at its original offset.

    Raises:
        AssertionError: The table or resources are invalid, blocks are not
            contiguous, or the replacement cannot fit before an untouched resource.
        ValueError: bank or first does not identify a valid resource allocation;
            noninteger indices, including booleans, are rejected.
    """
    table = read_bank_resource_table(bank, rom=source)
    if table is None:
        raise AssertionError("Native resource table is invalid")
    if type(first) is not int or not 0 <= first < len(table) - 1:
        raise ValueError("Resource group requires a valid first index and following boundary")

    packed: list[bytes] = []

    # Recompression may borrow space saved by neighboring blocks. Require their
    # original compressed spans to touch so unrelated bytes cannot enter the allocation.
    for index in range(first, len(table) - 1):
        raw, end = decompress(table[index], rom=source)
        if raw is None:
            raise AssertionError("Native resource could not be decompressed")
        if end != table[index + 1]:
            raise AssertionError("Native resource blocks must be contiguous")

        packed.append(compress_optimal(replacement if index == first else raw))

        # Stop at the first fit; the next table entry remains the untouched boundary.
        if sum(map(len, packed)) <= table[index + 1] - table[first]:
            return packed

    raise AssertionError("Replacement resource group does not fit its native bank")


def write_resource_group(
    rom: bytearray,
    source: bytes | bytearray,
    bank: int,
    first: int,
    packed: Sequence[bytes],
    replacement: bytes,
    clear_tail: bool = False,
) -> int:
    """
    Verify and install repacked resources with their bank-local pointers.

    Args:
        rom (bytearray): Destination ROM modified in place after content checks.
        source (bytes | bytearray): Original ROM defining the allocation and unchanged neighbors.
        bank (int): Zero-based 32 KiB file-bank index.
        first (int): Table index of the replaced resource.
        packed (Sequence[bytes]): Repacked blocks beginning at first.
            Each block must end exactly at its compression terminator.
        replacement (bytes): Expected uncompressed first block; subsequent blocks
            must decode identically to their original resources.
        clear_tail (bool, optional): Zero unused allocation bytes when true;
            otherwise retain their existing contents.

    Returns:
        int: File offset immediately after the installed compressed blocks.

    Raises:
        TypeError: rom is not a mutable bytearray, source is not bytes or
            bytearray, or clear_tail is not a boolean.
        AssertionError: The table, allocation or decoded contents are inconsistent.
        IndexError: A compressed block is malformed.
        ValueError: The resource indices are invalid, packed is empty, or the
            destination ROM does not contain the complete allocation, or a
            packed block contains bytes after its terminator.
    """
    _require_byte_buffer(rom, (bytearray,))
    if type(clear_tail) is not bool:
        raise TypeError("clear_tail must be a boolean")
    table = read_bank_resource_table(bank, rom=source)
    if table is None:
        raise AssertionError("Native resource table is invalid")
    if type(first) is not int or not packed or not 0 <= first < first + len(packed) < len(table):
        raise ValueError("Resource group must be nonempty and leave a following table boundary")
    # The first untouched resource defines the allocation limit. Check it before
    # changing pointers so an oversized group cannot overwrite the following block.
    limit = table[first + len(packed)]
    position = table[first]
    if not 0 <= position <= limit <= len(rom):
        raise ValueError("Destination ROM does not contain the complete resource allocation")
    if position + sum(map(len, packed)) > limit:
        raise AssertionError("Packed resources exceed their native allocation")

    pointers = slice(bank * 0x8000 + first * 2, bank * 0x8000 + (first + len(packed)) * 2)
    if rom[pointers] != source[pointers] or rom[position:limit] != source[position:limit]:
        raise AssertionError("Resource destination differs from its original allocation")

    # Check every payload before changing either the table or the destination data.
    for index, block in enumerate(packed, first):
        expected = replacement if index == first else decompress(table[index], rom=source)[0]
        if decompress_bytes(block, require_complete=True) != expected:
            raise AssertionError("Packed resource differs from its expected contents")

    # Resource order is preserved, but compressed lengths move each start.
    # Update the 16-bit bank-local pointer alongside its corresponding payload.
    for index, block in enumerate(packed, first):
        pointer = bank * 0x8000 + index * 2
        address = convert_file_offset_to_snes(position) & 0xFFFF
        rom[pointer : pointer + 2] = address.to_bytes(2, "little")
        rom[position : position + len(block)] = block
        position += len(block)

    # Zero padding is selected by the repair's clear_tail policy. Otherwise
    # leave trailing allocation bytes unchanged after installing the payload.
    if clear_tail:
        rom[position:limit] = bytes(limit - position)
    return position
