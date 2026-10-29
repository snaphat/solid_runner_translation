"""
This module repacks resources into their existing ROM allocation and installs them.

pack_resource_group() compresses a replacement together with enough contiguous
neighbors to fit before an untouched resource. write_resource_group() checks
allocation bounds, source-block contiguity, exclusive use of the allocation,
original destination bytes, and decoded payloads before updating pointers and
compressed data. Allocations shared with entries outside the selected group
are rejected. The caller supplies the source ROM and, for installation, a
mutable destination; neither function reads files.
"""

from __future__ import annotations

from typing import Sequence, cast

from core.graphics import compress_graphics, read_resources
from core.rom.convert_addresses import convert_file_offset_to_snes


def _require_exclusive_allocation(
    source: bytes | bytearray, table: Sequence[int], first: int, stop: int
) -> None:
    """
    Reject resource allocations referenced by entries outside a selected group.

    Args:
        source (bytes | bytearray): Original ROM used to locate stream ends.
        table (Sequence[int]): Validated table of resource file offsets.
        first (int): First selected table index, inclusive.
        stop (int): First unselected index after the group; its pointer bounds
            the allocation. Indices must satisfy 0 <= first < stop < len(table).

    Raises:
        AssertionError: An outside stream overlaps the allocation, or an
            earlier stream cannot be decoded to establish that it ends before it.
    """
    start, limit = table[first], table[stop]
    for index, offset in enumerate(table):
        if first <= index < stop or offset >= limit:
            continue

        # A second entry can point to a selected stream or into its interior.
        if offset >= start:
            raise AssertionError("Resource allocation is shared with an unselected entry")

        # An earlier stream can overlap even when its pointer lies outside the
        # allocation. Its terminator must precede or meet the allocation start.
        decoded, end = compress_graphics.decompress(offset, rom=source)
        if decoded is None:
            raise AssertionError("Cannot establish the end of an unselected resource")
        if end > start:
            raise AssertionError("Resource allocation overlaps an unselected resource")


def pack_resource_group(
    source: bytes | bytearray, bank: int, first: int, replacement: bytes
) -> list[bytes]:
    """
    Recompress a replacement and adjacent resources into their combined allocation.

    The final table entry can bound a group but cannot itself be replaced: a
    following untouched resource supplies the allocation end. Neighbors are
    included only until the first fit. Neither source nor replacement is changed;
    pass the returned blocks to write_resource_group() for installation.

    Args:
        source (bytes | bytearray): Original ROM supplying the resource table and neighbors.
        bank (int): Zero-based 32 KiB file-bank index.
        first (int): Table index of the resource being replaced.
        replacement (bytes): Uncompressed replacement for the first resource.

    Returns:
        list[bytes]: Packed blocks in table order, stopping at the first fit.
            The following untouched resource remains at its original offset.

    Raises:
        TypeError: source is not bytes or bytearray, or replacement is not bytes.
        AssertionError: The table or resources are invalid, blocks are not
            contiguous, the allocation is shared with an unselected entry, an
            earlier stream has no valid end, or the replacement cannot fit before
            an untouched resource.
        ValueError: bank or first does not identify a valid resource allocation;
            noninteger indices, including booleans, are rejected.
    """
    table = read_resources.read_bank_resource_table(bank, rom=source)
    if table is None:
        raise AssertionError("Native resource table is invalid")
    if type(first) is not int or not 0 <= first < len(table) - 1:
        raise ValueError("Resource group requires a valid first index and following boundary")

    packed: list[bytes] = []

    # Recompression may borrow space saved by neighboring blocks. Require their
    # original compressed spans to touch so unrelated bytes cannot enter the allocation.
    for index in range(first, len(table) - 1):
        raw, end = compress_graphics.decompress(table[index], rom=source)
        if raw is None:
            raise AssertionError("Native resource could not be decompressed")
        if end != table[index + 1]:
            raise AssertionError("Native resource blocks must be contiguous")

        packed.append(compress_graphics.compress_optimal(replacement if index == first else raw))

        # Stop at the first fit; the next table entry remains the untouched boundary.
        if sum(map(len, packed)) <= table[index + 1] - table[first]:
            _require_exclusive_allocation(source, table, first, index + 1)
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

    All allocation, original-content, and decoded-payload checks run before the
    first write. Supply the same original ROM and replacement used for packing.
    The selected pointers and compressed allocation in rom must still match
    source; unrelated changes elsewhere in rom are permitted.

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
            bytearray, a packed block is not bytes or bytearray, or clear_tail
            is not a boolean.
        AssertionError: The table, allocation or decoded contents are inconsistent,
            the original compressed blocks are not contiguous, the allocation
            is shared with an unselected entry, or an earlier stream has no valid end.
        IndexError: A compressed block is malformed.
        ValueError: The resource indices are invalid, packed is empty, or the
            destination ROM does not contain the complete allocation, or a
            packed block contains bytes after its terminator.
    """
    if not isinstance(cast(object, rom), bytearray):
        raise TypeError("Expected bytearray data")
    if type(clear_tail) is not bool:
        raise TypeError("clear_tail must be a boolean")
    table = read_resources.read_bank_resource_table(bank, rom=source)
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

    _require_exclusive_allocation(source, table, first, first + len(packed))

    pointers = slice(bank * 0x8000 + first * 2, bank * 0x8000 + (first + len(packed)) * 2)
    if rom[pointers] != source[pointers] or rom[position:limit] != source[position:limit]:
        raise AssertionError("Resource destination differs from its original allocation")

    # Check every payload before changing either the table or the destination data.
    for index, block in enumerate(packed, first):
        # Only contiguous compressed streams may share an allocation. A gap
        # can contain unrelated bytes that repacking or tail clearing would erase.
        original, end = compress_graphics.decompress(table[index], rom=source)
        if original is None:
            raise AssertionError("Native resource could not be decompressed")
        if end != table[index + 1]:
            raise AssertionError("Native resource blocks must be contiguous")
        expected = replacement if index == first else original
        if compress_graphics.decompress_bytes(block, require_complete=True) != expected:
            raise AssertionError("Packed resource differs from its expected contents")

    # Resource order is preserved, but compressed lengths move each start.
    # Update the 16-bit bank-local pointer alongside its corresponding payload.
    for index, block in enumerate(packed, first):
        pointer = bank * 0x8000 + index * 2
        address = convert_file_offset_to_snes(position) & 0xFFFF
        rom[pointer : pointer + 2] = address.to_bytes(2, "little")
        rom[position : position + len(block)] = block
        position += len(block)

    # Zero padding is selected by the caller through clear_tail. Otherwise
    # leave trailing allocation bytes unchanged after installing the payload.
    if clear_tail:
        rom[position:limit] = bytes(limit - position)
    return position
