"""
This module locates and decodes resources through bank-local ROM pointer tables.

read_bank_resource_table() checks a 32 KiB LoROM bank for a plausible table
and returns resource file offsets. read_resource_block() decodes one indexed
resource. iter_resource_blocks() scans complete banks and yields candidates
only when every listed resource decodes to at least 32 bytes. These checks
identify candidate graphics data; they do not establish its use by the game.
All readers use caller-supplied ROM bytes and leave them unchanged.
"""

from __future__ import annotations

from typing import Iterator, cast

from core.graphics import compress_graphics
from core.rom.convert_addresses import resolve_bank_pointer


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
    if not isinstance(cast(object, rom), (bytes, bytearray)):
        raise TypeError("Expected bytes or bytearray data")
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
            data, _ = compress_graphics.decompress(off, rom=rom)
            if data is None or len(data) < 32:
                break

            res.append((off, data))
        else:
            # Yield a bank only if every candidate resource decoded successfully;
            # a break above rejects the entire candidate table, not just one block.
            if res:
                yield bank, res


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
        TypeError: rom is not bytes or bytearray.
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
    raw, _ = compress_graphics.decompress(table[index], rom=rom)
    if raw is None:
        raise AssertionError("Native resource could not be decompressed")
    return raw
