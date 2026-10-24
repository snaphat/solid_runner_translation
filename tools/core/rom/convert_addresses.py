"""
This module converts between SNES addresses used in game pointers and byte offsets used to index a ROM file.

Game pointers use SNES bank and address values; Python code indexes the ROM
file by byte offset. These helpers translate between the two using the LoROM
mapping, including pointers stored as separate low/high/bank bytes and 16-bit
pointers whose bank is supplied separately. They return integers and do not
read or change any file.

The mapping stores 32 KiB per bank and covers file offsets $000000-$3FFFFF.
For example, SNES address $81:8000 maps to file offset $008000. File offsets
exclude any 512-byte copier header. Integer ranges are checked, but converting
a bus address does not establish that it selects ROM rather than RAM or I/O.
Address mirror bits are discarded, so the reverse conversion returns a
canonical address rather than necessarily the original address.
"""


def convert_snes_to_file(pointer: int) -> int:
    """
    Remove the CPU bank mirror and upper-half address bits.

    Args:
        pointer (int): Packed SNES bus address, from 0x000000 through 0xFFFFFF.

    Returns:
        int: Byte offset in the unheadered ROM.

    Raises:
        ValueError: pointer is outside the unsigned 24-bit range.
    """
    if not 0 <= pointer <= 0xFFFFFF:
        raise ValueError(f"SNES address must be in 0x000000-0xFFFFFF, got {pointer:#x}")

    # CPU banks $00-$7F and $80-$FF share the same file-bank numbering.
    # Each bank contributes 32 KiB; bit 15 of the CPU address is discarded.
    return ((pointer >> 16) & 0x7F) * 0x8000 + (pointer & 0x7FFF)


def convert_pointer_bytes_to_file(lo: int, hi: int, bank: int) -> int:
    """
    Decode a LoROM pointer stored as separate instruction operands.

    Args:
        lo (int): Low address byte, 0-255.
        hi (int): High address byte, 0-255.
        bank (int): SNES bus-bank byte, 0-255.

    Returns:
        int: Byte offset in the unheadered ROM.

    Raises:
        ValueError: Any operand is outside the unsigned byte range.
    """
    # Check each byte before packing so an oversized operand cannot spill into
    # the next address field and appear to be a valid packed pointer.
    for name, value in (("lo", lo), ("hi", hi), ("bank", bank)):
        if not 0 <= value <= 0xFF:
            raise ValueError(f"{name} must be in 0x00-0xFF, got {value:#x}")

    return convert_snes_to_file(lo | hi << 8 | bank << 16)


def convert_file_offset_to_snes(off: int) -> int:
    """
    Map a ROM offset into the upper half of a high CPU bank.

    Args:
        off (int): Byte offset in an unheadered LoROM image, from 0x000000
            through 0x3FFFFF, regardless of the actual image's size.

    Returns:
        int: Packed SNES address in banks $80-$FF at $8000-$FFFF. This
            canonical address may differ from a mirrored address originally
            converted to off; the original bank and address mirror bits are lost.

    Raises:
        ValueError: off is outside the standard 4 MiB LoROM range.
    """
    if not 0 <= off <= 0x3FFFFF:
        raise ValueError(f"LoROM file offset must be in 0x000000-0x3FFFFF, got {off:#x}")

    # Select a high CPU bank and its upper-half address window. The file's
    # 32 KiB bank index supplies the CPU bank; its low 15 bits supply the address.
    return 0x800000 | ((off // 0x8000) << 16) | 0x8000 | (off & 0x7FFF)


def resolve_bank_pointer(pointer: int, source: int) -> int:
    """
    Resolve a 16-bit pointer within its source resource's ROM bank.

    Args:
        pointer (int): Native CPU address from 0x0000 through 0xFFFF;
            its bank comes from source and its upper-half bit is discarded.
        source (int): File offset from 0x000000 through 0x3FFFFF identifying
            the owning 32 KiB ROM bank; its position within that bank is ignored.

    Returns:
        int: Byte offset in the same file bank as source.

    Raises:
        ValueError: pointer is outside the unsigned 16-bit range, or source is
            outside the standard 4 MiB LoROM range.
    """
    if not 0 <= pointer <= 0xFFFF:
        raise ValueError(f"Bank-local pointer must be in 0x0000-0xFFFF, got {pointer:#x}")
    if not 0 <= source <= 0x3FFFFF:
        raise ValueError(f"Source file offset must be in 0x000000-0x3FFFFF, got {source:#x}")

    # Retain the source's file-bank bits and replace its within-bank offset.
    return (source & ~0x7FFF) | (pointer & 0x7FFF)
