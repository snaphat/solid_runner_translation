"""
This module reserves ROM space for translated text without crossing bank boundaries.

Space allocates from ordered caller-supplied ranges. allocate_bank_tail()
finds a bank's trailing FF bytes, leaves a sixteen-byte gap after existing
data, and tracks reservations in a caller-supplied cursor cache. Reservations
return file offsets and update allocation state without writing ROM bytes.

The text-storage ranges select banks compatible with the game's proportional,
fixed-font, and short-jump readers. All offsets refer to an unheadered ROM.
"""

from __future__ import annotations

from typing import Iterable

# Allocation ranges below are half-open byte offsets in the unheadered ROM.
# Proportional-text storage uses CPU banks $BA-$BB and $E0-$FC. The latter
# range exists after expansion to the 4 MiB declared by the ROM header.
FREE_REGIONS = [(0x1D0000, 0x1E0000), (0x300000, 0x3E8000)]  # 0x3F0000+: VWF font
# A four-byte F8 20 jump carries only a 16-bit address, so its payload must
# live in CPU bank $FD. The five-byte F8 06 jump carries a complete SNES address.
SHORT_REGIONS = [(0x3E8000, 0x3F0000)]

# Mode-f records use fixed-font storage in CPU banks $CD and $D3-$D4,
# which the proportional-font renderer does not select.
FIXED_REGIONS = [(0x268000, 0x270000), (0x298000, 0x2A8000)]


def allocate_bank_tail(
    rom: bytes | bytearray, bank: int, n: int, free: dict[int, list[int]]
) -> int:
    """
    Reserve bytes from the FF-filled tail of a 32 KiB ROM bank.

    The first request scans the trailing FF run and leaves sixteen bytes before
    its first usable byte. Keep free tied to this ROM and allocation pass; later
    requests trust its cursors instead of rescanning. A failed first request may
    still add that bank to free, but does not advance its reservation cursor.
    A zero-byte request returns the cursor without consuming bytes.

    Args:
        rom (bytes | bytearray): ROM inspected for its FF-filled bank
            tail.
        bank (int): Zero-based 32 KiB ROM bank index, not an SNES bus-bank byte.
        n (int): Integer number of consecutive bytes to reserve in the bank tail;
            booleans are rejected.
        free (dict[int, list[int]]): Mutable bank-index to [next-free offset,
            exclusive bank end] cache.

    Returns:
        int: Reserved region start as an unheadered file offset.

    Raises:
        ValueError: The bank is not an integer identifying a complete LoROM bank,
            or its cached free-space cursors are not valid integer offsets.
        AssertionError: The size is not an integer from 0 through 32768, or no
            allowed region can satisfy the requested allocation.
    """
    if type(n) is not int or not 0 <= n <= 0x8000:
        raise AssertionError("Allocation size must be an integer and fit one 32KB bank")

    if type(bank) is not int or not 0 <= bank < 128 or (bank + 1) * 0x8000 > len(rom):
        raise ValueError("Allocation requires a complete ROM bank")

    # Find this bank's trailing FF run once. The shared cursor then tracks
    # allocations without rescanning bytes that the compiler has already written.
    if bank not in free:
        b0 = bank * 0x8000
        i = b0 + 0x8000
        while i > b0 and rom[i - 1] == 0xFF:
            i -= 1

        # Reserve sixteen FF bytes between the native data and the first allocation.
        free[bank] = [min(i + 16, b0 + 0x8000), b0 + 0x8000]

    lo, hi = free[bank]
    if (
        type(lo) is not int
        or type(hi) is not int
        or not bank * 0x8000 <= lo <= hi == (bank + 1) * 0x8000
    ):
        raise ValueError("Cached free-space cursor does not belong to its ROM bank")
    if not (lo + n <= hi):
        raise AssertionError(f"bank {bank:02X} full")

    free[bank][0] = lo + n
    return lo


class Space:
    """
    Allocate ROM regions without crossing 32 KiB bank boundaries.

    Regions are used in ascending order. A successful request consumes its bytes
    and any earlier spans it had to skip; an unsuccessful request leaves the
    allocator unchanged. The allocator neither examines existing ROM contents
    nor writes bytes. The caller must supply ranges safe to overwrite.

    Attributes:
        free (int): Next available unheadered file offset in the active region.
        end (int): Exclusive end offset of the active region.
        regions (list[tuple[int, int]]): Remaining half-open allocation
            regions, consumed as earlier regions fill.
    """

    def __init__(self, regions: Iterable[tuple[int, int]]) -> None:
        """
        Copy a nonempty iterable of (start, end) file-offset regions.

        Args:
            regions (Iterable[tuple[int, int]]): Nonempty ordered half-open (start,
                end) regions with 0 <= start < end <= 0x400000. Regions must be
                ascending and nonoverlapping; booleans are not bounds. The region
                iterable is consumed once into a private list. Subsequent allocations
                modify only that private list.

        Raises:
            ValueError: Allocation regions are empty, have noninteger bounds, are
                reversed or overlapping, or lie outside LoROM capacity.
        """
        self.regions = list(regions)
        if not self.regions:
            raise ValueError("At least one allocation region is required")
        previous_end = 0
        for start, end in self.regions:
            if (
                type(start) is not int
                or type(end) is not int
                or not previous_end <= start < end <= 0x400000
            ):
                raise ValueError("Allocation regions must be ordered, disjoint ROM ranges")
            previous_end = end
        self.free, self.end = self.regions.pop(0)

    def allocate(self, n: int) -> int:
        """
        Reserve n bytes and return their file offset.

        If the current bank has insufficient space, try the next bank boundary,
        then later configured regions. On success, earlier skipped spans are no
        longer available through this allocator. Zero returns the current cursor
        without advancing it.

        Args:
            n (int): Integer number of bytes to reserve, from 0 through 32768
                inclusive; booleans are rejected.

        Returns:
            int: Reserved region start as an unheadered file offset.

        Raises:
            AssertionError: The size is not an integer from 0 through 32768, or no
                allowed region can satisfy it. Allocation cursors remain unchanged.
        """
        if type(n) is not int or not 0 <= n <= 0x8000:
            raise AssertionError("Allocation size must be an integer and fit one 32KB bank")

        # Search without consuming regions: an unsuccessful large reservation
        # must leave smaller usable spans available to the caller.
        for index, (start, end) in enumerate([(self.free, self.end), *self.regions]):
            # The native reader increments only sixteen address bits, so a
            # payload must fit wholly within one 32 KiB bank.
            at = start
            if (at & 0x7FFF) + n > 0x8000:
                at = (at | 0x7FFF) + 1
            if at + n <= end:
                self.free, self.end = at + n, end
                del self.regions[:index]
                return at

        raise AssertionError("out of free space")
