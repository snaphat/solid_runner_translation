"""
This module decodes the background settings recorded while a captured frame was drawn.

The normal NMI register writes define the initial frame state. Recorded IRQ handlers and
captured HDMA transfers divide the visible frame into bands with their own tilemaps and scrolls.
The host reads HDMA payloads from the captured core memory associated with the image; no
emulated CPU table-copy routine changes the display timing. Only enabled two-bit backgrounds
using the native text graphics at VRAM $4000 or $C000 are returned. Unknown or repeated IRQ
handlers invalidate capture evidence.
"""

from __future__ import annotations

import ctypes
from typing import TypedDict

from verification.evidence.instrument_rom import DISPLAY

# B-bus register offsets written by HDMA modes zero through seven, in transfer
# order. The destination is the channel's $43n1 byte plus each offset, modulo 256.
HDMA_REGISTER_OFFSETS = (
    (0,),
    (0, 1),
    (0, 0),
    (0, 0, 1, 1),
    (0, 1, 2, 3),
    (0, 1, 0, 1),
    (0, 0),
    (0, 0, 1, 1),
)


def _targets_text_register(destination: int, mode: int) -> bool:
    """
    Determine whether any byte of an HDMA transfer changes text placement.

    Args:
        destination (int): Channel's eight-bit B-bus base register number.
        mode (int): HDMA transfer mode, zero through seven.

    Returns:
        bool: True if a write reaches $2105-$2114 or $212C/$212D, even when
            the transfer starts at a different register.
    """
    registers = {(destination + offset) & 255 for offset in HDMA_REGISTER_OFFSETS[mode]}
    return any(register in (0x2C, 0x2D) or 5 <= register <= 0x14 for register in registers)


class TextLayer(TypedDict):
    """
    One background's text placement within a visible scanline band.

    Attributes:
        layer (int): Background number, one through three.
        base (int): Byte offset of the first tilemap page in VRAM.
        size (int): BGnSC size bits; bit zero adds a horizontal page and bit one a vertical page.
        font_base (int): Byte address of the active two-bit character graphics in VRAM.
        tile_size (int): Hardware tile-cell width and height, eight or sixteen pixels.
        dx (int): Hardware horizontal scroll, modulo 1024 pixels.
        dy (int): Hardware vertical scroll plus the first-scanline adjustment, modulo 1024.
        top (int): First included output scanline.
        bottom (int): Exclusive final output scanline.
    """

    layer: int
    base: int
    size: int
    font_base: int
    tile_size: int
    dx: int
    dy: int
    top: int
    bottom: int


def capture_text_layers(
    rom: bytes | bytearray,
    sram: bytes,
    wram: bytes,
    registers: bytes,
    height: int = 224,
) -> list[TextLayer]:
    """
    This function determines text-background positions from memory and display-register data captured with the screenshot.

    Python code reads the captured HDMA tables and combines their register changes with the
    recorded interrupt handlers. It does not run additional emulated CPU instructions, write to
    game RAM, or read write-only PPU registers.

    Args:
        rom (bytes | bytearray): Release ROM supplying unchanged HDMA table data.
        sram (bytes): Cartridge SRAM copied with the image, including NMI/IRQ metadata.
        wram (bytes): Work RAM copied with that same image; supplies the applied PPU
            shadows and any RAM-backed HDMA tables or indirect payloads.
        registers (bytes): Frame register image returned by Emu.read_frame_registers();
            its $4300-$437F slots contain decoded DMA configuration from the same
            snapshot's separate DMA block, rather than unpopulated FillRAM bytes.
        height (int, optional): Visible scanlines to decode, zero through 256.
            Only built-in int values are accepted; booleans and int subclasses are rejected.
            Defaults to 224.

    Returns:
        list[TextLayer]: Supported enabled text backgrounds and their scanline bands.

    Raises:
        ValueError: If height is not a built-in int from zero through 256,
            frame memory is incomplete, an HDMA source is unavailable, display
            capture metadata is invalid, a text-register HDMA transfer uses
            the unsupported reverse direction, or a captured channel stream
            exceeds its 1 KiB slot.
    """
    if type(height) is not int or not 0 <= height <= 256:
        raise ValueError("Visible height must be an integer from zero through 256")
    if len(sram) < DISPLAY + 0x2400 or len(wram) != 0x20000 or len(registers) != 0x8000:
        raise ValueError("Frame memory is incomplete for display capture")
    data = bytearray(sram)
    data[DISPLAY : DISPLAY + 0x60] = wram[0x100:0x160]
    data[DISPLAY + 0x200 : DISPLAY + 0x280] = registers[0x4300:0x4380]

    def read_bus_byte(bank: int, address: int) -> int:
        """
        Read an HDMA source byte from the captured memory through the game's LoROM memory mapping.

        Args:
            bank (int): Source bus bank, zero through 255.
            address (int): Source address within the bank; wraps at 65536.

        Returns:
            int: Captured RAM, SRAM or ROM byte at the supplied bus address.

        Raises:
            ValueError: If the address selects unavailable ROM, SRAM or unsupported MMIO.
        """
        address &= 0xFFFF
        if bank in (0x7E, 0x7F):
            return wram[(bank - 0x7E) * 0x10000 + address]
        if (bank & 0x7F) < 0x40 and address < 0x2000:
            return wram[address]
        if (bank & 0x7F) >= 0x70 and address < 0x8000:
            if not sram:
                raise ValueError("HDMA source selects unavailable SRAM")
            return sram[((bank & 3) * 0x8000 + address) % len(sram)]
        if address >= 0x8000:
            offset = (bank & 0x7F) * 0x8000 + (address & 0x7FFF)
            if offset < len(rom):
                return rom[offset]
        raise ValueError(f"Unavailable HDMA source ${bank:02X}:{address:04X}")

    for channel in range(8):
        if not data[DISPLAY + 0x62] & (1 << channel):
            continue
        base = 0x4300 + channel * 16
        control, destination = registers[base : base + 2]
        if not _targets_text_register(destination, control & 7):
            continue
        if control & 0x80:
            raise ValueError("Display HDMA uses an unsupported reverse transfer")
        width = len(HDMA_REGISTER_OFFSETS[control & 7])
        cursor = int.from_bytes(registers[base + 2 : base + 4], "little")
        bank = registers[base + 4]
        stream = bytearray()
        lines = 0
        while lines < height:
            count = read_bus_byte(bank, cursor)
            cursor = (cursor + 1) & 0xFFFF
            if not count:
                break
            span = min(count & 127 or 128, height - lines)
            repeated = bool(count & 128)
            stream.append((128 if repeated else 0) | (span & 127))
            size = width * (span if repeated else 1)
            if control & 0x40:
                pointer = read_bus_byte(bank, cursor) | (read_bus_byte(bank, cursor + 1) << 8)
                cursor = (cursor + 2) & 0xFFFF
                payload_bank = registers[base + 7]
            else:
                pointer, payload_bank = cursor, bank
                cursor = (cursor + size) & 0xFFFF
            stream.extend(read_bus_byte(payload_bank, pointer + i) for i in range(size))
            if len(stream) >= 0x400:
                raise ValueError("Display HDMA capture exceeds its 1 KiB slot")
            lines += span
        stream.append(0)
        target = DISPLAY + 0x400 + channel * 0x400
        data[target : target + len(stream)] = stream
    return read_text_layers(bytes(data), height)


def read_text_layers(
    sram: bytes | ctypes.Array[ctypes.c_ubyte], height: int = 224
) -> list[TextLayer]:
    """
    This function decodes text-background positions from the combined display snapshot prepared by capture_text_layers().

    The input includes copied PPU settings, interrupt records, and HDMA table data.
    Raw cartridge SRAM does not contain all of those fields and is insufficient.

    Args:
        sram (bytes | ctypes.Array[ctypes.c_ubyte]): SRAM snapshot populated with
            applied PPU shadows, IRQ records and copied HDMA streams, as prepared
            by capture_text_layers(). Raw cartridge SRAM alone is insufficient.
        height (int, optional): Visible frame height in scanlines, from zero through 256.
            Only built-in int values are accepted; booleans and int subclasses
            are rejected.
            Defaults to 224.

    Returns:
        list[TextLayer]: Supported enabled text backgrounds, with consecutive matching scanlines
            combined into bands for each background. Bands from different backgrounds
            may overlap.
            An empty list means that no supported text background was enabled.

    Raises:
        ValueError: If height is invalid, snapshots are missing, IRQ evidence is ambiguous, or
            display HDMA evidence is unsupported or truncated.
    """
    if type(height) is not int or not 0 <= height <= 256:
        raise ValueError("Visible height must be an integer from zero through 256")
    if len(sram) < DISPLAY + 0x280 or sram[DISPLAY + 0x61] != 0xD7:
        raise ValueError("Frame has no display-register snapshot")
    if sram[DISPLAY + 0x64]:
        raise ValueError("Frame has an invalid display IRQ or HDMA snapshot")

    data = bytes(sram[DISPLAY : DISPLAY + 0x280])
    mode = data[0x11]
    maps = list(data[0x13:0x16])
    fonts = [data[0x17] & 15, data[0x17] >> 4, data[0x18] & 15]
    scrolls = [
        [int.from_bytes(data[i : i + 2], "little"), int.from_bytes(data[i + 2 : i + 4], "little")]
        for i in (0x22, 0x26, 0x2A)
    ]
    # $80:836D writes zero as BG3VOFS's high byte, even when the RAM
    # calculation stored a negative sixteen-bit value in $012C-$012D.
    scrolls[2][1] &= 255
    cpu_events = sorted(
        (data[0x101 + handler * 16] + 1, handler)
        for handler in range(0, 11, 2)
        if data[0x100 + handler * 16]
    )
    if data[0x1C0]:
        cpu_events.append((data[0x1C1] + 1, -1))
        cpu_events.sort()
    hdma = _read_hdma_writes(bytes(sram[DISPLAY:]), height)
    main, sub = (0, 0) if data[0x60] & 0x21 == 0x21 else (data[0x1B], data[0x1C])
    latch = 0  # The normal final BG3VOFS write supplies a zero high byte.
    layers: list[TextLayer] = []
    last: dict[int, TextLayer] = {}

    def write_register(register: int, value: int) -> None:
        """
        Apply a captured byte write to the background register model.

        Args:
            register (int): Low byte of the PPU destination address, $05-$14 or $2C/$2D.
            value (int): Captured unsigned byte written at this scanline.
        """
        nonlocal mode, main, sub, latch
        if register == 5:
            mode = value
        elif 7 <= register <= 9:
            maps[register - 7] = value
        elif register == 11:
            fonts[0], fonts[1] = value & 15, value >> 4
        elif register == 12:
            fonts[2] = value & 15
        elif 13 <= register <= 20:
            index, axis = divmod(register - 13, 2)
            if index < 3:
                if axis:
                    scrolls[index][1] = (value << 8) | latch
                else:
                    scrolls[index][0] = (value << 8) | (latch & ~7) | ((scrolls[index][0] >> 8) & 7)
            # BG4 shares the scroll-write latch even though it is not a text layer.
            latch = value
        elif register == 0x2C:
            main = value
        elif register == 0x2D:
            sub = value

    event_index = 0
    for y in range(height):
        while event_index < len(cpu_events) and cpu_events[event_index][0] <= y:
            line, handler = cpu_events[event_index]
            event_index += 1
            record = 0x100 + handler * 16
            if handler == -1:
                main, sub = data[0x1B], data[0x1C]
            elif handler == 2:
                main = sub = 0
            elif handler == 4:
                mode, main, sub = 0x19, 5, 0
                maps[0] = maps[2] = 0x70
                fonts[0], fonts[2] = 3, 6
                scrolls[0] = [0, (-(line + 1)) & 255]
                scrolls[2] = [0, 0x300 | ((-(line + 1)) & 255)]
                latch = 3
            elif handler == 6:
                scrolls[0][0] = int.from_bytes(data[record + 4 : record + 6], "little")
                latch = data[record + 5]
            elif handler == 8:
                mode, main, sub = 9, 6, 0
                scrolls[1] = [0, 0]
                scrolls[2] = [0, 0x35F]
                latch = 0
            elif handler == 10:
                scrolls[2][0] = int.from_bytes(data[record + 2 : record + 4], "little")
                latch = data[record + 3]
        for channel, register, value in hdma.get(y, []):
            if data[0x1C0] and y >= data[0x1C1] + 1 and data[0x63] & (1 << channel):
                continue
            write_register(register, value)
        enabled = main | sub
        for index in range(3):
            two_bit = mode & 7 == 0 or (mode & 7 == 1 and index == 2)
            if not enabled & (1 << index) or not two_bit or fonts[index] not in (2, 6):
                continue
            layer = TextLayer(
                layer=index + 1,
                base=(maps[index] & 0xFC) << 9,
                size=maps[index] & 3,
                font_base=fonts[index] << 13,
                tile_size=16 if mode & (0x10 << index) else 8,
                dx=scrolls[index][0] & 1023,
                dy=(scrolls[index][1] + 1) & 1023,
                top=y,
                bottom=y + 1,
            )
            previous = last.get(index)
            if (
                previous is not None
                and previous["bottom"] == y
                and all(
                    previous[key] == layer[key]
                    for key in ("layer", "base", "size", "font_base", "tile_size", "dx", "dy")
                )
            ):
                previous["bottom"] = y + 1
            else:
                layers.append(layer)
                last[index] = layer
    return layers


def _read_hdma_writes(data: bytes, height: int) -> dict[int, list[tuple[int, int, int]]]:
    """
    Decode captured HDMA transfers in scanline and channel order.

    Args:
        data (bytes): Display snapshot beginning at DISPLAY, including channel
            descriptors and bounded streams of captured direct or indirect payloads.
        height (int): Number of visible scanlines whose register writes are needed.

    Returns:
        dict[int, list[tuple[int, int, int]]]: Scanline to ordered channel/register/value writes.

    Raises:
        ValueError: If a relevant channel has missing, truncated or reverse-transfer evidence.
    """
    writes: dict[int, list[tuple[int, int, int]]] = {}
    if len(data) < 0x280:
        raise ValueError("Frame lacks HDMA channel descriptors")
    for channel in range(8):
        if not data[0x62] & (1 << channel):
            continue
        descriptor = 0x200 + channel * 16
        control, destination = data[descriptor], data[descriptor + 1]
        if not _targets_text_register(destination, control & 7):
            continue
        if control & 0x80:
            raise ValueError("Display HDMA uses an unsupported reverse transfer")
        pattern = HDMA_REGISTER_OFFSETS[control & 7]
        start = 0x400 + channel * 0x400
        stream = data[start : start + 0x400]
        offset = line = 0
        while line < height:
            if offset >= len(stream):
                raise ValueError("Captured HDMA table is truncated")
            count = stream[offset]
            offset += 1
            if not count:
                break
            lines = count & 127 or 128
            for repeat in range(lines):
                if repeat == 0 or count & 128:
                    if offset + len(pattern) > len(stream):
                        raise ValueError("Captured HDMA payload is truncated")
                    values = stream[offset : offset + len(pattern)]
                    offset += len(pattern)
                    if line + repeat < height:
                        writes.setdefault(line + repeat, []).extend(
                            (channel, (destination + register_offset) & 255, value)
                            for register_offset, value in zip(pattern, values)
                        )
            line += lines
    return writes
