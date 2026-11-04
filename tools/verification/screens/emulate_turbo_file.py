"""
This module builds a Turbo File driver for verification and modifies ROM data in memory without changing the input buffer or writing a ROM file.

Snes9x does not emulate the Turbo File device. The replacement driver is installed at the ASCII
driver entry at SNES address $A4:8000. The native caller passes a function number in A; the
driver delays its reply using four 65,535-iteration decrement loops, then returns a configured
status code in A. The game stores that code in $1811 and branches on it.

The file-serving variant also supplies a directory entry for function 03 and a save-slot payload
for function 0C. Native menu, confirmation and drawing code still executes. Verification callers
must label captures made with this replacement driver; these checks do not verify real
peripheral hardware.
"""

from __future__ import annotations

from typing import cast

from core.files.validate_binary_buffers import require_byte_buffer
from core.rom.convert_addresses import convert_snes_to_file
from core.validation.validate_integers import is_integer

# All addresses below are 24-bit SNES LoROM bus addresses, converted by convert_snes_to_file().
# Driver code and function-indexed status tables for status-only/file-serving variants.
STUB, TABLE, FILE_TABLE = 0x92FF00, 0x92FF40, 0x92FFC0
DATA_AT, ENTRY_AT = 0xE68000, 0xE68800  # 0x800-byte save payload, then 24-byte directory entry.
DRIVER = 0xA48000  # Native ASCII driver entry replaced by a long jump.

# The status-only driver preserves processor flags, X, and Y. It returns the
# status in A and leaves the data-bank register unchanged.
# PHP / REP #$30 / PHX / PHY / AND #$00FF / PHA / LDY #4 / {LDX #$FFFF / DEX / BNE / DEY / BNE} /
# PLX / LDA $92FF40,X / AND #$00FF / PLY / PLX / PLP / RTL
CODE = bytes.fromhex(
    "08 c2 30 da 5a 29 ff 00 48 a0 04 00 a2 ff ff ca d0 fd 88 d0 f7 fa bf 40 ff 92 29 ff 00 7a fa 28 6b"
)

# The directory entry the game's own Backup builds at $7F:AC80 for slot 1.
ENTRY = b"SHVC0000S-RUNNER  1     "


def build_file_driver() -> bytes:
    """
    The returned driver supplies one directory entry and one save-slot payload.

    Function 03 copies the 24-byte directory entry from $E6:8800 to $7F:100C.
    Function 0C copies the 0x800-byte save payload from $E6:8000 to $7F:D000.
    Other supported function IDs perform neither copy. The caller must supply a
    function ID from 0 through 16 in the low byte of A; the driver does not check
    that range during emulation. Every call returns the status byte
    indexed by the function ID in the table at $92:FFC0.

    The driver saves and restores processor flags, X, Y, and the data-bank
    register. It uses 16-bit accumulator and index registers internally and
    returns the configured status in A. The block-copy instructions temporarily
    set the data-bank register to $7F.

    Returns:
        bytes: Native driver bytecode serving one directory entry and one
            save-slot payload.
    """
    # Save flags, X, Y, and DB; select 16-bit registers and retain the function ID
    # on the stack. Four decrement loops delay the reply before function dispatch.
    c = bytearray.fromhex("08 c2 30 da 5a 8b 29 ff 00 48 a0 04 00 a2 ff ff ca d0 fd 88 d0 f7 68 48")
    find = bytes.fromhex("a2 00 88 a0 0c 10 a9 17 00 54 7f e6")  # MVN $E6->$7F, the 24-byte entry
    read = bytes.fromhex("a2 00 80 a0 00 d0 a9 ff 07 54 7f e6")  # MVN $E6->$7F, the 0x800-byte file
    # Recover the saved function ID into X, load its reply, then restore DB,
    # Y, X, and flags. A carries the reply back through RTL.
    done = bytes.fromhex("fa bf c0 ff 92 29 ff 00 ab 7a fa 28 6b")
    c += (
        bytes.fromhex("c9 03 00")
        + bytes([0xD0, len(find) + 2])
        + find
        + bytes([0x80, 3 + 2 + len(read)])
    )
    c += bytes.fromhex("c9 0c 00") + bytes([0xD0, len(read)]) + read
    return bytes(c + done)


def patch_turbo_driver(
    rom: bytes | bytearray, statuses: dict[int, int], file_data: bytes | None = None
) -> bytes:
    """
    The function returns modified ROM data with configured Turbo File replies.

    The function creates a new buffer in memory with bytearray(rom), patches that buffer, and
    returns bytes. It does not modify the supplied buffer or write a file. The status-only
    variant supplies per-function return codes. Supplying file_data also installs a native
    directory entry and one save-slot payload, letting the game's own Load path copy real data.
    Hook and free-space checks reject unexpected ROM layouts before the modified bytes are
    returned.

    Args:
        rom (bytes | bytearray): Input ROM bytes. A new bytearray is allocated
            from this buffer before patching; the supplied buffer is unchanged.
            Other types are rejected before conversion.
        statuses (dict[int, int]): Turbo File function ID to injected driver
            return status. A dictionary or dictionary subclass is required. IDs
            must be integers from 0 through 16, and replies integers from 0 through
            255; integer subclasses are accepted and booleans are rejected.
            Function IDs omitted from the dictionary return status 0.
        file_data (bytes | None, optional): One 0x800-byte save-slot
            payload stored as bytes; other buffer or iterable types are rejected.
            None installs the status-only driver without a file payload.

    Returns:
        bytes: Patched ROM bytes containing the driver and optional save-slot
            data. The supplied buffer is unchanged.

    Raises:
        TypeError: If rom is neither bytes nor bytearray, statuses is not a
            dictionary, or file_data is neither bytes nor None.
        ValueError: If a function ID or reply is not an integer in its accepted range,
            file_data is not exactly 2048 bytes, the original jump at $A4:8000 does not
            match, or the reserved driver or file-data space contains non-FF bytes.
    """
    # These type checks prevent bytearray() and slice assignment from interpreting
    # integers as byte counts or boolean elements as payload bytes.
    require_byte_buffer(rom, (bytes, bytearray))
    if not isinstance(cast(object, statuses), dict):
        raise TypeError("Turbo File statuses must be a dictionary")
    if file_data is not None:
        require_byte_buffer(file_data, (bytes,))
        if len(file_data) != 0x800:
            raise ValueError("Turbo File save payload must contain exactly 2048 bytes")

    if any(
        (not is_integer(function))
        or (not is_integer(status))
        or not 0 <= function <= 0x10
        or not 0 <= status <= 255
        for function, status in statuses.items()
    ):
        raise ValueError("Turbo File function IDs must be integers 0-16 and replies integers 0-255")
    rom = bytearray(rom)

    # The served file lives in the expanded fourth MiB.
    # Any expansion affects the newly allocated bytearray; the input is unchanged.
    # A stored-file fixture includes native save data in addition to status
    # responses; only that scenario needs the extra payload allocation.
    if file_data is not None and len(rom) < convert_snes_to_file(DATA_AT) + 0x900:
        rom.extend(b"\xff" * (0x400000 - len(rom)))

    if bytes(rom[convert_snes_to_file(DRIVER) : convert_snes_to_file(DRIVER) + 4]) != bytes.fromhex(
        "5c a3 90 a4"
    ):
        raise ValueError("Unexpected Turbo File driver entry")

    if set(rom[convert_snes_to_file(STUB) : convert_snes_to_file(STUB) + 0x100]) != {0xFF}:
        raise ValueError("$92:FF00 is no longer free")

    table = bytes(statuses.get(f, 0) for f in range(0x11))
    if file_data is None:
        rom[convert_snes_to_file(STUB) : convert_snes_to_file(STUB) + len(CODE)] = CODE
        rom[convert_snes_to_file(TABLE) : convert_snes_to_file(TABLE) + len(table)] = table
    else:
        if set(
            rom[convert_snes_to_file(DATA_AT) : convert_snes_to_file(DATA_AT) + 0x900]
        ) != {0xFF}:
            raise ValueError("Turbo File stub file area is not free")

        code = build_file_driver()
        rom[convert_snes_to_file(STUB) : convert_snes_to_file(STUB) + len(code)] = code
        rom[convert_snes_to_file(FILE_TABLE) : convert_snes_to_file(FILE_TABLE) + len(table)] = (
            table
        )
        rom[convert_snes_to_file(DATA_AT) : convert_snes_to_file(DATA_AT) + 0x800] = file_data
        rom[convert_snes_to_file(ENTRY_AT) : convert_snes_to_file(ENTRY_AT) + len(ENTRY)] = ENTRY

    rom[convert_snes_to_file(DRIVER) : convert_snes_to_file(DRIVER) + 4] = bytes(
        [0x5C]
    ) + STUB.to_bytes(3, "little")
    return bytes(rom)
