"""
This module checks compiled translation entries against the game's text-reading rules.

verify_compiled_text() loads active entries from the translation collections,
locates their text in a supplied 4 MiB ROM, and uses the compiled-text reader
to check relocation jumps, dialogue page boundaries, and name-copy restrictions.
It prints structural failures and a final failure count, returning zero on
success or one for an invalid image size or a reported text-chain failure.
The function reads translation sources and leaves the ROM unchanged.
"""

from __future__ import annotations

from typing import cast

from core.text import read_compiled_text
from core.text import read_translation_files as script_files


def verify_compiled_text(rom: bytes | bytearray) -> int:
    """
    Validate all active translations in a compiled 4 MiB ROM.

    Args:
        rom (bytes | bytearray): Compiled English ROM to validate; must
            be exactly 4 MiB. Other types are rejected before translation
            sources are loaded.

    Returns:
        int: Zero when every active entry validates; one on a ROM-size or
            text-chain failure.

    Raises:
        TypeError: rom is neither bytes nor bytearray.
    """
    if not isinstance(cast(object, rom), (bytes, bytearray)):
        raise TypeError("Compiled ROM must be bytes or bytearray")

    # Compiled text may live in expansion banks, so structural validation requires
    # the complete 4 MiB English image.
    if len(rom) != 0x400000:
        print(f"Invalid English ROM size: {len(rom)} bytes; expected {0x400000}")
        print("bad 1")
        return 1

    reader = read_compiled_text.Reader(rom)
    bad = 0

    for f in script_files.NAMES:
        for e in script_files.load(f):
            if not e["en"] or e.get("skip"):
                continue

            mode = e.get("mode", "v")
            if e.get("ptrs") or e.get("imm"):
                # Repointed pointers/immediates lead to strings with their own FF terminator.
                refs = read_compiled_text.read_entry_references(rom, e)

                for t in refs:
                    err = reader.validate_text_chain(t, -1, mode)
                    if err:
                        bad += 1
                        print(f, e["id"], "via pointer", err, repr(e["en"][:40]))

                continue

            # Entries without retargeted references must return to their original slot
            # terminator. Dialogue also checks agreement with the event interpreter scan.
            a, b = int(e["id"], 16), int(e["end"], 16)
            err = (
                reader.validate_dialogue_segments(a, b)
                if mode == "v"
                else reader.validate_text_chain(a, b, mode)
            )
            if err:
                bad += 1
                print(f, e["id"], mode, err, repr(e["en"][:40]))

    print("bad", bad)
    return 1 if bad else 0
