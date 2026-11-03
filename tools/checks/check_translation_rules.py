"""
This command checks active translation entries for encoding, storage, and layout errors before compilation.

An in-memory copy of the original ROM receives the installed Latin font so text measurements use
the same glyphs as compilation. Every active translation entry is checked for command
preservation, byte capacity and visible layout limits. The command reports entry-specific
diagnostics and fails if any rule is broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.rom import load_original_rom as original_rom
from core.text import measure_text as text_measurement
from core.text import read_translation_files as script_files
from core.text import validate_translation as script_rules


def prepare_validation_rom() -> bytearray:
    """
    Return a mutable original ROM padded to 4 MiB with the Latin font installed.

    Returns:
        bytearray: Independent 4 MiB image containing the validation font tables;
            the original ROM file and its cached bytes remain unchanged. Only the
            font patch is installed; this is not the full English build.

    Raises:
        OSError: If the original ROM or artwork cannot be read.
        ValueError: If the original ROM hash does not match, font artwork contains
            invalid JSON or bitmap definitions, or the ROM lacks a required font table.
        KeyError: If required ligature artwork is missing or a registered artwork
            key has no game encoding.
    """
    from patches.interface.install_font import patch_font

    rom = bytearray(original_rom.read_original_rom())
    rom.extend(b"\xff" * (0x400000 - len(rom)))

    # Measure against the installed English glyphs, not the original Japanese
    # font that happens to occupy the same code slots.
    patch_font(rom)
    return rom


def run_cli() -> int:
    """
    Check active translations against compiler rules; return zero on success.

    Returns:
        int: Zero if every active entry passes, otherwise one after printing
            entry-specific diagnostics. The printed problem count counts diagnostic
            messages, not unique entries.
            Source-load, font-installation, and name-measurement errors propagate
            instead of producing a numeric status.

    Raises:
        OSError: If the original ROM, font artwork or a translation source cannot
            be read.
        ValueError: If the original ROM hash does not match; artwork, translation
            text or metadata is malformed; an active translation is blank; no active
            system names are available; a name contains control markup or a newline;
            or a required font record is outside the ROM.
        KeyError: If required ligature artwork is missing, a registered artwork
            key has no game encoding, or a system name contains an unsupported character.
    """
    rom = prepare_validation_rom()
    name_width = text_measurement.measure_name_width(rom, script_files.load_names_for_dialogue_layout())

    problems = 0

    # Validate source entries against the installed Latin font in the scratch image,
    # so width checks use the glyphs that compilation will actually install.
    for name in script_files.NAMES:
        for e in script_files.load(name):
            if e.get("skip"):
                continue

            for err in script_rules.validate_translation_entry(e, rom, name_width=name_width):
                problems += 1
                print(f"translation/{name}.txt #### {e['id']}: {err}")

    print(f"script rules: {problems} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(run_cli())
