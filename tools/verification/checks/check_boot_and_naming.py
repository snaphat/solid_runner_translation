"""
This check boots the English ROM and verifies the initial naming screen and default surname.

The check runs the ordinary ROM with controller input, confirms the expected
room and default surname, and saves a naming-screen screenshot. It detects boot
or name-initialization faults without instrumenting or rewriting the release.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # tools/
from emulator.snes9x_core import Emu
from patches.interface.patch_naming import FAMILY, encode_name

SCREENSHOT = "outputs/naming-smoke.png"


def run_cli() -> None:
    """
    Boot the English ROM and check the hero-naming screen.

    Raises:
        RuntimeError: Fresh boot does not reach hero naming, or the expected
            default surname is absent.
    """
    e = Emu("build/Solid Runner (English).sfc")
    try:
        e.run(900)

        # Confirm the opening title/name-entry transitions through native input
        # before inspecting the naming screen.
        for _ in range(4):
            e.press("start", 4, after=300)

        room = list(e.read_wram_bytes()[0x1AA8:0x1AAA])
        if room != [22, 5]:
            raise RuntimeError(f"Fresh boot did not reach hero naming: {room}")

        if e.read_wram_bytes()[0x1AC3:0x1AC8] != encode_name(FAMILY):
            raise RuntimeError("Unexpected default surname")

        e.press("r", 4, after=120)
        Path("outputs").mkdir(exist_ok=True)
        e.save_screenshot(SCREENSHOT)
        print(f"Fresh boot / hero naming passed. Screenshot: {SCREENSHOT}")
    finally:
        e.lib.retro_unload_game()
        e.lib.retro_deinit()


if __name__ == "__main__":
    run_cli()
