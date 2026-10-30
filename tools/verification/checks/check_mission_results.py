"""
This check displays all 17 mission-result screens and checks their text pixels and reward animation.

The --rom argument selects the candidate image when the command is run from
the repository root.
The fixture selects the game's results room and mission index. The game
composes each screen; only disposable copies receive glyph-tracing hooks.
Generated output paths must not resolve or hard-link to the input ROM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.rom.convert_addresses import convert_snes_to_file
from core.text.read_compiled_text import decode_literal_glyphs
from emulator.snes9x_core import Emu
from verification.evidence import check_screen_pixels as screen_ink
from verification.evidence import instrument_rom as instrument
from verification.game.emulator_session import GameSession

# Number of records in the game's mission-results table; indices run from 0 to 16.
MISSION_COUNT = 17


def enter(emu: Emu, state: bytes, index: int) -> None:
    """
    Restore the prepared state and request the native mission-results room.

    Set the room-load request and report index without advancing emulation.
    The caller must run frames before inspecting the resulting screen.

    Args:
        emu (Emu): Live emulator receiving the state restoration and room/report
            selection writes.
        state (bytes): Serialized prepared state compatible with the loaded
            ROM.
        index (int): Zero-based mission-result record, from 0 through 16;
            booleans and noninteger values are rejected.

    Raises:
        ValueError: index is not an integer from 0 through 16. Validation
            happens before restoring state or changing emulated memory.
    """
    if type(index) is not int or not 0 <= index < MISSION_COUNT:
        raise ValueError("Mission-result index must be an integer from 0 through 16")

    emu.load_state(state)
    w = emu.read_wram_bytes()

    # Stage 23/7 owns the native results screen; SRAM $013E selects its report.
    emu.poke(0x1AA8, bytes([23, 7]))
    emu.poke(0x1719, bytes([w[0x1719] | 0x80]))
    emu.poke(0x13E, bytes([index]), id_=0)


def validate_output_paths(rom: Path, out: Path) -> None:
    """
    Reject generated artifact paths that alias the input ROM.

    Checks the scratch ROM, screenshots and results JSON before any output is
    created. Resolved paths catch symbolic links and directory aliases;
    existing files are also compared by identity to catch hard links. This
    check does not reserve paths against subsequent filesystem changes.

    Args:
        rom (Path): Input candidate ROM whose contents must be preserved.
        out (Path): Directory where this command will write its artifacts.

    Raises:
        ValueError: A generated artifact resolves or hard-links to rom.
        OSError: A path cannot be resolved or an existing file's identity
            cannot be checked.
    """
    source = rom.resolve()
    outputs = [out / name for name in ("trace.sfc", "plain.png", "results.json")]
    outputs.extend(
        out / f"mission-{index:02d}{suffix}.png"
        for index in range(MISSION_COUNT)
        for suffix in ("", "-counting")
    )
    for output in outputs:
        if output.resolve() == source or (output.exists() and output.samefile(source)):
            raise ValueError(f"Generated output aliases the input ROM: {output}")


def run_cli() -> None:
    """
    Validate every mission-result screen and archive comparison artifacts.

    Parse --rom and --out from the command line. Reject input/output aliases
    before creating directories or starting emulation. Artifacts in the output
    directory are replaced as the check proceeds.

    Raises:
        SystemExit: CLI arguments are invalid or an output aliases the input ROM.
        OSError: Input/output paths cannot be accessed or artifacts cannot be written.
        AssertionError: A mission result violates expected text, layout, DMA
            transfer, reward-counting or footer invariants.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=Path, default=Path("build/Solid Runner (English).sfc"))
    parser.add_argument("--out", type=Path, default=Path("build/mission-results"))
    args = parser.parse_args()
    try:
        validate_output_paths(args.rom, args.out)
    except ValueError as error:
        parser.error(str(error))

    args.out.mkdir(parents=True, exist_ok=True)
    rom = args.rom.read_bytes()
    game = GameSession(rom, args.out / "trace.sfc", accumulate=True)
    results: list[dict[str, Any]] = []
    static_frames: list[list[bytes]] = []
    try:
        game.boot_to_roam()
        state = game.emu.save_state()
        for index in range(MISSION_COUNT):
            enter(game.emu, state, index)
            game.clear_trace()
            game.emu.run(653)
            records = instrument.read_terminator_log(game.S)
            if not (len(records) == 9):
                raise AssertionError((index, len(records)))

            reward_glyphs = instrument.read_glyph_log(
                game.S, records[4]["draw_end"], records[5]["draw_end"]
            )
            reward_expected, _ = decode_literal_glyphs(
                rom, convert_snes_to_file(records[5]["start"])
            )
            if not ([g["font_offset"] // 22 for g in reward_glyphs] == reward_expected):
                raise AssertionError((index, "reward label text"))

            reward_ink = screen_ink.reconstruct_screen_ink(rom, reward_glyphs)
            if not (reward_ink and all((0 <= x < 48 and 96 <= y < 112 for x, y in reward_ink))):
                raise AssertionError((index, "reward label bounds"))

            # Reward label occupies the first six tiles of native font line 6.
            v = game.emu.read_vram_bytes()
            reward_tiles = v[0xD800:0xD860] + v[0xD900:0xD960]
            footer_glyphs = instrument.read_glyph_log(
                game.S, records[6]["draw_end"], records[7]["draw_end"]
            )
            checks: list[dict[str, Any]] = []
            start = 0

            # Prefix, title, Complete, Report:, summary. Reward and total have
            # colored glyphs/runtime numbers and are checked separately below.
            for record in records[:5]:
                glyphs = instrument.read_glyph_log(game.S, start, record["draw_end"])
                start = record["draw_end"]
                expected, _ = decode_literal_glyphs(rom, convert_snes_to_file(record["start"]))
                if not ([g["font_offset"] // 22 for g in glyphs] == expected):
                    raise AssertionError((index, record))

                check = screen_ink.check_screen_pixels(
                    rom, glyphs, game.emu.read_rgb_rows(), game.emu.read_vram_bytes()
                )
                if not (check["outside_ink"] == check["missing_screen_ink"] == 0):
                    raise AssertionError((index, check))

                checks.append(check)

            game.emu.save_screenshot(str(args.out / f"mission-{index:02d}.png"))

            # The reward is already counting on some 653-frame captures.
            # Above y=153 the report/title/background are static; compare that
            # region directly, then compare the full screen at the same final
            # reward state below.
            static_frames.append(game.emu.read_rgb_rows()[:153])
            summary = game.emu.read_wram_bytes()[0x2C00:0x3800]
            frames: list[dict[str, Any]] = []
            elapsed = 653

            for count in (120, 120, 120):
                game.emu.run(count)
                elapsed += count

                # A video frame can end after the CPU draws a new number but
                # before NMI drains the native DMA queue. Check the completed
                # transfer, with a strict bound, rather than that pending frame.
                for _ in range(4):
                    if game.emu.read_wram_bytes()[0x6B] == 0:
                        break

                    game.emu.run(1)
                    elapsed += 1

                if not (game.emu.read_wram_bytes()[107] == 0):
                    raise AssertionError((index, "DMA queue did not drain"))

                w, v = game.emu.read_wram_bytes(), game.emu.read_vram_bytes()
                if not (w[11264:14336] == summary):
                    raise AssertionError((index, "summary overwritten"))

                # The native completion path sets SRAM $020D, then releases
                # its temporary font buffers. VRAM retains the final numbers;
                # after that point cleared WRAM is no longer a DMA source.
                complete = bool(game.S[0x20D])
                if not complete:
                    if not (v[55296:56320] == w[14336:15360]):
                        raise AssertionError((index, "reward DMA"))

                    if not (v[56320:57344] == w[15360:16384]):
                        raise AssertionError((index, "total DMA"))

                frames.append(
                    dict(
                        complete=complete, font_sha256=hashlib.sha256(v[0xD800:0xDC00]).hexdigest()
                    )
                )

            # Let the native counter finish and the total-funds palette finish
            # fading in. Verify the label and total-funds footer as well.
            for _ in range(60):
                if game.S[0x20D]:
                    break

                game.emu.run(60)
                elapsed += 60

            if not (game.S[525]):
                raise AssertionError((index, "reward count did not finish"))

            game.emu.run(60)
            elapsed += 60
            footer = screen_ink.check_screen_pixels(
                rom, footer_glyphs, game.emu.read_rgb_rows(), game.emu.read_vram_bytes()
            )
            if not (footer["outside_ink"] == footer["missing_screen_ink"] == 0):
                raise AssertionError((index, "footer", footer))

            v = game.emu.read_vram_bytes()
            if not (v[55296:55392] + v[55552:55648] == reward_tiles):
                raise AssertionError((index, "reward label overwritten"))

            # Original label coordinates in font line 6 -> screen x40/y152.
            # At completion its native palette has finished fading to white.
            white = screen_ink.collect_bright_pixels(game.emu.read_rgb_rows())
            missing_reward = sum((x + 40, y + 56) not in white for x, y in reward_ink)
            if not (missing_reward == 0):
                raise AssertionError((index, "reward label missing", missing_reward))

            game.emu.save_screenshot(str(args.out / f"mission-{index:02d}-counting.png"))
            results.append(
                dict(
                    mission=index,
                    checks=checks,
                    reward_frames=frames,
                    footer=footer,
                    reward_label_ink=len(reward_ink),
                    counting_frame=elapsed,
                    counting_total=int.from_bytes(
                        game.emu.read_wram_bytes()[0x1AD6:0x1AD9], "little"
                    ),
                    counting_complete=bool(game.S[0x20D]),
                )
            )
    finally:
        game.close()

    if not (
        any((not frame["complete"] for result in results for frame in result["reward_frames"]))
    ):
        raise AssertionError("No active reward counting was checked")

    # Compare actual screens against an ordinary candidate with no trace hooks.
    plain = Emu(str(args.rom))
    try:
        for index in range(MISSION_COUNT):
            enter(plain, state, index)
            plain.run(653)
            plain.save_screenshot(str(args.out / "plain.png"))

            if not (plain.read_rgb_rows()[:153] == static_frames[index]):
                raise AssertionError(index)

            # Tracing consumes CPU cycles, so an equal frame count need not
            # land on the same reward tick. Match the native money/completion
            # state and a drained queue before comparing the full image.
            target = results[index]
            for _ in range(5000):
                w = plain.read_wram_bytes()
                total = int.from_bytes(w[0x1AD6:0x1AD9], "little")
                if (
                    total == target["counting_total"]
                    and not w[0x6B]
                    and bool(plain.read_memory_bytes(0)[0x20D]) == target["counting_complete"]
                ):
                    break

                plain.run(1)
            else:
                raise AssertionError((index, "ordinary reward counter did not reach traced state"))

            plain.run(60)  # finish the same native total-funds palette fade
            plain.save_screenshot(str(args.out / "plain.png"))

            if not (
                (args.out / "plain.png").read_bytes()
                == (args.out / f"mission-{index:02d}-counting.png").read_bytes()
            ):
                raise AssertionError(index)
    finally:
        plain.lib.retro_unload_game()
        plain.lib.retro_deinit()

    (args.out / "results.json").write_text(
        json.dumps(dict(rom_sha256=hashlib.sha256(rom).hexdigest(), missions=results), indent=2)
    )
    print(
        "PASS: all 17 results screens; literal text and visible pixels; reward/total DMA; ordinary-ROM screenshots"
    )


if __name__ == "__main__":
    run_cli()
