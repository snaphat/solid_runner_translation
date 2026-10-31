"""
This command checks the text and reward animation on all 17 mission-result screens in Snes9x.

The --rom argument selects the English ROM file. The check writes trace.sfc under
--out with added code that records which glyphs the game draws. Snes9x loads that
file; the input ROM file is not modified.

The check selects the game's results room and mission index for each screen. The recorded glyphs
and captured screen pixels are used to check the displayed text.
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
from verification.evidence.read_text_layers import capture_text_layers
from verification.game.emulator_session import GameSession

# Number of records in the game's mission-results table; indices run from 0 to 16.
MISSION_COUNT = 17


def enter(emu: Emu, state: bytes, index: int) -> None:
    """
    Restore the prepared state and request the native mission-results room.

    The room-load request and report index are written to emulator memory without advancing
    emulation. The resulting screen is drawn after the caller advances frames.

    Args:
        emu (Emu): Live emulator receiving the state restoration and room/report selection
            writes.
        state (bytes): Serialized prepared state compatible with the loaded ROM.
        index (int): Zero-based mission-result record, from 0 through 16; booleans and
            noninteger values are rejected.

    Raises:
        ValueError: If index is not an integer from 0 through 16. Validation happens before
            restoring state or changing emulated memory.
        RuntimeError: If the emulator rejects the serialized state.
    """
    if type(index) is not int or not 0 <= index < MISSION_COUNT:
        raise ValueError("Mission-result index must be an integer from 0 through 16")

    emu.load_state(state)
    w = emu.read_wram_bytes()

    # Stage 23, variant 7 loads mission results; SRAM $013E selects the report.
    emu.poke(0x1AA8, bytes([23, 7]))
    emu.poke(0x1719, bytes([w[0x1719] | 0x80]))
    emu.poke(0x13E, bytes([index]), id_=0)


def validate_output_paths(rom: Path, out: Path) -> None:
    """
    Reject generated output paths that alias the input ROM.

    The scratch ROM, screenshots and results JSON are checked before any output is created.
    Resolved paths catch symbolic links and directory aliases; existing files are also compared
    by identity to catch hard links. This check does not reserve paths against subsequent
    filesystem changes.

    Args:
        rom (Path): Input candidate ROM whose contents must be preserved.
        out (Path): Directory where this command will write its output files.

    Raises:
        ValueError: If a generated output file resolves or hard-links to rom.
        OSError: If a path cannot be resolved or an existing file's identity cannot be checked.
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
    Validate every mission-result screen and save comparison screenshots and results.

    The command parses --rom and --out from the command line. It rejects input/output aliases
    before creating directories or starting emulation. Files in the output directory are
    replaced as the check proceeds.

    Raises:
        SystemExit: If CLI arguments are invalid or an output aliases the input ROM.
        OSError: If input/output paths cannot be accessed or output files cannot be written.
        AssertionError: If mission text or pixels fail comparison, the reward or total-funds DMA
            data is incorrect, reward counting does not finish, or the instrumented and ordinary
            ROM screens differ.
        RuntimeError: If ROM/state loading, startup, or required frame snapshots fail.
        ValueError: If instrumentation, source metadata, text chains, glyphs or frame data are
            invalid.
        IndexError: If a decoded text entry or jump extends beyond the ROM.
        KeyError: If decoded text contains an unsupported F8 command.
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
            frame_sram = game.emu.read_frame_memory(0)
            records = instrument.read_terminator_log(frame_sram)
            if not (len(records) == 9):
                raise AssertionError(f"Mission {index}: expected 9 text records, found {len(records)}")

            reward_glyphs = instrument.read_glyph_log(
                frame_sram, records[4]["draw_end"], records[5]["draw_end"]
            )
            reward_expected, _ = decode_literal_glyphs(
                rom, convert_snes_to_file(records[5]["start"])
            )
            if not ([g["font_offset"] // 22 for g in reward_glyphs] == reward_expected):
                raise AssertionError(f"Mission {index}: reward-label glyphs do not match the compiled text")

            reward_ink = screen_ink.reconstruct_screen_ink(rom, reward_glyphs)
            if not (reward_ink and all((0 <= x < 48 and 96 <= y < 112 for x, y in reward_ink))):
                raise AssertionError(f"Mission {index}: reward-label ink is empty or outside buffer rectangle (0, 96, 48, 112)")

            # Reward label occupies the first six tiles of native font line 6.
            v = game.emu.read_vram_bytes()
            reward_tiles = v[0xD800:0xD860] + v[0xD900:0xD960]
            footer_glyphs = instrument.read_glyph_log(
                frame_sram, records[6]["draw_end"], records[7]["draw_end"]
            )
            checks: list[dict[str, Any]] = []
            start = 0

            # Prefix, title, Complete, Report:, summary. Reward and total have
            # colored glyphs/runtime numbers and are checked separately below.
            for record in records[:5]:
                glyphs = instrument.read_glyph_log(frame_sram, start, record["draw_end"])
                start = record["draw_end"]
                expected, _ = decode_literal_glyphs(rom, convert_snes_to_file(record["start"]))
                if not ([g["font_offset"] // 22 for g in glyphs] == expected):
                    raise AssertionError(f"Mission {index}: captured glyphs do not match text record {record!r}")

                check = screen_ink.check_screen_pixels(
                    rom,
                    glyphs,
                    game.emu.read_rgb_rows(),
                    game.emu.read_frame_memory(3),
                    layers=capture_text_layers(
                        rom,
                        game.emu.read_frame_memory(0),
                        game.emu.read_frame_memory(2),
                        game.emu.read_frame_registers(),
                    ),
                )
                if not check["passed"]:
                    raise AssertionError(f"Mission {index}: text pixel comparison failed: {check!r}")

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
                    raise AssertionError(f"Mission {index}: the DMA queue is still nonempty after four additional frames")

                w, v = game.emu.read_wram_bytes(), game.emu.read_vram_bytes()
                if not (w[11264:14336] == summary):
                    raise AssertionError(f"Mission {index}: reward counting changed the report-summary buffer")

                # The native completion path sets SRAM $020D, then releases
                # its temporary font buffers. VRAM retains the final numbers;
                # after that point cleared WRAM is no longer a DMA source.
                complete = bool(game.S[0x20D])
                if not complete:
                    if not (v[55296:56320] == w[14336:15360]):
                        raise AssertionError(f"Mission {index}: reward tiles in VRAM do not match the WRAM DMA source")

                    if not (v[56320:57344] == w[15360:16384]):
                        raise AssertionError(f"Mission {index}: total-funds tiles in VRAM do not match the WRAM DMA source")

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
                raise AssertionError(f"Mission {index}: reward counting did not set the completion flag within the frame limit")

            game.emu.run(60)
            elapsed += 60
            footer = screen_ink.check_screen_pixels(
                rom,
                footer_glyphs,
                game.emu.read_rgb_rows(),
                game.emu.read_frame_memory(3),
                layers=capture_text_layers(
                    rom,
                    game.emu.read_frame_memory(0),
                    game.emu.read_frame_memory(2),
                    game.emu.read_frame_registers(),
                ),
            )
            if not footer["passed"]:
                raise AssertionError(f"Mission {index}: total-funds footer pixel comparison failed: {footer!r}")

            v = game.emu.read_vram_bytes()
            if not (v[55296:55392] + v[55552:55648] == reward_tiles):
                raise AssertionError(f"Mission {index}: reward-label tiles changed during reward counting")

            # Original label coordinates in font line 6 -> screen x40/y152.
            # At completion its native palette has finished fading to white.
            white = screen_ink.collect_bright_pixels(game.emu.read_rgb_rows())
            missing_reward = sum((x + 40, y + 56) not in white for x, y in reward_ink)
            if not (missing_reward == 0):
                raise AssertionError(f"Mission {index}: {missing_reward} reward-label pixels are missing from the completed screen")

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
                raise AssertionError(f"Mission {index}: static report pixels differ between the ordinary and instrumented ROMs")

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
                raise AssertionError(f"Mission {index}: the ordinary ROM did not reach the instrumented ROM's funds, completion flag and empty DMA queue within 5000 frames")

            plain.run(60)  # finish the same native total-funds palette fade
            plain.save_screenshot(str(args.out / "plain.png"))

            if not (
                (args.out / "plain.png").read_bytes()
                == (args.out / f"mission-{index:02d}-counting.png").read_bytes()
            ):
                raise AssertionError(f"Mission {index}: completed result screenshots differ between the ordinary and instrumented ROMs")
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
