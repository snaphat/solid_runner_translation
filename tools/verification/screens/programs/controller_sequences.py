"""
This module supplies shared controller sequences and memory-fixture definitions used by screen-check programs.

The helpers submit inputs through ScreenActions and capture after configured waits or a traced
text draw. Inventory constants describe bounded WRAM writes; terminal constants describe SRAM
record availability. Named starts identify screen-runner setup states rather than serialized
emulator files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, TypedDict

from core.validation.validate_integers import is_integer

if TYPE_CHECKING:
    from verification.screens.run_screen_checks import ScreenActions


class MemoryFixture(TypedDict):
    """
    A documented bounded memory write used to prepare a screen scenario.

    Attributes:
        address (int): Byte offset relative to the selected memory region.
        data (list[int]): Replacement byte values written before playback.
        region (int): Libretro memory region ID, such as 2 for WRAM.
        why (str): Explanation of the fixture's purpose.
    """

    address: int
    data: list[int]
    region: int
    why: str


class QuestFlagFixture(TypedDict):
    """
    Description of a quest flag value established by a screen program.

    Attributes:
        address (str): Hexadecimal SNES bus address of the quest flag.
        value (int): Byte value established before capturing the screen.
        why (str): Explanation of which native menu state the flag selects.
    """

    address: str
    value: int
    why: str


# Libretro region 2 is WRAM: $7E:1AE9 begins the byte-per-item quantity array.
FULL_INVENTORY: MemoryFixture = {
    "address": 6889,
    "data": [80] * 98,
    "region": 2,
    "why": "Inventory fixture: every item slot holds 80, within the native item limit, so the original item menu lists every item.",
}


def walk(d: ScreenActions, inputs: Iterable[tuple[str, int]], tag: str) -> None:
    """
    Press each configured input and capture after its settling wait.

    Args:
        d (ScreenActions): Driver receiving the input sequence and captures.
        inputs (Iterable[tuple[str, int]]): Ordered (button name, settling frames)
            pairs; each button is held for four frames.
        tag (str): Prefix for capture labels and capture context.
    """
    for k, (b, wait) in enumerate(inputs):
        d.clear()
        d.press(b, 4, wait)
        d.capture(f"{tag}-{k:02d}", f"{tag}: after {b} #{k}")


UPGRADE_PARTS: MemoryFixture = {
    "address": 6889 + 98,
    "data": [1] * 26,
    "region": 2,
    "why": "Upgrade-part inventory (entries 98-123 of the $1AE9 array): one of each of the 26 parts, so the original garage Upgrade list shows every part.",
}


# Screen-start registry key for a native terminal with its main menu open.
TERMINAL = "terminal-open"


def read_game_mode(d: ScreenActions) -> int:
    """
    Read the current native game mode.

    Args:
        d (ScreenActions): Driver providing the native game-mode snapshot.

    Returns:
        int: Current native game mode from WRAM 0x179C.
    """
    # WRAM $179C is the native screen/game-mode byte, used to distinguish
    # a terminal list from its record viewer.
    return int(d.e.read_wram_bytes()[0x179C])


# Starting-state labels shared by screen definitions and driver setup: a fresh
# boot, or a core reset retaining the cartridge's SRAM, respectively.
POWER_ON = "power-on"
RESET_WITH_SAVE = "reset-with-save"


def catch_drawn(
    d: ScreenActions, target: str, frames: int = 900, keep: int = 10
) -> None:
    """
    Capture closely after a target record is drawn.

    Temporary menu text can disappear from the glyph log when dialogue starts, so detection and
    capture must closely follow its draw. If the target is never observed within frames, return
    without captures or an exception. The return value does not distinguish a timeout from
    completed sampling.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        d (ScreenActions): Driver whose draw trace and frame are sampled.
        target (str): Entry ID to detect in terminator-read records.
        frames (int, optional): Maximum frames to wait for the target draw.
            Defaults to 900.
        keep (int, optional): Capture count after detection. Each capture samples
            immediately and is followed by two further wait frames.
            Defaults to 10.

    Raises:
        ValueError: If the draw wait or capture count is negative.
            Also raised if any integer argument is a boolean or noninteger.
    """
    if not is_integer(frames) or not is_integer(keep):
        raise ValueError("frames, keep must be integers, not booleans")
    if frames < 0 or keep < 0:
        raise ValueError("Draw wait and capture count must be nonnegative")
    for f in range(frames):
        d.run(1)

        # Start captures as soon as the target terminator appears in the trace.
        # A later dialogue draw can clear transient menu text from that trace.
        if any(target in r["ids"] for r in d.read_records()):
            for k in range(keep):
                d.capture(
                    f"{target}-{k:02d}",
                    f"{target} drawn, +{f + 1 + 2 * k} frames",
                    settle=0,
                )
                d.run(2)

            return


# (Cartridge SRAM byte offset, record count) for terminal availability arrays:
# jobs, news, info, people, and foes, respectively. Each record uses one byte:
# bit 7 = available, bit 6 = read, bit 0 = second text variant.
ALL_AVAILABLE = [(0x141, 18), (0x153, 30), (0x171, 30), (0x18F, 20), (0x1A3, 38)]


def skip_attract_sequence(d: ScreenActions) -> None:
    """
    Leave the boot narration and return to the title's PUSH START prompt.

    Args:
        d (ScreenActions): Driver restored to power-on or reset-with-save;
            its SRAM and peripheral-detection flags remain intact.
    """
    d.run(900)
    d.press("start", 4, 120)
    d.press("down", 4, 60)
    d.press("start", 4, 300)


def select_field_menu_option(d: ScreenActions, option: int = 0) -> None:
    """
    Open the field menu and position its cursor without confirming a selection.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        d (ScreenActions): Driver at the roaming start; trace clearing remains
            the caller's responsibility so intermediate captures retain their text.
        option (int, optional): Zero-based cursor position: Items 0, Setup 1,
            Info 2 or Config 3. Each down press includes 30 settling frames.
            Defaults to 0.

    Raises:
        ValueError: If the option is outside the four native field-menu entries.
            Also raised if option is a boolean or noninteger.
    """
    if not is_integer(option):
        raise ValueError("option must be an integer, not a boolean")
    if not 0 <= option <= 3:
        raise ValueError("Field-menu option must be between 0 and 3")
    d.run(30)
    d.press("x", 4, 90)
    for _ in range(option):
        d.press("down", 4, 30)


def request_room_load(d: ScreenActions, stage: int, variant: int, reason: str) -> None:
    """
    Set a native room-load fixture and request its next-frame transition.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        d (ScreenActions): Driver applying the two WRAM writes; this helper
            does not retain a record of them.
        stage (int): Native stage byte, 0-255, for WRAM $1AA8.
        variant (int): Native room variant byte, 0-255, for WRAM $1AA9.
        reason (str): Explanation of the room selection passed to poke() for
            callback readability; it is not stored in capture results.

    Raises:
        ValueError: If either room component is a boolean, noninteger or outside 0-255.
    """
    if (
        not is_integer(stage)
        or not is_integer(variant)
        or not 0 <= stage <= 255
        or not 0 <= variant <= 255
    ):
        raise ValueError("stage and variant must be integer bytes, not booleans")
    d.poke(0x1AA8, [stage, variant], 2, reason)

    # The room pair alone does not initiate loading. Set the request bit
    # while retaining the other native transition flags in WRAM $1719.
    d.poke(
        0x1719,
        [d.e.read_wram_bytes()[0x1719] | 0x80],
        2,
        "Room-load request: $1719 | $80.",
    )
