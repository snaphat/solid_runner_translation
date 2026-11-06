"""
This module registers screen programs that exercise Turbo File menus with configured device-status and save-file fixtures.

Detection fixtures expose the native peripheral menu. Status-only and file-serving driver
fixtures exercise confirmations, progress messages, success and errors. Registrations retain
those device fixtures as a separate evidence category; the game still composes and draws every
captured screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

from core.validation.validate_integers import is_integer

if TYPE_CHECKING:
    from verification.screens.register_screen_programs import ScreenProgram
    from verification.screens.run_screen_checks import ScreenActions
from emulator.validate_controller_input import BTN

from verification.screens.programs.controller_sequences import (
    POWER_ON,
    RESET_WITH_SAVE,
    skip_attract_sequence,
    walk,
)

SCRIPTS: dict[str, ScreenProgram] = {}


# Detection changes which title-menu options are visible. Device responses
# come from the replacement driver; each check records which responses it supplies.
TURBO_FLAG = "$7E:1815 |= 01 (Turbo File detected at boot)"


def create_turbo_walkthrough(
    func: int, with_save: bool, presses: Iterable[tuple[str, int]]
) -> Callable[[ScreenActions], None]:
    """
    Create an input walkthrough for one Turbo File function.

    Integer inputs accept subclasses; booleans are rejected.

    Boolean options require bool values; numeric zero and one are rejected.

    Args:
        func (int): Zero-based Turbo File menu choice: Backup 0, Load 1,
            Delete 2 or File Check 3.
        with_save (bool): Whether the supplied start state contains a native
            saved slot.
        presses (Iterable[tuple[str, int]]): Ordered (button name, settling frames)
            pairs following function selection. Waits must be nonnegative integers;
            integer subclasses are accepted and booleans are rejected.

    Returns:
        Callable[[ScreenActions], None]: ScreenActions callback retaining diagnostic/peripheral
            attribution from its registry entry.

    Raises:
        TypeError: If with_save is not a boolean.
        ValueError: If function selection, button names or observation timing is
            invalid.
            Also raised if func is a boolean or noninteger.
    """
    if type(with_save) is not bool:
        raise TypeError("Boolean options require bool values: with_save")
    if not is_integer(func):
        raise ValueError("func must be an integer, not a boolean")

    presses = tuple(presses)
    if not 0 <= func < 4 or any(
        button not in BTN or not is_integer(wait) or wait < 0
        for button, wait in presses
    ):
        raise ValueError(
            "Turbo File walkthrough requires a valid function, buttons and waits"
        )

    def run(d: ScreenActions) -> None:
        """
        Walk the configured Turbo File menu function.

        Args:
            d (ScreenActions): Driver at power-on or reset-with-save, matching
                with_save; detection writes and menu inputs affect its emulator.
        """
        skip_attract_sequence(d)

        # This fixture sets detection only. The native menu still handles its
        # inputs; device-response fixtures are declared separately by the program registry.
        d.poke(0x1815, [d.e.read_wram_bytes()[0x1815] | 1], 2, TURBO_FLAG)
        d.press("a", 4, 200)
        d.press(
            "right" if with_save else "down", 4, 60
        )  # with a save: NEW GAME / CONTINUE  TURBO FILE
        d.press("a", 4, 400)

        for _ in range(func):
            d.press("down", 4, 60)

        walk(d, presses, f"x{func}")

    return run


# Zero-based native Turbo File menu order, shared by the input sequences.
# Native Turbo File menu order; the index is the number of Down presses.
_TF = ["Backup", "Load", "Delete", "FileCheck"]
for _f in range(4):
    for _save, _state in ((False, POWER_ON), (True, RESET_WITH_SAVE)):
        SCRIPTS[f"turbo-{_TF[_f].lower()}-{'save' if _save else 'empty'}"] = {
            "state": _state,
            "run": create_turbo_walkthrough(
                _f,
                _save,
                [
                    ("a", 200),
                    ("a", 200),
                    ("a", 300),
                    ("a", 300),
                    ("start", 200),
                    ("a", 300),
                    ("start", 200),
                ],
            ),
            "description": f"Turbo File > {_TF[_f]} ({('slot 1 saved' if _save else 'no saves')}), detection flag only.",
            "fixtures": [TURBO_FLAG],
        }


def create_turbo_fixture_capture(
    func: int, with_save: bool, inputs: Iterable[str], step_frames: int = 180
) -> Callable[[ScreenActions], None]:
    """
    This function creates a controller callback that captures a Turbo File menu using configured device replies.

    Each sample waits six emulated frames before capture, which adds three more settling frames.
    step_frames determines the number of six-frame waits and does not include those additional
    capture frames.

    Integer subclasses are accepted for numeric arguments; booleans are rejected. with_save
    requires a bool value.

    Args:
        func (int): Zero-based Turbo File menu choice: Backup 0, Load 1,
            Delete 2, or File Check 3.
        with_save (bool): Whether the supplied start state contains a native
            saved slot.
        inputs (Iterable[str]): Button names submitted individually after the
            function is selected.
        step_frames (int, optional): Positive multiple of six used as the
            explicit wait budget after menu entry and each button. There are
            step_frames // 6 captures per block, each adding its own settling wait.
            Defaults to 180.

    Returns:
        Callable[[ScreenActions], None]: ScreenActions callback retaining diagnostic/peripheral
            attribution from its registry entry.

    Raises:
        TypeError: If with_save is not a boolean.
        ValueError: If function selection, button names or observation timing is
            invalid.
            Also raised if any integer argument is a boolean or noninteger.
    """
    if type(with_save) is not bool:
        raise TypeError("Boolean options require bool values: with_save")
    if not is_integer(func) or not is_integer(step_frames):
        raise ValueError("func, step_frames must be integers, not booleans")

    inputs = tuple(inputs)
    if (
        not 0 <= func < 4
        or step_frames < 6
        or step_frames % 6
        or any(button not in BTN for button in inputs)
    ):
        raise ValueError(
            "Turbo File capture requires valid buttons and a positive six-frame interval"
        )

    def run(d: ScreenActions) -> None:
        """
        Capture the configured device-fixture scenario at short frame intervals.

        Args:
            d (ScreenActions): Driver using the registry's Turbo File device
                fixture; inputs and dense captures update its runner.
        """
        d.run(1200)  # the stubbed boot detection takes ~40 frames longer
        d.press("start", 4, 200)
        d.press("down", 4, 60)
        d.press("start", 4, 400)
        d.press("a", 4, 300)
        d.clear()  # the device check starts after the opening narration
        d.capture("title", "title menu (Turbo File detected by the game itself)")
        d.press("right" if with_save else "down", 4, 90)
        d.clear()
        d.press("a", 4, 0)

        for k in range(step_frames // 6):
            d.run(6)
            d.capture(f"open-{k:02d}", f"Turbo File screen, +{6 * (k + 1)}")

        for _ in range(func):
            d.press("down", 4, 60)

        # Use short samples after every input so progress, confirmation and
        # error text can be captured before the stand-in device response replaces it.
        for i, b in enumerate(inputs):
            d.clear()
            d.run(4, (b,))

            for k in range(step_frames // 6):
                d.run(6)
                d.capture(f"{i:02d}{b}-{k:02d}", f"after {b} #{i}, +{6 * (k + 1)}")

    return run


def _register_status_fixture(
    name: str,
    statuses: dict[int, int],
    func: int,
    save: bool,
    inputs: Iterable[str],
    note: str,
) -> None:
    """
    Register one Turbo File check with specified replacement-driver responses.

    The definition replaces any registration under turbo-stub-<name>. The statuses mapping is retained by
    reference; the input iterable is materialized by the callback factory.
    patch_turbo_driver() validates the status bytes when playback builds the
    instrumented ROM data; registration does not validate them.

    Args:
        name (str): Scenario suffix used in the turbo-stub registry key and
            device label.
        statuses (dict[int, int]): Device-call numbers mapped to fixture status
            bytes; other calls return zero.
        func (int): Zero-based Turbo File menu choice: Backup 0, Load 1,
            Delete 2 or File Check 3.
        save (bool): Whether to start after resetting with a native slot-one
            save instead of power-on.
        inputs (Iterable[str]): Button names passed to create_turbo_fixture_capture after
            function selection.
        note (str): Scenario description appended to the registry entry.
    """
    SCRIPTS[f"turbo-stub-{name}"] = {
        "state": RESET_WITH_SAVE if save else POWER_ON,
        "run": create_turbo_fixture_capture(func, save, inputs),
        "turbo_stub": name,
        "turbo_statuses": statuses,
        "description": f'Turbo File DEVICE STUB scenario "{name}": {note}',
        "fixtures": [
            f"Turbo File driver stub (emulate_turbo_file.py), statuses {statuses} (others 00); no other change"
        ],
    }


# Device call 4 returns status 0A in the success fixture; unspecified calls
# return zero through the stand-in driver.
_OK = {4: 0x0A}  # one file on the device (find-first 00, find-next "no more")


_register_status_fixture(
    "ok-backup",
    _OK,
    0,
    True,
    ["a", "a", "a", "start", "a"],
    "Backup: source slot 1, target, confirm.",
)


_register_status_fixture(
    "ok-load",
    _OK,
    1,
    True,
    ["a", "a", "a", "start", "a"],
    "Load: pick the device file, confirm.",
)


_register_status_fixture(
    "ok-delete",
    _OK,
    2,
    True,
    ["a", "a", "a", "start", "a"],
    "Delete: pick the device file, confirm.",
)


_register_status_fixture(
    "ok-filecheck", _OK, 3, True, ["a", "start", "a"], "File Check with no errors."
)


_register_status_fixture(
    "ok-backup-overwrite",
    _OK,
    0,
    True,
    ["a", "down", "a", "a", "start", "a"],
    "Backup onto the existing device file (overwrite).",
)


_register_status_fixture(
    "empty-load",
    {3: 0x0A, 4: 0x0A},
    1,
    True,
    ["a", "start", "a"],
    "Load with no file on the device.",
)


_register_status_fixture(
    "empty-delete",
    {3: 0x0A, 4: 0x0A},
    2,
    True,
    ["a", "start", "a"],
    "Delete with no file on the device.",
)


_register_status_fixture(
    "unformatted",
    {1: 0x12, 4: 0x0A},
    0,
    True,
    ["a", "a", "start", "a", "b"],
    'Device reports "not formatted" at boot; format it.',
)


_register_status_fixture(
    "unformatted-no",
    {1: 0x12, 4: 0x0A},
    0,
    True,
    ["down", "a", "start", "a"],
    'Device reports "not formatted"; decline.',
)


for _fn in (0x0F, 0x10):
    _register_status_fixture(
        f"filecheck-errors-{_fn:02x}",
        {4: 0x0A, _fn: 0x13},
        3,
        True,
        ["a", "a", "start", "a"],
        f"File Check, function {_fn:02X} reports an error; run maintenance.",
    )


for _code in range(1, 20):
    _register_status_fixture(
        f"backup-error-{_code:02d}",
        {4: 0x0A, 9: _code},
        0,
        True,
        ["a", "a", "a", "start", "a"],
        f"Backup, the first write call (function 09) returns error {_code}.",
    )


_register_status_fixture(
    "yes-delete",
    _OK,
    2,
    True,
    ["a", "left", "a", "start", "a"],
    "Delete: pick the device file, answer Yes.",
)


_register_status_fixture(
    "yes-load",
    _OK,
    1,
    True,
    ["a", "a", "left", "a", "start", "a"],
    "Load: pick the device file, then the target slot, answer Yes.",
)


_register_status_fixture(
    "yes-load-b",
    _OK,
    1,
    True,
    ["a", "left", "a", "a", "left", "a", "start"],
    "Load: alternative order of picker and confirmation.",
)


_register_status_fixture(
    "yes-overwrite",
    _OK,
    0,
    True,
    ["a", "down", "a", "left", "a", "start", "a"],
    "Backup onto the existing device file, answer Yes.",
)


_register_status_fixture(
    "yes-format",
    {1: 0x12, 4: 0x0A},
    0,
    True,
    ["left", "a", "start", "a"],
    'Device reports "not formatted" at boot; answer Yes to formatting.',
)


_register_status_fixture(
    "yes-maintain",
    {4: 0x0A, 0x10: 0x13},
    3,
    True,
    ["a", "left", "a", "start", "a"],
    "File Check reports errors; answer Yes to maintenance.",
)


# Alternating confirm/left inputs select Yes in successive native confirmation prompts.
# Alternate confirmation with Left to exercise each successive Yes response.
_YES = ["a", "left", "a", "left", "a", "left", "a", "left", "a"]


_register_status_fixture(
    "yes2-delete",
    _OK,
    2,
    True,
    _YES,
    "Delete: A/Left alternating, so every prompt is answered Yes.",
)


_register_status_fixture(
    "yes2-load",
    _OK,
    1,
    True,
    _YES,
    "Load: A/Left alternating, so every prompt is answered Yes.",
)


_register_status_fixture(
    "yes2-overwrite",
    _OK,
    0,
    True,
    ["a", "down", "a", "left", "a", "left", "a", "left", "a"],
    "Backup onto the existing device file, every prompt Yes.",
)


_register_status_fixture(
    "yes2-maintain",
    {4: 0x0A, 0x10: 0x13},
    3,
    True,
    _YES,
    "File Check reports errors; every prompt Yes (maintenance).",
)


_register_status_fixture(
    "yes2-maintain-0f",
    {4: 0x0A, 0x0F: 0x13},
    3,
    True,
    _YES,
    "File Check, function 0F reports errors; every prompt Yes.",
)


def _register_file_fixture(
    name: str,
    statuses: dict[int, int],
    func: int,
    save: bool,
    inputs: Iterable[str],
    note: str,
) -> None:
    """
    Register a device-fixture scenario with one stored file and a slot-one payload.

    Args:
        name (str): Scenario suffix used in the turbo-stub registry key and
            device label.
        statuses (dict[int, int]): Device-call numbers mapped to fixture status
            bytes; other calls return zero.
        func (int): Zero-based Turbo File menu choice: Backup 0, Load 1,
            Delete 2 or File Check 3.
        save (bool): Whether to start after resetting with a native slot-one
            save instead of power-on.
        inputs (Iterable[str]): Button names passed to create_turbo_fixture_capture after
            function selection.
        note (str): Scenario description appended to the registry entry.
    """
    _register_status_fixture(name, statuses, func, save, inputs, note)
    SCRIPTS[f"turbo-stub-{name}"]["turbo_file"] = True
    SCRIPTS[f"turbo-stub-{name}"].setdefault("fixtures", []).append(
        "device holds one file: the entry the game builds for a slot-1 backup and the 2 KB of save slot 1"
    )


_register_file_fixture(
    "file-load",
    _OK,
    1,
    True,
    _YES,
    "Load the device file (a real slot-1 save), every prompt Yes.",
)


_register_file_fixture(
    "file-load-empty-sram",
    _OK,
    1,
    False,
    _YES,
    "Load the device file into an empty save list, every prompt Yes.",
)


_register_file_fixture(
    "file-delete", _OK, 2, True, _YES, "Delete the device file, every prompt Yes."
)


_register_file_fixture(
    "file-overwrite",
    _OK,
    0,
    True,
    ["a", "down", "a", "left", "a", "left", "a", "left", "a"],
    "Backup slot 1 onto the existing device file, every prompt Yes.",
)


_register_file_fixture(
    "file-backup",
    _OK,
    0,
    True,
    _YES,
    "Backup slot 1 as a new file while the device holds one, every prompt Yes.",
)


_register_file_fixture(
    "file-overwrite2",
    _OK,
    0,
    True,
    ["a", "a", "down", "a", "left", "a", "left", "a", "left", "a"],
    "Backup slot 1 onto the existing device file (Down on the target list), every prompt Yes.",
)


_register_file_fixture(
    "file-overwrite2-up",
    _OK,
    0,
    True,
    ["a", "a", "up", "a", "left", "a", "left", "a", "left", "a"],
    "As file-overwrite2 with Up on the target list (wrap).",
)
