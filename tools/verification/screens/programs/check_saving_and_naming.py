"""
This module registers controller programs that exercise save, load, and hero or Runner naming screens and capture their text.

Named starts identify native save opportunities. Programs capture prompts, reopen saved-slot
lists, and exercise character entry from fresh-boot naming screens. Registrations describe the
required start; the shared driver owns emulator state, trace collection and screenshot
evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from verification.screens.register_screen_programs import ScreenProgram
    from verification.screens.run_screen_checks import ScreenActions
from verification.screens.programs.controller_sequences import (
    POWER_ON,
    RESET_WITH_SAVE,
    skip_attract_sequence,
    walk,
)

SCRIPTS: dict[str, ScreenProgram] = {}


# Named replay starts at native save opportunities, not arbitrary room loads.
# Defined in screens/data/screen_starts.json; each state preserves the progression
# needed to open that save device.
SAVE_STARTS = [
    "save-02-006r",
    "save-06-002",
    "save-06-038",
    "save-08-026v",
    "save-09-042v",
    "save-11-000",
    "save-13-022",
    "save-13-052v",
    "save-15-052",
    "save-17-008",
    "save-18-041v",
    "save-18-121",
    "save-20-050v",
    "save-25-041r",
    "save-25-084",
    "save-25-134r",
    "save-25-135r",
    "save-26-074r",
    "save-27-053",
    "save-28-014",
    "save-30-037",
    "save-31-023r",
]


def save_here(d: ScreenActions) -> None:
    """
    Save into a slot and capture its native prompts.

    Args:
        d (ScreenActions): Driver at one of SAVE_STARTS, with the native Save
            Data list open and slot 1 selected. Inputs save into that slot;
            captures record its prompts and resulting location label.
    """
    d.run(20)
    d.capture("list", "Save Data list")
    d.clear()
    d.press("a", 4, 90)  # slot 1 -> SAVE OK? YES/NO
    d.capture("confirm", "save confirmation")
    d.clear()
    d.press("a", 4, 6)  # YES -> Saving...
    d.capture("saving", "saving")
    d.clear()
    d.run(240)
    d.capture("saved", "slot after saving (location name)")


def _collect_save_starts() -> list[tuple[str, str]]:
    """
    Return the named native save opportunities registered for screen playback.

    Returns:
        list[tuple[str, str]]: Registration-name/start-key pairs, with both values
            taken from the same native save-start name.
    """
    return [(name, name) for name in SAVE_STARTS]


for _name, _state in _collect_save_starts():
    SCRIPTS[f"save-{_name[5:]}"] = {
        "state": _state,
        "run": save_here,
        "description": f"Save device in room {_name[5:]}: Save Data list, confirmation, saving, saved slot with its location.",
    }


def save_reopen(d: ScreenActions) -> None:
    """
    Reopen the native save menu, perform a slot save, and capture its prompts.

    The sequence leaves and reenters the menu before confirming the save, then waits for and
    captures the saved slot. It changes the session's SRAM.

    Args:
        d (ScreenActions): Driver restored to the save-28-014 start.
    """
    d.run(20)
    walk(d, [("b", 150), ("a", 200), ("a", 150), ("a", 90), ("a", 6), ("a", 30)], "r")
    d.clear()
    d.run(240)
    d.capture("r-saved", "slot after saving")


SCRIPTS["save-reopen-28-014"] = {
    "state": "save-28-014",
    "run": save_reopen,
    "description": "Save device in room 28/14, re-opened under trace.",
}


def save_saving(d: ScreenActions) -> None:
    """
    Capture the saving-in-progress message.

    Keeps one trace because the slot list draws the saving label before the save starts.

    Args:
        d (ScreenActions): Driver restored to the save-28-014 start.
    """
    d.run(20)
    d.press("b", 4, 150)
    d.clear()
    d.press("a", 4, 200)  # talk -> Save Data list (draws all labels)
    d.capture("list", "Save Data list, drawn under trace")
    d.press("a", 4, 150)  # slot 1 -> SAVE OK? YES/NO
    d.capture("confirm", "save confirmation")
    d.press("a", 4, 0)  # YES -> Saving... (brief)

    for k in range(20):
        d.run(4)
        d.capture(f"saving-{k:02d}", f"while saving, +{7 * (k + 1)} frames")


SCRIPTS["save-saving-28-014"] = {
    "state": "save-28-014",
    "run": save_saving,
    "description": "Save device in room 28/14: re-opened under one trace, captured while saving.",
}


def capture_intro(d: ScreenActions) -> None:
    """
    Capture the power-on attract narration without submitting input.

    Waits 60 frames before each sample without clearing the trace; capture adds three settling
    frames. A capture receives credit only for the visible page.

    Args:
        d (ScreenActions): Driver restored to the power-on start.
    """
    d.run(240)

    for k in range(90):
        d.run(60)
        d.capture(f"n-{k:02d}", f"intro narration, frame {303 + 63 * k}")


SCRIPTS["intro"] = {
    "state": POWER_ON,
    "run": capture_intro,
    "description": "Power-on attract narration (no input, empty SRAM).",
}


def capture_hero_naming(d: ScreenActions) -> None:
    """
    Capture the hero name-entry screen and its controls.

    Starts a new game from the title with empty SRAM.

    Args:
        d (ScreenActions): Driver restored to the power-on start.
    """
    d.run(900)
    d.press("start", 4, 120)
    d.press("start", 4, 300)
    d.press("start", 4, 300)  # title menu: NEW GAME
    d.clear()
    d.press("start", 4, 300)
    d.capture("open", "name entry opened")
    walk(
        d,
        [
            ("r", 90),
            ("l", 90),
            ("a", 60),
            ("select", 90),
            ("b", 60),
            ("b", 60),
            ("b", 60),
            ("b", 60),
            ("b", 60),
            ("b", 60),
            ("select", 90),
            ("r", 90),
            ("b", 60),
            ("b", 60),
            ("b", 60),
            ("b", 60),
            ("l", 90),
            ("select", 90),
        ],
        "n",
    )


SCRIPTS["naming"] = {
    "state": POWER_ON,
    "run": capture_hero_naming,
    "description": "Title -> NEW GAME -> name entry (empty SRAM, native inputs).",
}


def capture_runner_naming(d: ScreenActions) -> None:
    """
    Capture the Runner name-entry screen.

    Uses the naming screen opened during Riff's first garage visit.

    Args:
        d (ScreenActions): Driver restored to the runner-naming start.
    """
    d.run(120)
    d.capture("open", "Runner name entry opened (help line)")
    walk(d, [("r", 90), ("l", 90), ("a", 60), ("b", 60), ("select", 90)], "rn")


SCRIPTS["runner-naming"] = {
    "state": "runner-naming",
    "run": capture_runner_naming,
    "description": "First garage visit: Riff asks for the Runner name; the Runner naming screen (22/4).",
}


def capture_load_menu(d: ScreenActions, button: Literal["a", "start"]) -> None:
    """
    Select Continue from the title menu and capture the native Load Data list.

    Args:
        d (ScreenActions): Driver at the captured title menu with a saved slot.
        button (Literal["a", "start"]): Confirmation input for this title-menu check;
            both native input paths are exercised by separate programs.

    Raises:
        ValueError: If button is neither a nor start.
    """
    if button not in ("a", "start"):
        raise ValueError("Title-load confirmation must be a or start")
    d.press("down", 4, 60)
    d.clear()
    d.press(button, 4, 300)
    d.capture("load", "Load Data screen")


def capture_title_load(d: ScreenActions) -> None:
    """
    Capture Continue and its saved-slot selection.

    Uses a reset-with-save start that preserves a slot saved through the native save screen; the
    callback itself does not reset the console.

    Args:
        d (ScreenActions): Driver restored to the reset-with-save start.
    """
    d.run(1500)
    d.press("start", 4, 300)  # PUSH START -> NEW GAME / CONTINUE
    d.clear()  # opening narration is no longer on screen
    d.capture("menu", "title menu")
    capture_load_menu(d, "start")
    walk(d, [("down", 60), ("down", 60), ("up", 60), ("up", 60)], "l")


SCRIPTS["title-load"] = {
    "state": RESET_WITH_SAVE,
    "run": capture_title_load,
    "description": "Title CONTINUE -> Load Data after a reset that keeps SRAM.",
    "fixtures": [
        "Starting state: console reset after a native save at the room 28/14 save device."
    ],
}


def capture_empty_name_prompts(d: ScreenActions) -> None:
    """
    Capture naming prompts for an empty name.

    Erases the entered name and attempts confirmation with START.

    Args:
        d (ScreenActions): Driver restored to the power-on start.
    """
    skip_attract_sequence(d)
    d.press("a", 4, 200)  # -> NEW GAME
    d.clear()  # discard the attract-mode narration before name entry
    d.press("a", 4, 300)  # -> name entry
    d.capture("open", "name entry")

    # Erase beyond the maximum name length so confirmation exercises the
    # native empty-name response rather than retaining a default character.
    for _ in range(12):
        d.press("b", 4, 30)

    d.clear()
    d.press("start", 4, 0)

    for k in range(20):
        d.run(6)
        d.capture(f"e-{k:02d}", f"START on an empty name, +{9 * (k + 1)} frames")


def capture_additional_load_states(d: ScreenActions) -> None:
    """
    Capture additional title-load and slot-selection states.

    Args:
        d (ScreenActions): Driver restored to the reset-with-save start.
    """
    skip_attract_sequence(d)
    d.press("a", 4, 200)  # -> NEW GAME / CONTINUE (CONTINUE selected)
    d.clear()  # opening narration is no longer on screen
    d.capture("menu", "title menu")
    capture_load_menu(d, "a")
    walk(d, [("down", 60), ("down", 60), ("up", 60)], "l")


SCRIPTS["naming-empty"] = {
    "state": POWER_ON,
    "run": capture_empty_name_prompts,
    "description": "Name entry: erase the name, press START.",
}


SCRIPTS["title-load2"] = {
    "state": RESET_WITH_SAVE,
    "run": capture_additional_load_states,
    "description": "Title CONTINUE -> Load Data after a reset that keeps SRAM.",
    "fixtures": [
        "Starting state: console reset after a native save at the room 28/14 save device."
    ],
}
