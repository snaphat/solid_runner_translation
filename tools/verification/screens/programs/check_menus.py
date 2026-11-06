"""
This module registers controller programs that open field, inventory, equipment, and party menus and capture their text.

Programs open native menu paths and capture the labels, descriptions and cursor states they
expose. Inventory and equipment fixtures make list contents available. SCRIPTS registers each
callback with its native start and fixture descriptions; the shared screen runner grades the
resulting captures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from core.validation.validate_integers import is_integer

if TYPE_CHECKING:
    from verification.screens.register_screen_programs import ScreenProgram
    from verification.screens.run_screen_checks import ScreenActions
from verification.screens.programs.controller_sequences import (
    ALL_AVAILABLE,
    FULL_INVENTORY,
    catch_drawn,
    select_field_menu_option,
    walk,
)

SCRIPTS: dict[str, ScreenProgram] = {}

# Native field-menu interiors, excluding their border tiles. Coordinates are
# screen pixels at the unscrolled menu layout, with exclusive right/bottom edges.
# The main choices occupy the upper window; Info's choices occupy the lower one.
FIELD_MENU_TEXT_BOXES = {
    "01448F": (16, 32, 96, 96),
    "0144BF": (16, 32, 96, 48),
    "014515": (16, 88, 88, 136),
}


def capture_field_menu(d: ScreenActions) -> None:
    """
    Capture the ordinary field-menu options.

    Args:
        d (ScreenActions): Driver restored to the roam start.
    """
    d.run(30)
    d.clear()
    d.press("x", 4, 90)
    d.capture("menu", "field menu (X)")


def capture_item_lists(d: ScreenActions) -> None:
    """
    Capture item lists and descriptions with the inventory fixture.

    Args:
        d (ScreenActions): Driver restored to the roam start.
    """
    f = FULL_INVENTORY
    d.poke(f["address"], f["data"], f["region"], f["why"])
    select_field_menu_option(d, 0)
    d.clear()
    d.press("a", 4, 90)  # Items: cursor on the item list, first description shown
    d.capture("list", "Items: item list and first description")

    # Clear the trace for each cursor move so its capture belongs to the
    # currently highlighted item and description.
    for i in range(1, 48):
        d.clear()
        d.press("down", 4, 40)
        d.capture(f"cursor-{i:02d}", f"item list cursor {i}")


SCRIPTS.update(
    {
        "field-menu": {
            "state": "roam",
            "run": capture_field_menu,
            "description": "Field menu opened with X.",
        },
        "items": {
            "state": "roam",
            "run": capture_item_lists,
            "fixtures": [FULL_INVENTORY],
            "description": "Field menu > Items; cursor moved down the item list, one capture per position (name list scrolls, description of the highlighted item).",
        },
    }
)


def capture_setup_menu(d: ScreenActions) -> None:
    """
    Capture the field Setup menu.

    Args:
        d (ScreenActions): Driver restored to the roam start.
    """
    select_field_menu_option(d, 1)

    # Enter Setup, open a slot's parts list, then return and visit the other
    # slot lists. Each tuple supplies a button and settling frames after the press.
    walk(
        d,
        [
            ("a", 90),
            ("a", 90),
            ("down", 40),
            ("down", 40),
            ("b", 60),
            ("down", 40),
            ("a", 90),
            ("b", 60),
            ("down", 40),
            ("a", 90),
            ("b", 60),
            ("down", 40),
            ("a", 90),
        ],
        "setup",
    )


def capture_info_menu(d: ScreenActions) -> None:
    """
    Capture the field Info menu.

    Args:
        d (ScreenActions): Driver restored to the roam start.
    """
    select_field_menu_option(d, 2)

    # Capture Status and Job, then the Data submenu's Information and Cast
    # views, returning to each parent menu before moving its cursor.
    walk(
        d,
        [
            ("a", 90),
            ("a", 120),
            ("b", 60),
            ("down", 40),
            ("a", 120),
            ("b", 60),
            ("down", 40),
            ("a", 90),
            ("a", 120),
            ("b", 60),
            ("down", 40),
            ("a", 120),
        ],
        "info",
    )


def capture_config_menu(d: ScreenActions) -> None:
    """
    Capture the configuration menu options.

    Args:
        d (ScreenActions): Driver restored to the roam start.
    """
    select_field_menu_option(d, 3)

    # Open the first Config option, return, then visit successive options
    # with a fresh draw capture after each input.
    walk(
        d,
        [
            ("a", 90),
            ("a", 90),
            ("b", 60),
            ("down", 40),
            ("a", 90),
            ("b", 60),
            ("down", 40),
            ("a", 90),
        ],
        "config",
    )


SCRIPTS.update(
    {
        "setup": {
            "state": "roam",
            "run": capture_setup_menu,
            "description": "Field menu > Setup: slots, part lists per slot.",
        },
        "info": {
            "state": "roam",
            "run": capture_info_menu,
            "description": "Field menu > Info: Status, Job, Data > Info/Cast.",
        },
        "config": {
            "state": "roam",
            "run": capture_config_menu,
            "description": "Field menu > Config: Sound, Text Speed, Controller.",
        },
    }
)


def capture_equipment_lists(d: ScreenActions) -> None:
    """
    Capture each Setup slot list with every weapon owned.

    Args:
        d (ScreenActions): Driver restored to the roam start.
    """
    f = FULL_INVENTORY
    d.poke(f["address"], f["data"], f["region"], f["why"])
    select_field_menu_option(d, 1)
    d.press("a", 4, 90)  # Setup: slot picker

    for slot in range(4):
        d.clear()
        d.press("a", 4, 120)  # open this slot's weapon list
        d.capture(f"s{slot}-00", f"Setup slot {slot}: list top")

        for i in range(1, 60):
            d.clear()
            d.press("down", 4, 40)
            d.capture(f"s{slot}-{i:02d}", f"Setup slot {slot}: cursor {i}")

        d.press("b", 4, 90)
        d.press("down", 4, 40)


SCRIPTS["setup-full"] = {
    "state": "roam",
    "run": capture_equipment_lists,
    "fixtures": [FULL_INVENTORY],
    "description": "Field menu > Setup with every weapon owned (full_inventory); each slot list stepped through.",
}


def create_job_capture(job: int) -> Callable[[ScreenActions], None]:
    """
    Create an accepted-job display callback.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        job (int): Accepted-job SRAM index; 0xFF selects the no-current-job
            case.

    Returns:
        Callable[[ScreenActions], None]: ScreenActions callback for the configured Info menu state.

    Raises:
        ValueError: If the job is neither an index from 0 through 17 nor the no-job
            value 255.
            Also raised if job is a boolean or noninteger.
    """
    if not is_integer(job):
        raise ValueError("job must be an integer, not a boolean")

    if not (0 <= job < 18 or job == 255):
        raise ValueError("Job must be 0-17 or the no-job sentinel 255")

    def run(d: ScreenActions) -> None:
        """
        Capture the Info menu for the configured current job.

        Args:
            d (ScreenActions): Driver at the roaming start; job-state fixtures and
                menu inputs update its emulator.
        """
        d.poke(
            0x13B,
            [job],
            0,
            f"Quest fixture: current job SRAM $70:013B = {job:#04x} (accepted job shown by Info > Job).",
        )

        if job != 0xFF:
            d.poke(
                0x141 + job,
                [0xC0],
                0,
                f"Quest fixture: job {job} flags $70:{0x141 + job:04X} = C0 (available + accepted).",
            )

        select_field_menu_option(d, 2)
        d.press("a", 4, 90)  # Info
        d.press("down", 4, 30)  # Job
        d.clear()
        d.press("a", 4, 150)
        d.capture("job", f"Info > Job with current job {job:#04x}")

    return run


# The FF current-job value represents no accepted job; other values select
# the corresponding Info > Job record.
for _job in list(range(12)) + [0xFF]:
    SCRIPTS[f"info-job-{_job:02x}"] = {
        "state": "roam",
        "run": create_job_capture(_job),
        "fixtures": [
            {"address": "70:013B", "value": _job, "why": "current job (quest fixture)"}
        ],
        "description": f"Field menu > Info > Job with current job {_job:#04x}.",
    }


for _job in range(0x0C, 0x12):
    SCRIPTS[f"info-job-{_job:02x}"] = {
        "state": "roam",
        "run": create_job_capture(_job),
        "fixtures": [
            {"address": "70:013B", "value": _job, "why": "current job (quest fixture)"}
        ],
        "description": f"Field menu > Info > Job with current job {_job:#04x}.",
    }


def create_info_data_capture(value: int) -> Callable[[ScreenActions], None]:
    """
    Create an Info header and Data screen capture callback.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        value (int): Availability/read flag byte applied to every terminal
            record, typically 0x80 or 0xC0.

    Returns:
        Callable[[ScreenActions], None]: ScreenActions callback for the configured Info menu state.

    Raises:
        ValueError: If record availability is outside 0 through 255.
            Also raised if value is a boolean or noninteger.
    """
    if not is_integer(value):
        raise ValueError("value must be an integer, not a boolean")

    if not 0 <= value <= 255:
        raise ValueError("Record availability must be a byte")

    def run(d: ScreenActions) -> None:
        """
        Record Info initialization and capture the configured Data list.

        Args:
            d (ScreenActions): Driver at the roaming start; availability writes and
                menu inputs produce recorded captures.
        """
        for lo, n in ALL_AVAILABLE:
            d.poke(
                lo,
                [value] * n,
                0,
                f"terminal records (SRAM unlock bytes = {value:02X})",
            )

        select_field_menu_option(d, 2)
        d.clear()
        # Preserve initialization observations separately from visible labels.
        # Window 9 replaces the Info buffer with the selected submenu text.
        for frame in range(4):
            d.run(1, ("a",))
            d.capture(f"info-opening-{frame}", "Info menu opening", settle=0)
        catch_drawn(d, "0144BF", frames=120, keep=6)
        d.press("down", 4, 40)
        d.press("down", 4, 40)  # Data
        walk(d, [("a", 120), ("a", 120), ("b", 60), ("b", 60)], "data")

    return run


SCRIPTS["info-header-data-80"] = {
    "state": "roam",
    "run": create_info_data_capture(128),
    "description": "Field menu > Info: record its overwritten initialization, then verify Info > Data with information available.",
    "fixtures": ["SRAM availability bytes $70:0141/0153/0171/018F/01A3 = 80"],
}


SCRIPTS["info-header-data-c0"] = {
    "state": "roam",
    "run": create_info_data_capture(192),
    "description": "As info-header-data-80 with records read (C0).",
    "fixtures": ["SRAM availability bytes $70:0141/0153/0171/018F/01A3 = C0"],
}


def capture_controller_pages(d: ScreenActions) -> None:
    """
    Capture controller configuration pages.

    Args:
        d (ScreenActions): Driver restored to the roam start.
    """
    select_field_menu_option(d, 3)

    d.press("a", 4, 90)
    d.press("down", 4, 40)
    d.press("down", 4, 40)

    # Move past the area-map controls into battle controls, then exercise
    # the native remapping inputs while capturing their help text.
    walk(
        d,
        [
            ("a", 150),
            ("down", 40),
            ("down", 40),
            ("down", 60),
            ("down", 60),
            ("right", 60),
            ("r", 60),
            ("l", 60),
            ("select", 60),
        ],
        "k",
    )


SCRIPTS["controller-pages"] = {
    "state": "roam",
    "run": capture_controller_pages,
    "description": "Field menu > Config > Controller: Area Map Key Config, then Down past the last row to Battle Key Config (014574), rows remapped in turn.",
}


def capture_forced_info_header(d: ScreenActions) -> None:
    """
    Display the Info initializer in its header window using the guarded ROM fixture.

    Args:
        d (ScreenActions): Driver at the roaming start with the Info-header verification patch active.
    """
    select_field_menu_option(d, 2)
    d.clear()
    d.press("a", 4, 90)
    d.capture("header", "Forced Info header in its native field-menu window")


SCRIPTS["forced-info-header"] = {
    "state": "roam",
    "run": capture_forced_info_header,
    "description": "Display the Info header through the native window renderer in a verification-only ROM.",
    "forced": "Window 9's list draw is replaced with window 3's Info draw in the emulator; the input ROM file is not modified.",
    # $82:AB94 normally selects window 9, which overwrites Info's glyph buffer.
    # Reopening window 3 retains the Info glyphs and its upper header placement.
    "rom_patches": [
        (
            0x12B94,
            bytes.fromhex("A9 09 22 4C C3 82"),
            bytes.fromhex("A9 03 22 4C C3 82"),
        )
    ],
    "targets": ["0144BF"],
}

for _name in ("info", "info-header-data-80", "info-header-data-c0"):
    SCRIPTS[_name]["initialization_only"] = {
        "0144BF": "Window 3 initializes Info's glyph buffer; window 9 overwrites it with Status/Job/Data before observed display."
    }

# These programs visit the same field-menu windows. Their literal entries must
# match inside those windows rather than borrowing a reused tile copy elsewhere.
for _program in SCRIPTS.values():
    _program["text_boxes"] = FIELD_MENU_TEXT_BOXES
