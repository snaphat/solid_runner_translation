"""
This module combines screen-check callbacks into the SCRIPTS registry and defines the state and fixture fields each program supplies.

SCRIPTS maps unique names to dictionaries with state (a named native start), run (callable
accepting ScreenActions), and description. fixtures describes memory writes and
configured peripheral behavior used to prepare each screen.
turbo_stub, turbo_statuses and turbo_file opt into the replacement peripheral driver, which is
labelled separately in results.

Callbacks use run/press/clear/read_records/capture and declared fixture writes; the screen
runner loads the selected start and evaluates its captures. Worker execution closes the emulator
and adds ROM and translation-source hashes before saving the results. Importing the registry
creates definitions, not an emulator. Individual check callbacks are program implementations,
not the reusable screen-control API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NotRequired, TypedDict

from verification.screens.programs import check_location_labels as location_labels
from verification.screens.programs import check_menus as menus
from verification.screens.programs import check_saving_and_naming as save_naming
from verification.screens.programs import check_shops as shops
from verification.screens.programs import check_terminal as terminal
from verification.screens.programs import check_turbo_file as turbo_file

if TYPE_CHECKING:
    from verification.screens.programs.controller_sequences import (
        MemoryFixture,
        QuestFlagFixture,
    )
    from verification.screens.run_screen_checks import ScreenActions


class ScreenProgram(TypedDict):
    """
    Native start, playback callback and fixture metadata for a screen check.

    Attributes:
        state (str): Named starting-state definition restored before the callback runs.
        run (Callable[[ScreenActions], None]): Controller program that captures
            the intended screens through the supplied driver.
        description (str): Human-readable path and purpose of this program.
        fixtures (list[MemoryFixture | QuestFlagFixture | str]): Optional setup-write
            or peripheral-fixture descriptions stored in the program definition;
            these are not copied into individual capture records.
        text_boxes (dict[str, tuple[int, int, int, int]]): Optional entry-specific
            output rectangles, (left, top, right, bottom), in native screen pixels.
            Matching copies outside the specified rectangle cannot satisfy a capture.
        turbo_stub (str): Optional device-fixture label selecting the stand-in driver.
        turbo_statuses (dict[int, int]): Optional device-call IDs mapped to status bytes.
        turbo_file (bool): Whether the stand-in holds a native slot-one save file.
        forced (str): Optional explanation of a forced display. All captures from
            that program are diagnostic and cannot provide native coverage.
        rom_patches (list[tuple[int, bytes, bytes]]): Optional ROM edits used only for verification:
            file offset, required original bytes and equal-length replacement.
            Requires forced attribution; the ordinary image is restored afterward.
        targets (list[str]): Optional entry IDs that this program must verify.
        initialization_only (dict[str, str]): Entries whose observed draws only
            initialize a buffer that is overwritten before display. Failed captures
            retain the reason but receive no display credit; verified captures remain eligible.
    """

    state: str
    run: Callable[[ScreenActions], None]
    description: str
    fixtures: NotRequired[list[MemoryFixture | QuestFlagFixture | str]]
    text_boxes: NotRequired[dict[str, tuple[int, int, int, int]]]
    turbo_stub: NotRequired[str]
    turbo_statuses: NotRequired[dict[int, int]]
    turbo_file: NotRequired[bool]
    forced: NotRequired[str]
    rom_patches: NotRequired[list[tuple[int, bytes, bytes]]]
    targets: NotRequired[list[str]]
    initialization_only: NotRequired[dict[str, str]]


# Program modules reference ScreenProgram only during type checking. Importing
# them does not require this module to finish constructing the registry.
SCRIPTS: dict[str, ScreenProgram] = {}
for _family in (menus, save_naming, location_labels, shops, terminal, turbo_file):
    _duplicates = SCRIPTS.keys() & _family.SCRIPTS.keys()
    if _duplicates:
        raise ValueError(f"Duplicate screen program names: {sorted(_duplicates)}")

    SCRIPTS.update(_family.SCRIPTS)
