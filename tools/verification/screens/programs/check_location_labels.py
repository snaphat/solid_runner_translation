"""
This module registers screen programs that load specified rooms or scenes and capture location banners and mission-result text.

Banner definitions map native stage/room pairs to expected text IDs. Factory functions create
callbacks that request those rooms and sample their text at configured intervals. Mission and
special-scene programs use the same screen driver, which applies the fixture writes and performs
the pixel validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from core.validation.validate_integers import is_integer

if TYPE_CHECKING:
    from verification.screens.register_screen_programs import ScreenProgram
    from verification.screens.run_screen_checks import ScreenActions

from verification.screens.programs.controller_sequences import request_room_load

SCRIPTS: dict[str, ScreenProgram] = {}


# Decimal stage/variant room keys -> text IDs (hex unheadered ROM file offsets).
# Stage/room keys map to the text-entry IDs expected from their native
# location banner. These are not screen coordinates.
BANNER_ROOMS = {
    "6/1": ["01B4B2"],
    "6/6": ["01B97F"],
    "6/17": ["01B89E"],
    "6/26": ["01B550"],
    "6/38": ["01B562"],
    "6/52": ["01B574"],
    "8/35": ["01B4C2"],
    "8/36": ["01B4D2"],
    "8/37": ["01B4E2"],
    "8/40": ["01B4F2"],
    "9/4": ["01B502"],
    "11/1": ["01B514"],
    "11/10": ["01B523"],
    "11/18": ["01B532"],
    "11/30": ["01B541"],
    "12/3": ["01B586"],
    "12/7": ["01B596"],
    "12/10": ["01B98F"],
    "12/13": ["01B99F"],
    "13/1": ["01B8BE"],
    "13/3": ["01B8CD"],
    "13/4": ["01B8AF"],
    "13/8": ["01B5A6"],
    "13/22": ["01B5B7"],
    "13/30": ["01B5C8"],
    "14/1": ["01B5EA"],
    "14/3": ["01B5D9"],
    "15/1": ["01B61F"],
    "15/18": ["01B631"],
    "15/33": ["01B643"],
    "15/51": ["01B60D"],
    "15/61": ["01B5FB"],
    "17/8": ["01B655"],
    "18/2": ["01B687"],
    "18/20": ["01B698"],
    "18/48": ["01B6A8"],
    "18/63": ["01B6B8"],
    "18/75": ["01B6C8"],
    "18/92": ["01B6D8"],
    "18/119": ["01B6E8"],
    "20/0": ["01B70C"],
    "20/9": ["01B71D"],
    "20/14": ["01B72E"],
    "20/42": ["01B73E"],
    "20/60": ["01B74E"],
    "20/78": ["01B75E"],
    "20/91": ["01B76E"],
    "20/110": ["01B77E"],
    "20/123": ["01B78E"],
    "21/1": ["01B676"],
    "21/4": ["01B665"],
    "25/6": ["01B80E"],
    "25/7": ["01B820"],
    "25/8": ["01B832"],
    "25/9": ["01B844"],
    "25/81": ["01B856"],
    "25/97": ["01B868"],
    "25/120": ["01B87A"],
    "25/134": ["01B88C"],
    "26/45": ["01B7C2"],
    "26/46": ["01B7D5"],
    "26/47": ["01B7E8"],
    "26/48": ["01B7FB"],
    "27/25": ["01B79E"],
    "27/45": ["01B7B0"],
    "30/6": ["01B920"],
    "30/14": ["01B933"],
    "30/26": ["01B959"],
    "30/33": ["01B946"],
    "31/0": ["01B8DC"],
    "31/4": ["01B8EE"],
    "31/10": ["01B8FE"],
    "31/16": ["01B90F"],
}


# Location banners made reachable by event patches, using the same room/ID format.
REPAIR_BANNER_ROOMS = {"13/45": ["01B96C"], "14/12": ["01B6F8"]}


# Per-room capture counts for banners appearing after a longer entrance sequence.
# Values are sample counts passed to create_banner_capture(), not story chapters.
LATE_BANNERS = {"14/12": 20}


def create_banner_capture(
    stage: int, variant: int, captures: int = 4
) -> Callable[[ScreenActions], None]:
    """
    Create a location-banner capture callback.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        stage (int): Native stage number placed in WRAM 0x1AA8.
        variant (int): Room variant placed in WRAM 0x1AA9.
        captures (int, optional): Number of timed banner captures after
            requesting the room load.
            Defaults to 4.

    Returns:
        Callable[[ScreenActions], None]: Callback that loads the configured room
            and samples its transient location banner.

    Raises:
        ValueError: If stage or variant is outside 0 through 255, or captures is not
            positive.
            Also raised if any integer argument is a boolean or noninteger.
    """
    if not is_integer(stage) or not is_integer(variant) or not is_integer(captures):
        raise ValueError("stage, variant, captures must be integers, not booleans")

    if not 0 <= stage <= 255 or not 0 <= variant <= 255 or captures <= 0:
        raise ValueError(
            "Banner capture requires a two-byte room and a positive capture count"
        )

    def run(d: ScreenActions) -> None:
        """
        Load the configured room and sample its location banner.

        Args:
            d (ScreenActions): Driver whose emulator receives room-load fixtures
                and records timed banner captures.
        """
        request_room_load(
            d, stage, variant, f"Room-load fixture: stage {stage} variant {variant}."
        )
        d.clear()

        # Allow the room loader to finish before the first banner sample, then
        # sample at shorter intervals while its transient label remains visible.
        for k in range(captures):
            d.run(150 if k == 0 else 50)
            d.capture(f"b{k}", f"room {stage}/{variant} banner, frame {153 + 53 * k}")

    return run


def _collect_banner_rooms() -> dict[str, list[str]]:
    """
    Combine ordinary and repaired location-banner definitions.

    Returns:
        dict[str, list[str]]: Native stage/room keys mapped to the expected
            translation entry IDs for their banner captures.
    """
    return {**BANNER_ROOMS, **REPAIR_BANNER_ROOMS}


for _room, _ids in _collect_banner_rooms().items():
    _s, _v = map(int, _room.split("/"))
    SCRIPTS[f"banner-{_s:02d}-{_v:03d}"] = {
        "state": "roam",
        "run": create_banner_capture(_s, _v, LATE_BANNERS.get(_room, 4)),
        "description": f"Location banner on loading room {_room} ({', '.join(_ids)}).",
    }


def capture_mission_results(d: ScreenActions) -> None:
    """
    Capture result labels after a native job report.

    The reward and total-funds labels draw separately; captures continue across both draws. Wait
    up to 300 ten-frame polling steps for the Reward label, periodically pressing A to advance
    preceding dialogue. If it never appears, return without a results capture; the caller grades
    missing target evidence.

    Args:
        d (ScreenActions): Driver restored to the mission-18 start.
    """
    d.clear()

    for k in range(300):
        d.run(10)
        traced = {i for r in d.read_records() for i in r["ids"]}

        # Entry 091611 is the Reward label. Its draw marks the results screen
        # rather than the preceding dialogue, so start the dense capture sequence here.
        if "091611" in traced:
            for j in range(30):
                elapsed = 3 + 23 * j + (4 if j >= 15 else 0)
                d.capture(f"r-{j:02d}", f"results screen, +{elapsed} frames")
                d.run(20)

                if j == 14:
                    d.run(4, ("a",))

            return

        if k % 6 == 5:
            d.run(4, ("a",))


SCRIPTS["mission-results-18"] = {
    "state": "mission-18",
    "run": capture_mission_results,
    "description": "Job ending of context-dialogue-native-victory-18 (native report), Results screen labels.",
}


def create_mission_capture(index: int) -> Callable[[ScreenActions], None]:
    """
    Create a mission-complete room callback for one result index.

    Uses native room 23/7 and the mission index in SRAM $70:013E.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        index (int): Mission-result index written to SRAM 0x13E.

    Returns:
        Callable[[ScreenActions], None]: Callback that loads room 23/7 with the
            selected mission index and captures its results screen.

    Raises:
        ValueError: If the mission index is outside 0 through 16.
            Also raised if index is a boolean or noninteger.
    """
    if not is_integer(index):
        raise ValueError("index must be an integer, not a boolean")

    if not 0 <= index < 17:
        raise ValueError("Mission index must be between 0 and 16")

    def run(d: ScreenActions) -> None:
        """
        Load the configured mission result and capture its report.

        Args:
            d (ScreenActions): Driver receiving room and mission-index fixtures;
                captures accumulate on its runner.
        """
        request_room_load(
            d, 23, 7, "Room-load fixture: the mission-complete room 23/7."
        )

        # The common results room selects its report through SRAM $013E.
        # Set the index before advancing frames so the native loader composes that report.
        d.poke(0x13E, [index], 0, f"Mission index fixture: $70:013E = {index}.")
        d.clear()

        for k, wait in enumerate((450, 100, 100, 100, 100)):
            d.run(wait)
            d.capture(f"m{k}", f"mission {index} screen, frame {453 + 103 * k}")

    return run


def create_special_scene_capture(variant: int) -> Callable[[ScreenActions], None]:
    """
    Create a stage-23 callback for one special-scene variant.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        variant (int): Special room variant in stage 23, displayed with scene
            index zero.

    Returns:
        Callable[[ScreenActions], None]: Callback that loads the selected stage-23
            variant with scene index zero and captures its text.

    Raises:
        ValueError: If the special-scene variant is outside 0 through 8.
            Also raised if variant is a boolean or noninteger.
    """
    if not is_integer(variant):
        raise ValueError("variant must be an integer, not a boolean")

    if not 0 <= variant < 9:
        raise ValueError("Special-scene variant must be between 0 and 8")

    def run(d: ScreenActions) -> None:
        """
        Load the configured stage-23 scene and capture its text.

        Args:
            d (ScreenActions): Driver receiving the special-scene room fixture and
                recording the resulting frames.
        """
        request_room_load(
            d, 23, variant, f"Room-load fixture: special scene 23/{variant}."
        )

        # Special-scene rooms reuse the scene-index byte. Clear it so the
        # requested room variant is tested independently of a previous mission result.
        d.poke(0x13E, [0], 0, "Scene index fixture: $70:013E = 0.")
        d.clear()

        for k, wait in enumerate((450, 100, 100, 100, 100)):
            d.run(wait)
            d.capture(f"s{k}", f"special scene 23/{variant}, frame {453 + 103 * k}")

    return run


for _index in range(17):
    SCRIPTS[f"mission-{_index:02d}"] = {
        "state": "roam",
        "run": create_mission_capture(_index),
        "description": f"Mission-complete screen {_index} (room 23/7).",
    }


for _variant in range(9):
    SCRIPTS[f"special-23-{_variant}"] = {
        "state": "roam",
        "run": create_special_scene_capture(_variant),
        "description": f"Special scene room 23/{_variant}.",
    }
