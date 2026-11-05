"""
This module invokes the game's text display routines for forced checks or replay in a previously calibrated context.

Forced tasks run supplied text pointers for rendering diagnostics. Direct, scene and map
tasks replay display contexts supplied by the caller. These helpers label their captures,
including partial output when playback raises, but do not read or validate calibration records.
The caller must establish native reachability before using calibrated replay results.
Result evaluation handles pixel verification and coverage credit separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from core.rom.convert_addresses import convert_file_offset_to_snes, convert_snes_to_file
from core.validation.validate_integers import is_integer
from verification.tasks import validate_requirements as requirements

if TYPE_CHECKING:
    from verification.game.control_gameplay import GameplayController


def _parse_hex(value: object, label: str, maximum: int) -> int:
    """
    Parse a bounded hexadecimal address supplied by a display task.

    Args:
        value (object): Nonempty hexadecimal string; Python's optional 0x prefix
            and surrounding whitespace are accepted. Negative values are rejected.
        label (str): Task-field name included in validation errors.
        maximum (int): Inclusive upper bound for the parsed address.

    Returns:
        int: Address from zero through maximum; no memory or files are changed.

    Raises:
        ValueError: If maximum is a boolean, noninteger or negative, or value is not
            a hexadecimal string within the bounds.
    """
    if not is_integer(maximum) or maximum < 0:
        raise ValueError("Hexadecimal address bounds must be nonnegative integers")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a hexadecimal string")
    try:
        address = int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a hexadecimal string") from error
    if not 0 <= address <= maximum:
        raise ValueError(f"{label} lies outside its supported address range")
    return address


def _read_entry_ids(value: object, label: str, rom_size: int) -> list[str]:
    """
    Validate translation-entry IDs used as text offsets and capture identifiers.

    Args:
        value (object): List of six-digit uppercase hexadecimal entry IDs.
        label (str): Task-field name included in validation errors.
        rom_size (int): Release-ROM byte count bounding the original text offsets.

    Returns:
        list[str]: New list retaining entry order and duplicates; the input is unchanged.

    Raises:
        ValueError: If rom_size is a boolean, noninteger or negative, or the container,
            ID spelling or ROM offset is invalid.
    """
    if not is_integer(rom_size) or rom_size < 0:
        raise ValueError("ROM size must be a nonnegative integer byte count")
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of entry IDs")
    result: list[str] = []
    for entry in cast(list[object], value):
        if (
            not isinstance(entry, str)
            or len(entry) != 6
            or any(character not in "0123456789ABCDEF" for character in entry)
        ):
            raise ValueError(
                f"{label} must contain six-digit uppercase hexadecimal IDs"
            )
        _parse_hex(entry, label, min(rom_size, 0x400000) - 1)
        result.append(entry)
    return result


def _validate_task(
    game: GameplayController, task: dict[str, Any], routes: tuple[str, ...]
) -> list[str]:
    """
    Check shared display-task fields before starting playback.

    Args:
        game (GameplayController): Controller supplying the release-ROM size.
        task (dict[str, Any]): Task with required route and targets fields plus
            setup/requirement fields understood by validate_definition(). Read unchanged.
        routes (tuple[str, ...]): Allowed route names for the calling display helper.

    Returns:
        list[str]: Validated target IDs in declared order.

    Raises:
        ValueError: If the route, target IDs, setup or requirement fields are invalid.
        KeyError: If route or targets is missing.
    """
    if not isinstance(cast(object, task), dict):
        raise ValueError("Display task must be an object")
    requirements.validate_definition(task)
    if task["route"] not in routes:
        raise ValueError(f"Display route must be one of {routes!r}")
    return _read_entry_ids(task["targets"], "targets", len(game.release))


def _classify_pages(
    pages: list[dict[str, Any]], forced: bool, description: str
) -> None:
    """
    Label captures produced by a display helper, including partial failed playback.

    A direct label describes how text was displayed; it does not certify calibration.
    Existing forced labels remain intact when a direct label is added.

    Args:
        pages (list[dict[str, Any]]): Newly captured page dictionaries, modified in place.
            The list itself is not changed.
        forced (bool): True for diagnostic displays; False for caller-calibrated replay.
        description (str): Display mechanism recorded with the selected category.
    """
    for page in pages:
        page["forced" if forced else "direct"] = description


def _check_targets(status: str, targets: list[str], pages: list[dict[str, Any]]) -> str:
    """
    Reject idle playback that did not capture every declared target entry.

    Args:
        status (str): Aggregated playback outcome; non-idle outcomes are preserved.
        targets (list[str]): Required entry IDs, already validated by the caller.
        pages (list[dict[str, Any]]): Captures from this display, read without mutation.

    Returns:
        str: Original status when non-idle or no targets are missing. Otherwise,
            targets_incomplete when some targets were observed or target_not_reached
            when none were observed. Page-level pixel verification remains the
            responsibility of result evaluation.
    """
    seen = {page.get("entry") for page in pages}
    if status == "idle" and set(targets) - seen:
        return "targets_incomplete" if set(targets) & seen else "target_not_reached"
    return status


def show_text(game: GameplayController, task: dict[str, Any]) -> str:
    """
    Display text using a forced diagnostic or calibrated direct-display task.

    Multiple texts share the initialized room; each generated event opens and closes its own
    dialogue window. Only captures made after room initialization receive forced or direct
    labels, including partial captures retained when a controller call raises. A direct label
    records the invocation mechanism; the caller must validate its calibration separately.

    Args:
        game (GameplayController): Initialized room controller; event scripts,
            text state and captured pages change. Existing captures are preserved.
        task (dict[str, Any]): Definition with route=forced or direct, quest_flags, and
            targets as six-digit uppercase hexadecimal entry IDs within the release ROM.
            Either text (one ID) or texts (a nonempty ID list) selects dialogue entries;
            otherwise block supplies a hexadecimal SNES ROM script address in a bank's
            upper address half, excluding WRAM banks 7E/7F. Optional stub
            supplies hexadecimal prefix bytes for a block display. Optional states maps
            entry IDs to WRAM hexadecimal addresses and integer byte values; booleans
            are rejected and integer subclasses accepted. The definition is not modified.

    Returns:
        str: Initialization failure or the first non-idle display outcome. If every display
            is idle, missing target IDs produce targets_incomplete or target_not_reached;
            otherwise idle is returned. Pixel verification is evaluated separately.

    Raises:
        ValueError: If task requirements, entry IDs, addresses, byte values or stub bytes
            are invalid, an event write exceeds WRAM, or a text chain is cyclic. Also
            raised if glyph positions, font offsets, frame dimensions or VRAM size are invalid.
        KeyError: If a required task field is missing or a text command is unsupported.
        IndexError: If text traversal reads beyond the ROM.
        PlaybackStopped: If the controller reaches a separately configured stop.
        RuntimeError: If page verification requires an unavailable or failed frame snapshot.
        AssertionError: If page verification needs a frame before one has been captured.
        OSError: If verification sources or screenshot files cannot be read or written.
    """
    targets = _validate_task(game, task, ("forced", "direct"))
    flags = task["quest_flags"]
    if "text" in task and "texts" in task:
        raise ValueError("A display must select text or texts, not both")
    texts = (
        _read_entry_ids(task["texts"], "texts", len(game.release))
        if "texts" in task
        else _read_entry_ids([task["text"]], "text", len(game.release))
        if "text" in task
        else []
    )
    if ("texts" in task or "text" in task) and not texts:
        raise ValueError("Text displays require at least one entry")
    if texts and "block" in task:
        raise ValueError("A display must select entries or a block, not both")
    states: Any = task.get("states", {})
    if not isinstance(states, dict):
        raise ValueError("states must map entry IDs to WRAM byte values")
    writes: dict[str, list[tuple[int, int]]] = {}
    for entry, state in cast(dict[str, Any], states).items():
        _read_entry_ids([entry], "states", len(game.release))
        if not isinstance(state, dict):
            raise ValueError("Each entry state must map WRAM addresses to byte values")
        writes[entry] = []
        for address, value in cast(dict[str, object], state).items():
            offset = _parse_hex(address, "state address", len(game.W) - 1)
            if not is_integer(value) or not 0 <= value <= 255:
                raise ValueError(
                    "State values must be integer bytes, excluding booleans"
                )
            writes[entry].append((offset, value))
    block = None if texts else _parse_hex(task["block"], "block", 0xFFFFFF)
    if block is not None and (
        block & 0xFFFF < 0x8000
        or block >> 16 in (0x7E, 0x7F)
        or convert_snes_to_file(block) >= len(game.release)
    ):
        raise ValueError("Display block must address the release ROM")
    stub = task.get("stub")
    if stub is not None and not isinstance(stub, str):
        raise ValueError("stub must be a hexadecimal byte string or None")
    prefix = bytes.fromhex(stub or "")
    if texts and prefix:
        raise ValueError("stub applies only to a block display")

    # Compile every text invocation before playback can apply partial setup writes.
    # 0E 01 00 opens the dialogue window; 18 F8 06 <address> FF displays a text
    # slot; 0D closes the window, 0B releases control and 00 ends the event.
    programs: list[tuple[str, bytes]] = []
    for text in texts:
        starts = (
            [
                start
                for _, start in sorted(
                    (page, start)
                    for start, (entry, page) in game.starts.items()
                    if entry == text
                )
            ]
            if game.english
            else []
        )
        show = bytearray()
        for slot in starts or [int(text, 16)]:
            if not is_integer(slot) or not 0 <= slot < len(game.release):
                raise ValueError("Compiled text start lies outside the release ROM")
            pointer = convert_file_offset_to_snes(slot)
            show += bytes.fromhex("18 F8 06") + pointer.to_bytes(3, "little") + b"\xff"
        programs.append(
            (text, bytes.fromhex("0E 01 00") + show + bytes.fromhex("0D 0B 00"))
        )

    prep = game.play_dialogue("initialization", 12000)
    if prep["status"] != "idle":
        return prep["status"]

    game.set_flags(flags)
    before = len(game.pages)
    status = "idle"
    try:
        if programs:
            for text, program in programs:
                for address, value in writes.get(text, []):
                    game.poke(address, [value])
                game.start_script(program)
                outcome = game.play_dialogue(
                    "forced",
                    12000,
                    30 if task["route"] == "direct" else 1200,
                    plan=[0] * 4,
                )["status"]
                if status == "idle":
                    status = outcome
        else:
            assert block is not None
            if prefix:
                # Event opcode 43 transfers control to the supplied three-byte script address.
                game.start_script(prefix + b"\x43" + block.to_bytes(3, "little"))
            else:
                # Actor slot 2 stores its event pointer in parallel WRAM byte arrays.
                for address, value in zip(
                    (0xB297, 0xB2BC, 0xB2E1), block.to_bytes(3, "little")
                ):
                    game.poke(address + 2, [value])
                # Clear this actor's script wait and control byte before resuming it.
                game.poke(0xB712 + 2, [0])
                game.poke(0xB1B9 + 2, [0])
            status = game.play_dialogue("forced", 12000, plan=[0] * 4)["status"]
    finally:
        # Even playback failures must retain diagnostic classification on partial captures.
        _classify_pages(
            game.pages[before:],
            task["route"] == "forced",
            "text invoked by a generated display event",
        )
    return _check_targets(status, targets, game.pages[before:])


def show_scene(game: GameplayController, task: dict[str, Any]) -> str:
    """
    Display calibrated scene text using its own screen setup.

    The helper starts at the command that prepares the scene's text screen and shortens waits
    outside text. It retains only requested entries from this invocation, preserves earlier
    captures, and labels retained pages as direct replay even if playback raises. Calibration
    must already have been validated by the caller.

    Args:
        game (GameplayController): Controller used to start the scene;
            skip_waits and targets_left settings are restored afterward. Emulator state
            and new captures remain changed; earlier capture dictionaries are untouched.
        task (dict[str, Any]): Definition with route=scene and start as a hexadecimal
            SNES ROM address in a bank's upper address half, excluding WRAM banks 7E/7F.
            targets is a list of six-digit uppercase hexadecimal entry IDs within
            the release ROM. The definition is not modified.

    Returns:
        str: Initialization or scene-playback outcome, adjusted to targets_incomplete or
            target_not_reached if idle playback omits required entries. Pixel verification
            remains separate from execution status.

    Raises:
        ValueError: If task requirements, start address or target IDs are invalid, a
            scene-memory write exceeds WRAM, or a text chain is cyclic. Also raised if
            glyph positions, font offsets, frame dimensions or VRAM size are invalid.
        KeyError: If a required task field is missing or a text command is unsupported.
        IndexError: If text traversal reads beyond the ROM.
        PlaybackStopped: If the controller reaches a separately configured stop.
        RuntimeError: If page verification requires an unavailable or failed frame snapshot.
        AssertionError: If page verification needs a frame before one has been captured.
        OSError: If verification sources or screenshot files cannot be read or written.
    """
    targets = _validate_task(game, task, ("scene",))
    start = _parse_hex(task["start"], "start", 0xFFFFFF)
    if (
        start & 0xFFFF < 0x8000
        or start >> 16 in (0x7E, 0x7F)
        or convert_snes_to_file(start) >= len(game.release)
    ):
        raise ValueError("Scene start must address the release ROM")
    prep = game.play_dialogue("initialization", 12000)
    if prep["status"] != "idle":
        return prep["status"]

    skip, previous_targets = game.skip_waits, game.targets_left
    previous_pages = list(game.pages)
    before = len(previous_pages)

    # Done once every page of the listed lines is checked (the scene may go on long after them).
    targets_left: set[tuple[str, int] | str] = (
        {(entry, page) for entry, page in game.starts.values() if entry in targets}
        if game.english
        else set(targets)
    )
    game.skip_waits, game.targets_left = True, targets_left
    try:
        # Carrier actor 2 runs the scene setup through the native cutscene handler.
        game.start_cutscene(2, start)
        status = game.play_dialogue("forced", 20000, 240, plan=[0] * 4)["status"]
    finally:
        game.skip_waits, game.targets_left = skip, previous_targets
        # Preserve earlier captures unchanged and classify this scene's partial output on errors.
        pages = [page for page in game.pages[before:] if page.get("entry") in targets]
        _classify_pages(pages, False, f"scene script started at ${start:06X}")
        game.pages = previous_pages + pages
    return _check_targets(status, targets, pages)


def show_map_labels(game: GameplayController, task: dict[str, Any]) -> str:
    """
    Display calibrated city-map labels on their native map screen.

    The helper saves a settled city map prepared by the caller. Each label runs its native
    show-command script on a newly spawned actor. Only that label's captures are retained
    from each invocation; captures preceding this call are preserved. Calibration must
    already have been validated by the caller.

    Args:
        game (GameplayController): Controller on the city map; saves/restores
            its emulator state between labels. Collected pages are added to existing
            captures, including partial output when playback raises. The final label's
            emulator state is retained rather than rolled back after the call.
        task (dict[str, Any]): Definition with route=map, targets and texts as lists of
            six-digit uppercase hexadecimal entry IDs within the release ROM. Each
            text offset must immediately follow its native 19 map-label opcode. Empty texts executes
            no playback. The definition is not modified.

    Returns:
        str: First non-idle label outcome; otherwise idle, targets_incomplete or
            target_not_reached according to the newly captured target IDs. Empty texts
            succeeds only if no targets are required. Pixel verification is separate.

    Raises:
        ValueError: If task requirements or IDs are invalid, a text does not follow its
            native 19 opcode, a script write exceeds WRAM, or a text chain is cyclic.
            Also raised if glyph positions, font offsets, frame dimensions or VRAM size are invalid.
        KeyError: If a required task field is missing or a text command is unsupported.
        IndexError: If text traversal reads beyond the ROM.
        StopIteration: If there is no free actor slot for a label script.
        PlaybackStopped: If the controller reaches a separately configured stop.
        RuntimeError: If the emulator cannot save/restore state or a required frame snapshot
            is unavailable or failed.
        AssertionError: If page verification needs a frame before one has been captured.
        OSError: If verification sources or screenshot files cannot be read or written.
    """
    targets = _validate_task(game, task, ("map",))
    texts = _read_entry_ids(task["texts"], "texts", len(game.release))
    if any(
        int(text, 16) == 0 or game.release[int(text, 16) - 1] != 0x19 for text in texts
    ):
        raise ValueError("Map text must follow its native 19 show-text opcode")
    if not texts:
        return _check_targets("idle", targets, [])
    game.run(300)  # Allow the caller-prepared map screen to settle before saving it.

    # Restore the same settled map state for each label, preventing an earlier
    # label's inputs or window state from affecting the next capture.
    settled, status = game.emu.save_state(), "idle"
    previous_pages = list(game.pages)
    pages: list[dict[str, Any]] = []

    for text in texts:
        game.emu.load_state(settled)
        game.pages = []
        try:
            game.clear_log()
            # The entry offset follows its native show-text opcode by one byte.
            start = convert_file_offset_to_snes(int(text, 16) - 1)
            # Label scripts use the city map's actor 8 as their spawn parent.
            game.spawn_script(start, 8)
            outcome = game.play_dialogue("map", 2000, 30, plan=[0] * 4)["status"]
            if status == "idle":
                status = outcome
        finally:
            captured = [page for page in game.pages if page.get("entry") == text]
            _classify_pages(captured, False, "label displayed by its city-map script")
            pages.extend(captured)
            game.pages = previous_pages + pages
    return _check_targets(status, targets, pages)
