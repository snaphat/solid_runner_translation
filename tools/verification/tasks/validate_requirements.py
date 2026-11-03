"""
This module checks task definitions and determines which entry IDs and pages each task must reach or verify.

Task keys and entry IDs are strings; page numbers are positive, one-based integers.
target/targets declare required entries. expect records entry/page pairs. conditional_expect
explains optional observations without weakening a declared target. original_lacks or
original_absent_targets plus a reason classifies content absent from the Japanese ROM.

These functions validate or derive requirements without modifying definitions.
resolve_required_pages() explicitly loads the compiled English ROM; other functions use their
supplied dictionaries and compiled-page mapping.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence, cast

from core.validation.validate_integers import is_integer
from emulator.validate_controller_input import BTN

# Maximum exposed sizes of the supported libretro regions: SRAM (expanded by
# instrumentation), WRAM and VRAM. A core may expose less SRAM; runtime writes
# still check its actual capacity. Preflight rejects spans beyond these bounds.
MEMORY_REGION_CAPACITIES = {0: 0x20000, 2: 0x20000, 3: 0x10000}


def resolve_task_requirements(definition: dict[str, Any]) -> dict[str, Any]:
    """
    This function copies the fields that define a task's required text and verification category.

    Quick replay can shorten a route's setup and actions without dropping its expected entries,
    pages, or explanations of Japanese-ROM differences.

    Args:
        definition (dict[str, Any]): Task supplying key/route, target(s), expect,
            conditional expectations, original-ROM differences, and forced/peripheral
            classification. Operational setup fields are not copied.

    Returns:
        dict[str, Any]: Deep copy of requirement fields present in the supplied
            definition.
    """
    fields = (
        "key",
        "route",
        "target",
        "targets",
        "expect",
        "conditional_expect",
        "original_lacks",
        "original_absent_targets",
        "original_absent_reason",
        "forced",
        "peripheral",
    )
    return {name: deepcopy(definition[name]) for name in fields if name in definition}


def collect_declared_targets(definition: dict[str, Any]) -> set[str]:
    """
    Return the union of target and targets as a set of entry IDs.

    Args:
        definition (dict[str, Any]): Task containing an optional singular target ID and
            targets list; absent fields contribute no IDs.

    Returns:
        set[str]: Distinct entry IDs declared by target or targets.
    """
    return set(definition.get("targets") or []) | (
        {definition["target"]} if definition.get("target") else set()
    )


def validate_conditional_entries(definition: dict[str, Any]) -> set[str]:
    """
    Validate conditional-expectation explanations and return their IDs.

    Args:
        definition (dict[str, Any]): Task containing conditional_expect explanations and
            declared targets.

    Returns:
        set[str]: Entry IDs used as conditional_expect keys.

    Raises:
        ValueError: If conditional_expect is not an entry-to-explanation mapping, an
            explanation is blank, or a declared target is conditional.
    """
    observations: Any = definition.get("conditional_expect", {})
    if not isinstance(observations, dict) or any(
        not isinstance(entry, str) or not entry or not isinstance(reason, str) or not reason.strip()
        for entry, reason in cast(dict[Any, Any], observations).items()
    ):
        raise ValueError("conditional_expect must map entry IDs to explanations")

    # Conditional entries explain optional observations; they cannot excuse
    # a target that the task explicitly promises to reach.
    if set(cast(dict[str, str], observations)) & collect_declared_targets(definition):
        raise ValueError("A declared target cannot be made conditional")

    return set(cast(dict[str, str], observations))


def collect_required_entries(definition: dict[str, Any]) -> set[str]:
    """
    Return declared targets plus nonconditional expect entry IDs.

    Args:
        definition (dict[str, Any]): Task containing target(s), expect entry/page pairs and
            optional conditional_expect keys; conditional expectations are excluded.

    Returns:
        set[str]: Declared target IDs plus expected entry IDs not explicitly classified as
            optional observations.

    Raises:
        ValueError: If conditional expectations are malformed or contradict the declared
            targets.
    """
    return collect_declared_targets(definition) | (
        {e for e, _ in definition.get("expect", [])} - validate_conditional_entries(definition)
    )


def resolve_expected_pages(
    definition: dict[str, Any], required: dict[str, set[int]]
) -> set[tuple[str, int]]:
    """
    Resolve required pages using current-ROM pagination first.

    Args:
        definition (dict[str, Any]): Task whose target and nonconditional expect entries
            require page coverage.
        required (dict[str, set[int]]): Current-ROM required page numbers keyed
            by entry ID; pagination takes precedence over saved expectations.

    Returns:
        set[tuple[str, int]]: Required (entry ID, one-based page number) pairs.
            Use required when its key exists, even for an empty page set. Otherwise
            use the entry's stored expect pages, falling back to page one.

    Raises:
        ValueError: If conditional expectations are malformed or contradict a target,
            or an expected/required page number is a boolean, noninteger or not positive.
    """
    # Editing a translation may add pages. Stored expectations must not let
    # playback stop before every page in the current compiled entry is checked.
    for pages in required.values():
        if any(not is_integer(page) or page < 1 for page in pages):
            raise ValueError("Required page numbers must be positive integers, not booleans")
    for _, page in definition.get("expect", []):
        if not is_integer(page) or page < 1:
            raise ValueError("Expected page numbers must be positive integers, not booleans")
    old: dict[str, set[int]] = {}
    for entry, page in definition.get("expect", []):
        old.setdefault(entry, set()).add(page)

    # Use current compiled pagination when available. Stored expect pairs are
    # a fallback for entries absent from the supplied page index.
    return {
        (entry, page)
        for entry in collect_required_entries(definition)
        for page in required.get(entry, old.get(entry, {1}))
    }


def collect_original_absences(definition: dict[str, Any]) -> set[str]:
    """
    Return justified original-ROM target absences.

    Args:
        definition (dict[str, Any]): Task containing declared target(s) and optional
            original-ROM absence fields. original_lacks and original_absent_reason,
            when present, must be nonblank explanation strings. Explicit absence
            targets must be a list of nonempty entry IDs with a reason.

    Returns:
        set[str]: All declared targets for original_lacks, otherwise the
            explained original_absent_targets.

    Raises:
        ValueError: If an absence explanation is not a nonblank string, the explicit
            absence list has an invalid type or entry, or an explicit absence lacks
            a reason or names an undeclared target.
    """
    # This helper is also called directly during evaluation, so explanations
    # must be checked here rather than relying on task preflight alone.
    for name in ("original_lacks", "original_absent_reason"):
        if name in definition and (
            not isinstance(definition[name], str) or not definition[name].strip()
        ):
            raise ValueError(f"{name} must explain the original-ROM difference")

    raw_absent: Any = definition.get("original_absent_targets", [])
    if not isinstance(raw_absent, list) or any(
        not isinstance(entry, str) or not entry for entry in cast(list[Any], raw_absent)
    ):
        raise ValueError("original_absent_targets must be a list of entry IDs")
    absent = set(cast(list[str], raw_absent))
    if absent and (
        "original_absent_reason" not in definition
        or not absent <= collect_declared_targets(definition)
    ):
        raise ValueError("Original target differences need a reason and must name declared targets")

    return collect_declared_targets(definition) if definition.get("original_lacks") else absent


def validate_definition(definition: dict[str, Any]) -> None:
    """
    This function checks a replay task's expected text and supported game-setup fields before playback.

    Declared targets must agree with expected pages and explanations of optional text or
    Japanese-ROM differences. Setup checks cover room and actor values, random-number state,
    controller actions, frame limits, inventory, quest flags, and nested prerequisite tasks in
    before.

    Numeric fields accept integer subclasses but reject booleans. Boolean flags require bool
    values; forced also accepts an explanation string. A supplied field_trigger must be a
    dictionary. Its optional native_record_type must be an integer from 0 through 255,
    excluding booleans; its optional handler_checks_a_button must be a boolean.

    Route-specific fields not handled here are not validated. The task key is used only in error
    messages; index_definitions() checks that keys are nonempty and unique.

    Args:
        definition (dict[str, Any]): Task schema to validate before replay or replacing
            stored evidence.

    Raises:
        ValueError: If targets, expected pages, conditional explanations or original-ROM
            differences have invalid types or contradict each other, or a supported
            setup field has an invalid type or value.
    """
    # These dictionaries come from JSON. Check container and element types
    # before casts used for iteration; casts do not perform runtime validation.
    key = definition.get("key", "<unnamed>")
    # Operational fields must be checked before a rerun revokes saved evidence,
    # rather than waiting for a malformed task to reach a native memory write.
    for name, maximum in (
        ("stage", 255),
        ("variant", 255),
        ("rng_seed", 255),
        ("room_rng_seed", 255),
        ("actor", 36),
    ):
        if name in definition and (
            (not is_integer(definition[name])) or not 0 <= definition[name] <= maximum
        ):
            raise ValueError(f"{key}: {name} must be an integer from 0 to {maximum}")
    if "frame_budget" in definition and (
        (not is_integer(definition["frame_budget"])) or definition["frame_budget"] <= 0
    ):
        raise ValueError(f"{key}: frame_budget must be a positive integer")

    # Counts feed range() directly; booleans must not become one or zero repeats.
    for name in ("field_repeat", "talk_repeat", "preparatory_talk_limit"):
        if name in definition:
            value = definition[name]
            if not is_integer(value) or value < 0:
                raise ValueError(f"{key}: {name} must be a nonnegative integer")

    # Pickup stock is capped at 80. Enemy HP occupies an unsigned 16-bit word;
    # None and zero leave a dialogue-controller override inactive. Story-battle
    # setup also requires positive HP when applying it; each battle path checks
    # the opponent-specific maximum after native initialization.
    for name, maximum in (
        ("pickup_stock_fixture", 80),
        ("native_enemy_health_fixture", 0xFFFF),
        ("story_enemy_hp", 0xFFFF),
    ):
        if name not in definition:
            continue
        value = definition[name]
        if value is None and name != "pickup_stock_fixture":
            continue
        if not is_integer(value) or not 0 <= value <= maximum:
            raise ValueError(f"{key}: {name} must be an integer from 0 to {maximum}")

    # Wallet amounts are limited by the game's six-digit display. Initial
    # enemy HP must be positive; phase HP also permits zero to disable it.
    for name, minimum, maximum in (
        ("wallet_fixture", 0, 999999),
        ("enemy_health_fixture", 1, 0xFFFF),
        ("phase_enemy_hp", 0, 0xFFFF),
    ):
        if name not in definition:
            continue
        value = definition[name]
        if value is None and name != "wallet_fixture":
            continue
        if not is_integer(value) or not minimum <= value <= maximum:
            raise ValueError(f"{key}: {name} must be an integer from {minimum} to {maximum}")

    if "walk_until_text" in definition:
        walk = definition["walk_until_text"]
        if not isinstance(walk, dict):
            raise ValueError(f"{key}: walk_until_text must be an input-step object")
        # Reuse the input-step checks below for walking frames and button names.
        validate_definition({"key": key, "inputs": [{
            "frames": cast(dict[str, Any], walk).get("frames"),
            "buttons": cast(dict[str, Any], walk).get("buttons"),
        }]})

    # Native addresses, masks, actor indices and menu indices have fixed-width
    # fields. A progression level is stored in a byte; the experience-table
    # lookup also requires it to be at least one.
    for name, minimum, maximum in (
        ("level", 1, 255),
        ("phase_sram", 0, MEMORY_REGION_CAPACITIES[0] - 1),
        ("phase_mask", 0, 255),
        ("class_index", 0, 2),
        ("parent", 0, 36),
    ):
        if name in definition and (
            not is_integer(definition[name]) or not minimum <= definition[name] <= maximum
        ):
            raise ValueError(f"{key}: {name} must be an integer from {minimum} to {maximum}")
    for name in ("field_start_distance", "field_move_frames"):
        if name in definition and (not is_integer(definition[name]) or definition[name] < 0):
            raise ValueError(f"{key}: {name} must be a nonnegative integer")

    # Validate coordinate components before offsets turn a boolean into an int.
    # Signed positions and offsets are intentional; relocation clamps negatives
    # and checks the final native coordinate fields after combining them.
    for name in ("position", "position_offset", "field_position", "native_field_prelude"):
        if name not in definition:
            continue
        pair = definition[name]
        if not isinstance(pair, (list, tuple)) or len(cast(Sequence[object], pair)) != 2:
            raise ValueError(f"{key}: {name} must contain two integer coordinates")
        if any(not is_integer(value) for value in cast(Sequence[object], pair)):
            raise ValueError(f"{key}: {name} must contain integers, not booleans")
        if name == "native_field_prelude" and any(
            not 0 <= value <= 255 for value in cast(Sequence[int], pair)
        ):
            raise ValueError(f"{key}: native_field_prelude must contain two bytes")

    for name in ("choice_plan", "preparatory_choice_plan", "plan"):
        if name not in definition or definition[name] is None:
            continue
        plan = definition[name]
        if not isinstance(plan, (list, tuple)) or any(
            value != "cancel" and (not is_integer(value) or not 0 <= value <= 3)
            for value in cast(Sequence[object], plan)
        ):
            raise ValueError(f"{key}: {name} must contain menu indices 0-3 or cancel")

    if "native_maze_goals" in definition:
        goals = definition["native_maze_goals"]
        if not isinstance(goals, (list, tuple)) or any(
            not is_integer(value) or not 0xA0 <= value <= 0xF5 or value % 16 >= 6
            for value in cast(Sequence[object], goals)
        ):
            raise ValueError(f"{key}: native_maze_goals must contain native maze addresses")

    if "inventory_fixture" in definition:
        inventory = definition["inventory_fixture"]
        if not isinstance(inventory, list):
            raise ValueError(f"{key}: inventory_fixture must be a list")
        for stock in cast(list[object], inventory):
            if not isinstance(stock, dict):
                raise ValueError(f"{key}: inventory entries must contain item and quantity")
            record = cast(dict[str, object], stock)
            for name, maximum in (("item", 97), ("quantity", 80)):
                value = record.get(name)
                if not is_integer(value) or not 0 <= value <= maximum:
                    raise ValueError(f"{key}: inventory {name} must be an integer 0-{maximum}")

    # Direct-display state dictionaries supply raw WRAM bytes. Battle tasks
    # instead use a list of serialized states, which has no numeric components.
    if isinstance(definition.get("states"), dict):
        for state in cast(dict[str, object], definition["states"]).values():
            if not isinstance(state, dict) or any(
                not is_integer(value) or not 0 <= value <= 255
                for value in cast(dict[str, object], state).values()
            ):
                raise ValueError(f"{key}: direct-display state values must be integer bytes")

    for name in (
        "peripheral",
        "chained",
        "continue_state",
        "consumable_fixture",
        "defeat_fixture",
        "full_inventory",
        "lose_fixture",
        "position_feet",
        "reapply_quest_flags",
        "talk_early",
        "wait_for_player_control",
        "wait_full",
    ):
        if name in definition and not isinstance(definition[name], bool):
            raise ValueError(f"{key}: {name} must be a boolean")

    if "forced" in definition and not isinstance(definition["forced"], (bool, str)):
        raise ValueError(f"{key}: forced must be a boolean or diagnostic explanation")

    if "field_trigger" in definition:
        trigger = definition["field_trigger"]
        if not isinstance(trigger, dict):
            raise ValueError(f"{key}: field_trigger must be an object")
        trigger_fields = cast(dict[str, object], trigger)
        if "native_record_type" in trigger_fields:
            value = trigger_fields["native_record_type"]
            if not is_integer(value) or not 0 <= value <= 255:
                raise ValueError(f"{key}: native_record_type must be an integer byte")
        if "handler_checks_a_button" in trigger_fields and not isinstance(
            trigger_fields["handler_checks_a_button"], bool
        ):
            raise ValueError(f"{key}: handler_checks_a_button must be a boolean")

    for name in ("controller_steps", "inputs"):
        if name not in definition:
            continue
        steps: Any = definition[name]
        if not isinstance(steps, list):
            raise ValueError(f"{key}: {name} must be a list")
        for step in cast(list[Any], steps):
            if not isinstance(step, dict):
                raise ValueError(f"{key}: {name} steps must be objects")
            item = cast(dict[str, Any], step)
            if name == "inputs" and "poke" in item:
                region = item.get("region", 2)
                if (not is_integer(region)) or region not in MEMORY_REGION_CAPACITIES:
                    raise ValueError(
                        f"{key}: input write region must be SRAM (0), WRAM (2) or VRAM (3)"
                    )
                if not isinstance(item.get("data"), str):
                    raise ValueError(f"{key}: input write data must be hexadecimal")
                payload = bytes.fromhex(item["data"])
                # Check the complete span before a rerun can revoke earlier
                # results. An in-range start does not make an overlong write safe.
                capacity = MEMORY_REGION_CAPACITIES[region]
                if (
                    (not is_integer(item["poke"]))
                    or not 0 <= item["poke"] <= capacity
                    or item["poke"] + len(payload) > capacity
                ):
                    raise ValueError(f"{key}: input write span is outside native memory")
                continue
            if (not is_integer(item.get("frames"))) or item["frames"] < 0:
                raise ValueError(f"{key}: input frames must be a nonnegative integer")
            buttons: Any = item.get("buttons")
            if not isinstance(buttons, list) or any(
                not isinstance(button, str) or button not in BTN
                for button in cast(list[Any], buttons)
            ):
                raise ValueError(f"{key}: input buttons must name supported joypad buttons")

    for name in ("quest_flags", "deferred_quest_flags", "condition_fixture"):
        if name not in definition:
            continue
        flags: Any = definition[name]
        if not isinstance(flags, list):
            raise ValueError(f"{key}: {name} must be a list")
        for flag in cast(list[Any], flags):
            if not isinstance(flag, dict):
                raise ValueError(f"{key}: {name} entries must be objects")
            record = cast(dict[str, Any], flag)
            if any((not is_integer(record.get(field))) for field in ("address", "mask", "value")):
                raise ValueError(f"{key}: flag address, mask and value must be integers")
            if (
                not (
                    0 <= record["address"] < 0x20000
                    and 0 <= record["mask"] <= 255
                    and 0 <= record["value"] <= 255
                )
                or record["value"] & ~record["mask"]
            ):
                raise ValueError(f"{key}: flag address or masked value is out of range")

    if "before" in definition:
        before: Any = definition["before"]
        if not isinstance(before, dict):
            raise ValueError(f"{key}: before must be a prerequisite task object")
        validate_definition(cast(dict[str, Any], before))
    declared: Any = definition.get("targets", [])
    expected: Any = definition.get("expect", [])

    if not isinstance(declared, list) or any(
        not isinstance(e, str) or not e for e in cast(list[Any], declared)
    ):
        raise ValueError(f"{key}: targets must be a list of entry IDs")

    if "target" in definition and (
        not isinstance(definition["target"], str) or not definition["target"]
    ):
        raise ValueError(f"{key}: target must be an entry ID")

    # Each expectation is an entry ID and a positive one-based page number;
    # reject booleans as page numbers even though bool subclasses int.
    if not isinstance(expected, list) or any(
        not isinstance(p, (list, tuple))
        or len(pair := cast(Sequence[object], p)) != 2
        or not isinstance(pair[0], str)
        or not pair[0]
        or (not is_integer(pair[1]))
        or pair[1] < 1
        for p in cast(list[Any], expected)
    ):
        raise ValueError(f"{key}: expect must contain entry IDs and positive page numbers")

    # Optional observations must still refer to declared expectation records;
    # otherwise the metadata could silently exempt unrelated entries.
    observed = validate_conditional_entries(definition)
    if not observed <= {e for e, _ in cast(list[tuple[str, int]], expected)}:
        raise ValueError(f"{key}: conditional observations must belong to expect")

    collect_original_absences(definition)


def index_definitions(definitions: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Validate task definitions and index them by unique key.

    Args:
        definitions (Iterable[dict[str, Any]]): Task definitions, each containing a
            unique key.

    Returns:
        dict[str, dict[str, Any]]: Validated task definitions indexed by unique key.
            Values are the original dictionaries, not copies; later caller changes
            affect those same objects.

    Raises:
        ValueError: If a task key is missing or duplicated, or a contained task
            definition is invalid.
    """
    result: dict[str, Any] = {}
    for definition in definitions:
        key = definition.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("Every task needs a nonempty key")

        # Reject duplicate keys before indexing; dictionary replacement would
        # otherwise hide one required task and its expected pages from later audits.
        if key in result:
            raise ValueError(f"Duplicate task key: {key}")

        validate_definition(definition)
        result[key] = definition

    return result


def resolve_required_pages() -> dict[str, set[int]]:
    """
    Index required compiled pages from the current English ROM.

    The mapping is derived from compiled text, independently of stored captures, so a previously
    unseen or newly added page cannot disappear from requirements.

    Returns:
        dict[str, set[int]]: Required current-ROM page numbers by active entry
            ID. Entries absent from the compiled start index have no key.

    Raises:
        OSError: If the default English ROM or a translation source cannot be read.
        ValueError: If source metadata or a compiled text chain is malformed.
        IndexError: If a referenced entry or text token lies outside the ROM.
        KeyError: If a compiled page contains an unsupported F8 command.
    """
    from core.text import read_translation_files as script_files
    from verification.evidence.index_gameplay_text import index_page_starts

    rom = (script_files.ROOT / "build/Solid Runner (English).sfc").read_bytes()
    expected_pages: dict[str, set[int]] = {}

    # Required pages come from compiled page boundaries, including automatic
    # wrapping, rather than the count of explicit page breaks in source text.
    for entry, number in index_page_starts(rom).values():
        expected_pages.setdefault(entry, set()).add(number)

    return expected_pages
