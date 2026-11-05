"""
This module loads verification tasks and computes hashes used to select changed tasks for replay.

Native route, battle and forced-display definitions are read from their feature data files.
Files are read when a loader is called, not during import. Screen registries and room-start
definitions are supplied by callers. Screen hashes include callback source, captured values,
defaults, program metadata and referenced room starts. Data hashes exclude only the top-level
expect field; nested prerequisite expectations still contribute to the hash. These hashes do
not include runner code or transitive helper implementations and are not security checks.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Collection, Mapping
from operator import itemgetter
from pathlib import Path
from types import FunctionType
from typing import Any, cast

from core.validation.validate_integers import is_integer
from verification.tasks import validate_requirements as requirements

# Feature data paths are relative to tools/verification, not the working directory.
VERIFICATION_ROOT = Path(__file__).resolve().parents[1]


def _read_unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """
    Decode a JSON object without silently replacing repeated fields.

    Args:
        pairs (list[tuple[str, object]]): Decoded field names and values in file order.

    Returns:
        dict[str, object]: Fields with their original decoded values.

    Raises:
        ValueError: If an object contains the same field name twice.
    """
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON field: {name}")
        result[name] = value
    return result


def _read_json(path: Path) -> Any:
    """
    Read UTF-8 task data with duplicate-field detection.

    Args:
        path (Path): JSON file to read; the file is not modified.

    Returns:
        Any: Decoded JSON value; callers validate the required data structure.

    Raises:
        OSError: If the file cannot be opened or read.
        UnicodeError: If its contents are not valid UTF-8.
        ValueError: If JSON syntax is invalid, an object repeats a field name, or a
            number uses the nonstandard NaN or Infinity spelling.
    """
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_read_unique_object,
        parse_constant=_reject_nonfinite,
    )


def _reject_nonfinite(value: str) -> None:
    """
    Reject nonstandard floating-point constants while decoding task JSON.

    Args:
        value (str): NaN, Infinity or -Infinity token supplied by the JSON decoder.

    Raises:
        ValueError: Whenever the decoder encounters one of these tokens.
    """
    raise ValueError(f"Nonfinite number in task data: {value}")


def _validate_task_list(value: Any) -> list[dict[str, Any]]:
    """
    Validate a task array before indexing can hide duplicate task keys.

    Args:
        value (Any): Decoded array of task objects with unique nonempty keys.

    Returns:
        list[dict[str, Any]]: The supplied array, unchanged and in its original order.

    Raises:
        ValueError: If the array, task objects, keys or task requirements are invalid.
    """
    if not isinstance(value, list) or any(
        not isinstance(task, dict) for task in cast(list[Any], value)
    ):
        raise ValueError("Task data must be an array of objects")
    tasks = cast(list[dict[str, Any]], value)
    requirements.index_definitions(tasks)
    return tasks


def load_routes(root: Path = VERIFICATION_ROOT) -> list[dict[str, Any]]:
    """
    Read fresh field-route dictionaries from routes/data/routes.json.

    Args:
        root (Path): Verification directory containing routes/data/routes.json;
            defaults to this checkout's tools/verification directory.

    Returns:
        list[dict[str, Any]]: Fresh route definitions; caller mutations do not change the
            maintained routes.json file.

    Raises:
        OSError: If the data file cannot be read.
        UnicodeError: If the file is not UTF-8.
        ValueError: If JSON, task requirements or unique task keys are invalid.
    """
    return _validate_task_list(_read_json(root / "routes/data/routes.json"))


def load_battles(root: Path = VERIFICATION_ROOT) -> list[dict[str, Any]]:
    """
    Read fresh battle dictionaries from battles/data/battles.json.

    Args:
        root (Path): Verification directory containing battles/data/battles.json;
            defaults to this checkout's tools/verification directory.

    Returns:
        list[dict[str, Any]]: Fresh battle and arena definitions from battles.json.

    Raises:
        OSError: If the data file cannot be read.
        UnicodeError: If the file is not UTF-8.
        ValueError: If JSON, task requirements or unique task keys are invalid.
    """
    return _validate_task_list(_read_json(root / "battles/data/battles.json"))


def load_forced_tasks(root: Path = VERIFICATION_ROOT) -> list[dict[str, Any]]:
    """
    Read diagnostic task dictionaries from routes/data/forced.json.

    Args:
        root (Path): Verification directory containing routes/data/forced.json;
            defaults to this checkout's tools/verification directory.

    Returns:
        list[dict[str, Any]]: Fresh forced-display diagnostic definitions from forced.json.

    Raises:
        OSError: If the data file cannot be read.
        UnicodeError: If the file is not UTF-8.
        ValueError: If JSON, the tasks array or its task requirements are invalid.
    """
    data = _read_json(root / "routes/data/forced.json")
    if not isinstance(data, dict):
        raise ValueError("Forced-display data must be an object")
    return _validate_task_list(cast(dict[str, Any], data).get("tasks"))


def load_forced_room(root: Path = VERIFICATION_ROOT) -> tuple[int, int]:
    """
    Read the stage and variant used to display diagnostic text.

    This room selection does not establish native reachability of the displayed text.

    Args:
        root (Path): Verification directory containing routes/data/forced.json;
            defaults to this checkout's tools/verification directory.

    Returns:
        tuple[int, int]: Stage and variant, each an integer from zero through 255.

    Raises:
        OSError: If the data file cannot be read.
        UnicodeError: If the file is not UTF-8.
        ValueError: If JSON is invalid or room is not a two-integer array in range.
            Booleans are rejected.
    """
    data = _read_json(root / "routes/data/forced.json")
    room: Any = (
        cast(dict[str, Any], data).get("room") if isinstance(data, dict) else None
    )
    if (
        not isinstance(room, list)
        or len(cast(list[Any], room)) != 2
        or any(
            not is_integer(value) or not 0 <= value <= 255
            for value in cast(list[Any], room)
        )
    ):
        raise ValueError(
            "Diagnostic room must contain a stage and variant from 0 through 255"
        )
    values = cast(list[int], room)
    return (values[0], values[1])


def load_screen_starts(root: Path = VERIFICATION_ROOT) -> dict[str, dict[str, Any]]:
    """
    Read named room-setup definitions used by screen programs.

    These objects describe setup instructions, not serialized emulator states. This loader
    checks names and object structure; screen runners validate individual setup instructions.

    Args:
        root (Path): Verification directory containing screens/data/screen_starts.json;
            defaults to this checkout's tools/verification directory.

    Returns:
        dict[str, dict[str, Any]]: Fresh setup objects indexed by nonempty names.

    Raises:
        OSError: If the data file cannot be read.
        UnicodeError: If the file is not UTF-8.
        ValueError: If JSON is invalid or named starts are not objects.
    """
    data = _read_json(root / "screens/data/screen_starts.json")
    if not isinstance(data, dict) or any(
        not name or not isinstance(value, dict)
        for name, value in cast(dict[str, Any], data).items()
    ):
        raise ValueError("Screen starts must map nonempty names to setup objects")
    return cast(dict[str, dict[str, Any]], data)


def _encode_hash_value(value: object, active: set[int] | None = None) -> object:
    """
    Represent a supported task value with explicit type information for hashing.

    Args:
        value (object): None, bool, int, finite float, str, bytes, list, tuple or dict.
            Containers may nest these types. Subclasses use their base type's value;
            subclass identity is not encoded. Dictionary order is ignored, while
            sequence order and dictionary-key types are preserved.
        active (set[int] | None): Container identities on the current recursion path.
            Defaults to a new set. Recursive calls temporarily add their container
            identity and remove it before returning or raising.

    Returns:
        object: JSON-compatible tagged lists containing the value and its base type.
            No input container is modified. Dictionary pairs are sorted by the encoded
            key, so mixed key types do not require Python comparisons. String encodings
            preserve every code point, including explicitly supplied surrogates.

    Raises:
        TypeError: If a value has an unsupported type.
        ValueError: If a float is nonfinite or a container contains a reference cycle.
    """
    if value is None:
        return ["none"]
    # Boolean task flags are valid inputs, but must differ from integer settings.
    if isinstance(value, bool):
        return ["bool", value]
    if is_integer(value):
        # Hexadecimal avoids Python's decimal conversion limit for large integers.
        return ["int", hex(int.__int__(value))]
    if isinstance(value, float):
        number = float.__float__(value)
        if not math.isfinite(number):
            raise ValueError("Task hash inputs must contain only finite floats")
        return ["float", number.hex()]
    if isinstance(value, str):
        # JSON escapes can equate a non-BMP character with two explicit surrogates.
        # UTF-8 with surrogatepass preserves that distinction before JSON serialization.
        return ["str", str.__str__(value).encode("utf-8", "surrogatepass").hex()]
    if isinstance(value, bytes):
        return ["bytes", bytes.hex(value)]
    if not isinstance(value, (list, tuple, dict)):
        raise TypeError(f"Unsupported task hash value: {type(value).__name__}")

    if active is None:
        active = set()
    identity = id(cast(object, value))
    if identity in active:
        raise ValueError("Task hash inputs contain a reference cycle")
    active.add(identity)
    try:
        if isinstance(value, dict):
            # Encode keys as values rather than letting JSON convert keys to strings.
            pairs: list[tuple[str, object]] = []
            for key, item in dict[object, object].items(
                cast(dict[object, object], value)
            ):
                encoded_key = _encode_hash_value(key, active)
                pairs.append(
                    (
                        json.dumps(encoded_key, ensure_ascii=True),
                        [encoded_key, _encode_hash_value(item, active)],
                    )
                )
            pairs.sort(key=itemgetter(0))
            return ["dict", [pair[1] for pair in pairs]]
        sequence = (
            list[object].__iter__(cast(list[object], value))
            if isinstance(value, list)
            else tuple[object, ...].__iter__(cast(tuple[object, ...], value))
        )
        items = [_encode_hash_value(item, active) for item in sequence]
        return ["list" if isinstance(value, list) else "tuple", items]
    finally:
        active.remove(identity)


def _validate_kind(kind: str) -> None:
    """
    Reject task types for which no loading or hashing behavior is defined.

    Args:
        kind (str): One of routes, battles, forced, screens or direct.

    Raises:
        ValueError: If kind is not a supported task type.
    """
    if kind not in ("routes", "battles", "forced", "screens", "direct"):
        raise ValueError(f"Unknown task type: {kind!r}")


def _collect_start_dependencies(
    state: str, starts: Mapping[str, Mapping[str, object]]
) -> list[tuple[str, dict[str, object]]]:
    """
    Collect the earlier start definitions needed by a screen's reset-with-save setup.

    Args:
        state (str): Selected screen-start name; roam and power-on are built-in snapshots.
        starts (Mapping[str, Mapping[str, object]]): Named setup objects. A truthy after
            field must name another start. Objects are read without mutation.

    Returns:
        list[tuple[str, dict[str, object]]]: Ancestor names and shallow copies of their
            definitions in traversal order; the selected start itself is excluded.

    Raises:
        KeyError: If a selected or ancestor start has no definition.
        ValueError: If a definition is not a mapping, an after reference is not a string,
            or the references form a cycle.
    """
    seen: set[str] = set()
    result: list[tuple[str, dict[str, object]]] = []
    while state not in ("roam", "power-on"):
        if state in seen:
            raise ValueError("Screen start dependencies contain a cycle")
        seen.add(state)
        start = starts[state]
        if not isinstance(cast(object, start), Mapping):
            raise ValueError("Screen start must be a mapping")
        if len(seen) > 1:
            result.append((state, dict(start)))
        previous = start.get("after")
        if not previous:
            break
        if not isinstance(previous, str):
            raise ValueError("Screen start after must name an earlier start")
        state = previous
    return result


def hash_task_inputs(
    kind: str,
    item: dict[str, Any] | str,
    *,
    programs: Mapping[str, Mapping[str, object]] | None = None,
    starts: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    """
    Hash the maintained definition inputs for one replay task.

    The supplied objects are read without mutation and callbacks are not executed.
    Only the top-level expect field is excluded from data tasks. Screen callback source,
    closure values, defaults, metadata and start definitions contribute to the hash. This
    includes earlier starts referenced through after; called helpers and module globals do not.
    Types are encoded explicitly, so bytes, strings, tuples, lists and dictionary-key types
    remain distinct. Hashes also include the task type.

    Args:
        kind (str): Task type: routes, battles, screens, forced or direct.
        item (dict[str, Any] | str): Task definition for route/battle/forced kinds;
            registered program name for screens. Direct tasks also use a dictionary.
            This function checks serialization, not the complete task schema.
        programs (Mapping[str, Mapping[str, object]] | None): Screen programs indexed
            by name, each with a Python function in run and a nonempty state name.
            Required for screens; ignored otherwise. Defaults to None.
        starts (Mapping[str, Mapping[str, object]] | None): Room-start objects indexed
            by state name. Built-in roam and power-on starts have no data record.
            Required for screens; ignored otherwise. Defaults to None.

    Returns:
        str: Hexadecimal SHA-1 of the type-tagged inputs. Supported values
            are None, booleans, integers, finite floats, strings, bytes, lists, tuples
            and dictionaries nesting those values. Dictionary order is ignored;
            subclasses use their base type's value rather than their class identity.

    Raises:
        ValueError: If kind, item, required mappings or callback metadata are invalid,
            a closure cell is empty, or inputs contain cycles or nonfinite floats.
        TypeError: If inputs contain unsupported values.
        KeyError: If the program, its selected start or an ancestor start is missing.
        OSError: If inspect cannot retrieve a screen callback's source.
    """
    _validate_kind(kind)
    # The screen task hash includes the program and its start definition, not
    # just the displayed text, so changed input routes invalidate saved results.
    if kind == "screens":
        if not isinstance(item, str) or not item:
            raise ValueError("Screen item must be a nonempty program name")
        if not isinstance(cast(object, programs), Mapping) or not isinstance(
            cast(object, starts), Mapping
        ):
            raise ValueError("Screen hashes require program and start mappings")
        assert programs is not None and starts is not None
        script = programs[item]
        if not isinstance(cast(object, script), Mapping):
            raise ValueError("Screen program must be a mapping")
        run = script.get("run")
        state = script.get("state")
        if not isinstance(run, FunctionType) or not isinstance(state, str) or not state:
            raise ValueError("Screen program needs a Python function and a state name")
        # The runner supplies these two snapshots directly, without a JSON setup record.
        start = None if state in ("roam", "power-on") else starts[state]
        if start is not None and not isinstance(cast(object, start), Mapping):
            raise ValueError("Screen start must be a mapping")
        if start is None and state not in ("roam", "power-on"):
            raise ValueError("Screen start must be a mapping")

        # A callback factory may produce identical function source for different
        # menus; include its captured arguments to distinguish those programs.
        closure = [c.cell_contents for c in (run.__closure__ or ())]
        values: Any = [
            {k: v for k, v in script.items() if k != "run"},
            dict(start) if start is not None else None,
            inspect.getsource(run),
            closure,
        ]
        # Factory arguments can also be stored in defaults instead of closure cells.
        if run.__defaults__ is not None or run.__kwdefaults__ is not None:
            values.append(
                {"defaults": run.__defaults__, "kwdefaults": run.__kwdefaults__}
            )
        # Reset-with-save setup replays an earlier start before resetting the console.
        dependencies = _collect_start_dependencies(state, starts)
        if dependencies:
            values.append({"start_dependencies": dependencies})
    else:
        if not isinstance(item, dict):
            raise ValueError("Data task input must be a dictionary")

        # Expected pages are recorded output. Excluding expect prevents merely
        # recording those pages from invalidating the route's input fingerprint.
        values = {k: v for k, v in item.items() if k != "expect"}

    text = json.dumps(_encode_hash_value((kind, values)), ensure_ascii=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def is_selected_task(keys: Collection[str] | None, key: str) -> bool:
    """
    Determine whether a task is included in the requested replay selection.

    Args:
        keys (Collection[str] | None): Collection of nonempty task-key strings; None
            selects every key and an empty collection selects none. Bare strings are
            rejected rather than interpreted as substring selections.
        key (str): Nonempty task key whose inclusion is being tested.

    Returns:
        bool: True for every key when selection is None, otherwise the result of
            membership in the selection; an empty selection permits no keys.

    Raises:
        ValueError: If key is empty or not a string, or keys is not a collection of
            nonempty strings. A bare string is not a selection collection.
    """
    if not isinstance(cast(object, key), str) or not key:
        raise ValueError("Task key must be a nonempty string")
    if keys is not None and (
        isinstance(keys, (str, bytes, bytearray))
        or not isinstance(cast(object, keys), Collection)
        or any(not isinstance(cast(object, value), str) or not value for value in keys)
    ):
        raise ValueError("Selected keys must be a collection of nonempty strings")
    return keys is None or key in keys


def hash_task_definitions(
    kind: str,
    root: Path = VERIFICATION_ROOT,
    *,
    programs: Mapping[str, Mapping[str, object]] | None = None,
    starts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, str]:
    """
    Index the current input hash of every maintained task of one type.

    Saved-result auditing and changed-task replay compare this mapping with recorded task
    hashes. Screen keys include their screen: prefix; other task types retain the keys
    stored in their definition data.

    Args:
        kind (str): Task type: routes, battles, screens, forced or direct. Direct tasks
            are generated at runtime and have no maintained definitions to index.
        root (Path): Verification directory containing feature data subdirectories;
            defaults to this checkout's tools/verification directory.
        programs (Mapping[str, Mapping[str, object]] | None): Named screen programs,
            required for screens and ignored otherwise. Defaults to None.
        starts (Mapping[str, Mapping[str, object]] | None): Named room-start objects,
            required for screens and ignored otherwise. Defaults to None.

    Returns:
        dict[str, str]: Task key to current definition hash; empty for direct tasks.

    Raises:
        OSError: If a definition file or callback source cannot be read.
        UnicodeError: If a definition file is not UTF-8.
        ValueError: If kind, task definitions, screen mappings or serializable inputs
            are invalid, including duplicate keys, cycles and nonfinite floats.
        TypeError: If hash inputs contain unsupported values.
        KeyError: If a screen's selected start or an ancestor start is missing.
    """
    # Screen keys carry a family prefix because their names come from the
    # program registry; data-defined route families already store complete task keys.
    definitions = load_task_definitions(kind, root, programs=programs)
    if kind == "screens":
        if not isinstance(cast(object, starts), Mapping):
            raise ValueError("Screen hashes require a start mapping")
        return {
            key: hash_task_inputs(
                kind, key[len("screen:") :], programs=programs, starts=starts
            )
            for key in definitions
        }
    return {key: hash_task_inputs(kind, task) for key, task in definitions.items()}


def load_task_definitions(
    kind: str,
    root: Path = VERIFICATION_ROOT,
    *,
    programs: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Read and validate task definitions indexed by key.

    Args:
        kind (str): Task type: routes, battles, screens, forced or direct.
        root (Path): Verification directory containing feature data subdirectories;
            defaults to this checkout's tools/verification directory.
        programs (Mapping[str, Mapping[str, object]] | None): Named screen programs,
            required for screens and ignored otherwise. Defaults to None. This function
            validates result-evaluation fields; it does not execute screen callbacks.

    Returns:
        dict[str, dict[str, Any]]: Validated data-file definitions by key, or
            fresh result-evaluation dictionaries derived from the supplied screen registry.
            Direct tasks return an empty mapping without reading files. Screen target
            lists are copied, so editing them does not change the registry.

    Raises:
        OSError: If a definition file cannot be read.
        UnicodeError: If a definition file is not UTF-8.
        ValueError: If kind, JSON, task requirements, unique keys or screen evaluation
            fields are invalid.
    """
    _validate_kind(kind)
    if kind == "routes":
        return requirements.index_definitions(load_routes(root))

    if kind == "battles":
        return requirements.index_definitions(load_battles(root))

    if kind == "forced":
        return requirements.index_definitions(load_forced_tasks(root))

    # Screens declare their checks in executable programs instead of route
    # JSON. Expose the result-evaluation metadata needed to distinguish peripheral fixtures.
    if kind == "screens":
        if not isinstance(cast(object, programs), Mapping):
            raise ValueError("Screen definitions require a program mapping")
        assert programs is not None
        result: dict[str, dict[str, Any]] = {}
        for name, script in programs.items():
            if (
                not isinstance(cast(object, name), str)
                or not name
                or not isinstance(cast(object, script), Mapping)
            ):
                raise ValueError("Screen programs must map nonempty names to metadata")
            targets = script.get("targets", [])
            if not isinstance(targets, list) or any(
                not isinstance(value, str) or not value
                for value in cast(list[object], targets)
            ):
                raise ValueError("Screen targets must be a list of nonempty entry IDs")
            forced = script.get("forced")
            label = script.get("turbo_stub")
            if any(
                value is not None and (not isinstance(value, str) or not value.strip())
                for value in (forced, label)
            ):
                raise ValueError(
                    "Screen forced and turbo_stub fields must be nonblank labels"
                )
            result[f"screen:{name}"] = dict(
                route="screen",
                peripheral=bool(label),
                forced=forced,
                targets=list(cast(list[str], targets)),
            )
        return result

    return {}


def collect_task_entries(task: dict[str, Any]) -> set[str]:
    """
    Collect entry IDs referenced by a task definition or result.

    Args:
        task (dict[str, Any]): Definition or stored result with an optional nonempty
            target string, targets list, expect entry/page pairs and captured pages list.
            Absent fields contribute no IDs. Capture objects with no entry, None or an
            empty entry string are skipped. Supplied objects are not modified.

    Returns:
        set[str]: Referenced target, expected and own-page entry IDs. Prerequisite
            pages and conditional_expect keys are not independently included.

    Raises:
        ValueError: If the task, target IDs, expectation pairs or captured pages are
            malformed. Page numbers must be positive integers, excluding booleans.
    """
    if not isinstance(cast(object, task), dict):
        raise ValueError("Task must be an object")
    targets: Any = task.get("targets", [])
    if not isinstance(targets, list) or any(
        not isinstance(value, str) or not value for value in cast(list[Any], targets)
    ):
        raise ValueError("targets must be a list of nonempty entry IDs")
    result = set(cast(list[str], targets))
    if "target" in task:
        target: Any = task["target"]
        if not isinstance(target, str) or not target:
            raise ValueError("target must be a nonempty entry ID")
        result.add(target)

    # Expected pages are one-based; booleans cannot identify a page.
    expected: Any = task.get("expect", [])
    if not isinstance(expected, list):
        raise ValueError("expect must be a list of entry/page pairs")
    for pair in cast(list[Any], expected):
        values = cast(list[Any] | tuple[Any, ...], pair)
        if (
            not isinstance(pair, (list, tuple))
            or len(values) != 2
            or not isinstance(values[0], str)
            or not values[0]
            or not is_integer(values[1])
            or values[1] < 1
        ):
            raise ValueError("expect must contain entry IDs and positive page numbers")
        result.add(values[0])

    # Only this task's own captures contribute, not captures from prerequisite routes.
    pages: Any = task.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError("pages must be a list of capture objects")
    for page in cast(list[Any], pages):
        if not isinstance(page, dict):
            raise ValueError("Each captured page must be an object")
        entry: Any = cast(dict[str, Any], page).get("entry")
        if entry is None or entry == "":
            continue
        if not isinstance(entry, str):
            raise ValueError("Captured entry IDs must be strings")
        result.add(entry)
    return result
