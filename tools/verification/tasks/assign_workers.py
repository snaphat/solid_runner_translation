"""
This module divides screen and route tasks among workers while grouping tasks that share costly setup.

Screen assignment takes program definitions supplied by the caller and reads optional timing
data from a verification workspace. Programs sharing a named starting state are grouped to
reuse setup. Route assignment groups shared prerequisites and battle-caption starts. Both
operations return one worker's subset without running an emulator or modifying definitions.
"""

from __future__ import annotations

import collections
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

from core.validation.validate_integers import is_integer
from verification.results import store_results as storage

# Screen-name prefixes mapped to relative replay costs when timing data is absent.
# Unlisted programs cost one unit; these are balancing weights, not frame counts.
HEAVY = {
    "shop-buy-arms": 12,
    "shop-sell-arms": 6,
    "garage-swap-all": 4,
    "terminal-lib": 2,
    "terminal-news": 2,
    "terminal-info": 2,
    "terminal-requests": 2,
    "items": 2,
    "setup-full": 2,
}

# Extra relative cost paid each time a worker has to reach one of these starts.
START_COST = {"riff-garage": 4, "weapon-shop": 2, "reset-with-save": 1}


def validate_worker_partition(index: int, total: int, limit: int | None = None) -> None:
    """
    Validate integer worker bounds and an optional task limit without performing I/O.

    Args:
        index (int): Zero-based worker index, from zero through total minus one.
            Integer subclasses are accepted; booleans are rejected.
        total (int): Positive worker count. Integer subclasses are accepted;
            booleans are rejected.
        limit (int | None, optional): Nonnegative task limit; zero or None means
            unlimited. Booleans and noninteger values are rejected.
            Defaults to None.

    Raises:
        ValueError: If the index, count or limit has an invalid type or range.
    """
    if (
        (not is_integer(total))
        or (not is_integer(index))
        or total <= 0
        or not 0 <= index < total
    ):
        raise ValueError("Worker index must be within a positive integer worker count")
    if limit is not None and ((not is_integer(limit)) or limit < 0):
        raise ValueError("Task limit must be a nonnegative integer")


def _read_timing_table(value: object, label: str) -> dict[str, float]:
    """
    Validate a saved name-to-duration mapping and copy its numeric values.

    Args:
        value (object): JSON object mapping program or starting-state names to
            finite, nonnegative durations in seconds. Integers and floats,
            including subclasses, are accepted; booleans are rejected.
        label (str): Field description included in validation errors.

    Returns:
        dict[str, float]: Independent mapping with durations converted to floats.

    Raises:
        ValueError: If the input is not a dictionary with string keys and finite,
            nonnegative numeric values representable as floats.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{label} must map names to durations")

    result: dict[str, float] = {}
    for name, duration in cast(dict[object, object], value).items():
        if (
            not isinstance(name, str)
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
        ):
            raise ValueError(f"{label} must map string names to numeric durations")
        try:
            seconds = float(duration)
        except OverflowError as error:
            raise ValueError(f"{label} duration is too large") from error
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"{label} durations must be finite and nonnegative")
        result[name] = seconds
    return result


def _sum_costs(values: Iterable[float]) -> float:
    """
    Sum estimated costs without allowing overflow to corrupt worker selection.

    Args:
        values (Iterable[float]): Finite, nonnegative costs in a common unit,
            consumed once. Empty input has zero total cost.

    Returns:
        float: Sum of the supplied costs.

    Raises:
        ValueError: If the total is too large to represent as a finite float.
    """
    try:
        return math.fsum(values)
    except OverflowError as error:
        raise ValueError("Combined worker costs are too large") from error


def assign_screen_workers(
    index: int,
    total: int,
    limit: int | None,
    tag: Literal["en", "jp"],
    workspace: storage.Workspace | None = None,
    *,
    programs: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """
    Assign screen programs by estimated execution and setup cost.

    Names are sorted and limited before assignment. Programs sharing the same state are
    grouped, then split when a piece would exceed the estimated per-worker execution budget.
    Setup cost is charged for each piece. Pieces are assigned from heaviest to lightest to
    the least-loaded worker; ties use the lowest worker index. Every worker must receive the
    same definitions and timing-file contents to compute complementary assignments.

    Neither definitions nor timing files are changed. Invalid arguments are rejected before
    reading timings. An empty selection returns an empty list without reading the costs file.
    A missing file, missing language, null language value, or empty language object selects
    fallback weights. Otherwise both scripts and starts timing tables are required. An
    unmeasured program uses the mean of all saved program timings for that language; a
    missing starting-state duration costs zero.

    Args:
        index (int): Zero-based worker index, strictly less than total. Integer
            subclasses are accepted; booleans are rejected.
        total (int): Positive worker count, accepting integer subclasses but not
            booleans. Workers receiving no tasks return an empty list.
        limit (int | None): Maximum task count before partitioning; None or
            zero leaves the task list unlimited.
        tag (Literal['en', 'jp']): Language whose saved timings to use; only en and
            jp are accepted.
        workspace (Workspace | None, optional): Selects the costs JSON
            supplying observed program/start timings; None uses
            storage.DEFAULT.
        programs (Mapping[str, Mapping[str, object]]): Program names mapped to
            definitions containing a nonempty string state identifying a reusable
            starting state. Names must be nonempty strings. Extra fields such as
            playback callbacks are ignored. Supplied mappings are read without mutation.

    Returns:
        list[str]: Screen-program names assigned to this worker, grouped by
            pieces of programs sharing the same starting state. The combined results
            for all worker indices contain every selected name exactly once.

    Raises:
        ValueError: If worker bounds, the limit, language, or program definitions
            are invalid; costs JSON or timing mappings are malformed; a duration
            is negative, nonnumeric, boolean, or nonfinite; or combined costs overflow.
        UnicodeError: If an existing costs file is not valid UTF-8.
        OSError: If an existing costs file cannot be read.
    """
    validate_worker_partition(index, total, limit)
    if tag not in ("en", "jp"):
        raise ValueError("Timing language must be en or jp")
    if not isinstance(cast(object, programs), Mapping):
        raise ValueError("Screen programs must be a name-to-definition mapping")
    for name, program in programs.items():
        if (
            not isinstance(cast(object, name), str)
            or not name
            or not isinstance(cast(object, program), Mapping)
            or not isinstance(program.get("state"), str)
            or not program["state"]
        ):
            raise ValueError(
                "Each screen program needs a name and starting-state string"
            )
    workspace = storage.DEFAULT if workspace is None else workspace

    names = sorted(programs)
    names = names[:limit] if limit else names
    if not names:
        return []

    # A missing language uses estimates. A present timing record must have both
    # tables; malformed or nonfinite values cannot silently affect sorting.
    measured: object = None
    try:
        raw: object = json.loads(workspace.costs.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    else:
        if not isinstance(raw, dict):
            raise ValueError("Screen costs JSON must contain an object")
        measured = cast(dict[str, object], raw).get(tag)
    if measured is not None and measured != {}:
        if not isinstance(measured, dict):
            raise ValueError("Language timings must contain scripts and starts tables")
        timings = cast(dict[str, object], measured)
        scripts = _read_timing_table(timings.get("scripts"), "Program timings")
        start_cost = _read_timing_table(timings.get("starts"), "Starting-state timings")
        typical = _sum_costs(scripts.values()) / max(1, len(scripts))
        cost = {n: scripts.get(n, typical) for n in names}
    else:
        cost = {
            n: float(next((c for k, c in HEAVY.items() if n.startswith(k)), 1))
            for n in names
        }
        start_cost = {name: float(value) for name, value in START_COST.items()}

    # Programs sharing a start can reuse its emulator state within one worker.
    # Group by start first, then split only when the estimated group is too expensive.
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(cast(str, programs[n]["state"]), []).append(n)

    # Split costly screen groups at the estimated per-worker budget while
    # keeping their shared start state reusable within a piece.
    share = _sum_costs(cost.values()) / total
    pieces: list[tuple[float, list[str]]] = []
    for state, members in sorted(groups.items()):
        piece: list[str] = []
        weight = start_cost.get(state, 0)
        for n in sorted(members, key=lambda n: -cost[n]):
            combined = _sum_costs((weight, cost[n]))
            if piece and combined > share:
                pieces.append((weight, piece))
                piece = []
                weight = start_cost.get(state, 0)

            piece.append(n)
            weight = _sum_costs((weight, cost[n]))

        pieces.append((weight, piece))

    load: list[float] = [0] * total
    mine: list[str] = []

    # Assign the heaviest pieces first to the least-loaded worker. Stable
    # tie-breaking gives every worker the same partition from identical inputs.
    for weight, piece in sorted(pieces, key=lambda p: (-p[0], p[1])):
        worker = load.index(min(load))
        load[worker] = _sum_costs((load[worker], weight))

        if worker == index:
            mine += piece

    return mine


def assign_route_workers(
    todo: list[dict[str, Any]], index: int, total: int
) -> list[dict[str, Any]]:
    """
    Assign routes while keeping shared prerequisites together.

    Tasks with equal JSON-serialized before values share a group. Caption tasks instead use
    their first battle start, split into batches of at most fifteen in input order. Groups
    are assigned from largest to smallest, charging twenty setup units plus one per task.
    Ungrouped tasks are then assigned individually at one unit each. Ties use the lowest
    worker index. All workers must receive the same task order to produce complementary lists.

    Args:
        todo (list[dict[str, Any]]): Selected definitions, each with a nonempty route
            string. Caption tasks require a nonempty starts list or tuple whose first
            element is a JSON-serializable battle-definition dictionary. Other routes
            may have a JSON-serializable before value; falsey values mean no prerequisite.
            Dictionary key ordering does not affect grouping. Repeated input records
            are retained as separate task occurrences.
        index (int): Zero-based worker index, strictly less than total. Integer
            subclasses are accepted; booleans are rejected.
        total (int): Positive worker count, accepting integer subclasses but not
            booleans. Workers receiving no tasks return an empty list.

    Returns:
        list[dict[str, Any]]: New list containing the original task dictionaries
            assigned to this worker, not copied definitions. No input task is
            mutated by assignment.

    Raises:
        ValueError: If worker bounds or a route/caption definition are invalid,
            or a grouping value contains a JSON cycle.
        TypeError: If a grouping value cannot be serialized as JSON, or sorted
            JSON object keys cannot be compared with one another.
    """
    validate_worker_partition(index, total)
    groups: dict[str | None, list[dict[str, Any]]] = {}
    battle_lines: collections.Counter[str] = collections.Counter()
    for task in todo:
        if not isinstance(task.get("route"), str) or not task["route"]:
            raise ValueError("Each task needs a nonempty route string")
        if task["route"] == "caption":
            # Batch at most fifteen captions per battle start. Each batch stays
            # together; separate batches can be assigned to different workers.
            starts = task.get("starts")
            if (
                not isinstance(starts, (list, tuple))
                or not starts
                or not isinstance(starts[0], dict)
            ):
                raise ValueError("Caption tasks need a first battle-start dictionary")
            start = json.dumps(starts[0], sort_keys=True)
            key = f"battle {start} {battle_lines[start] // 15}"
            battle_lines[start] += 1
        else:
            key = (
                json.dumps(task["before"], sort_keys=True)
                if task.get("before")
                else None
            )

        groups.setdefault(key, []).append(task)

    load: list[int] = [0] * total
    mine: list[dict[str, Any]] = []

    # Keep prerequisite groups together to amortize setup; charge a setup
    # allowance as well as task count when balancing them across workers.
    for key, members in sorted(
        ((k, m) for k, m in groups.items() if k), key=lambda g: (-len(g[1]), g[0])
    ):
        worker = load.index(min(load))
        load[worker] += 20 + len(members)

        if worker == index:
            mine += members

    # Ungrouped routes have no shared prerequisite to preserve, so distribute
    # them individually after the more expensive groups have been placed.
    for task in groups.get(None, []):
        worker = load.index(min(load))
        load[worker] += 1

        if worker == index:
            mine.append(task)

    return mine
