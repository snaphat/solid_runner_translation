"""
This module defines verification workspace paths and saves task results, ROM and translation-source hashes, screenshot indexes, and rerun status.

Workspace separates maintained task/calibration sources from generated results, screenshots and
reports. Result helpers collect language-tagged batches, attach ROM and translation-source
hashes, and replace related files with rollback. Selected reruns revoke overlapping evidence and
retain pending failures until every selected task has a fresh result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, cast

from core.validation.validate_integers import is_integer

if TYPE_CHECKING:
    import os
    import pathlib

    from verification.evidence.evaluate_results import Decision
import json
from dataclasses import dataclass
from pathlib import Path

from core.files import replace_related_files as source_files
from core.rom import load_original_rom as original_rom
from verification.evidence import identify_captures as evidence


@dataclass(init=False)
class Workspace:
    """
    Workspace stores paths to verification source files and generated output locations.

    The constructor converts path options to Path objects. Supplied relative
    paths remain relative; their directory links are not resolved. Defaults for
    the calibration JSON and task definition files remain in the repository
    containing this module when root or out changes.

    Args:
        root (str | os.PathLike[str], optional): Repository root selecting the English build
            path; defaults to the repository containing this module.
        out (str | os.PathLike[str] | None, optional): Output root; defaults to
            root/build/verification.
        results (str | os.PathLike[str] | None, optional): Stored-result directory; defaults to
            out/results.
        report (str | os.PathLike[str] | None, optional): HTML/screenshot directory; defaults to
            out/report.
        calibrate (str | os.PathLike[str] | None, optional): Calibration captures; defaults to
            out/calibrate.
        direct (str | os.PathLike[str] | None, optional): Calibration source JSON; defaults to
            tools/verification/calibration/data/direct.json in the repository containing this
            module, independently of root/out.
        costs (str | os.PathLike[str] | None, optional): Screen timing JSON; defaults to
            out/screen-costs.json.
        task_sources (dict[str, str | os.PathLike[str]] | None, optional): routes/battles
            definition paths; defaults to routes/data/routes.json and battles/data/battles.json
            beneath tools/verification in the repository containing this module, independently
            of root/out. None or an empty dictionary selects these defaults. Keys must be
            strings.
    """

    root: Path
    out: Path
    results: Path
    report: Path
    calibrate: Path
    direct: Path
    costs: Path
    task_sources: dict[str, Path]

    def __init__(
        self,
        root: str | os.PathLike[str] = Path(__file__).resolve().parents[3],
        out: str | os.PathLike[str] | None = None,
        results: str | os.PathLike[str] | None = None,
        report: str | os.PathLike[str] | None = None,
        calibrate: str | os.PathLike[str] | None = None,
        direct: str | os.PathLike[str] | None = None,
        costs: str | os.PathLike[str] | None = None,
        task_sources: dict[str, str | os.PathLike[str]] | None = None,
    ) -> None:
        """
        Convert path options to Path objects and select defaults without creating directories.

        None selects each optional path default. Other path values are converted
        directly, so an empty string denotes the current directory. Relative
        inputs remain relative; caller-supplied paths are not resolved through
        filesystem links.

        Args:
            root (str | os.PathLike[str], optional): Repository root used to find the English
                build; defaults to the repository containing this module.
            out (str | os.PathLike[str] | None, optional): Output root; None uses
                root/build/verification.
            results (str | os.PathLike[str] | None, optional): Result directory; None uses
                out/results.
            report (str | os.PathLike[str] | None, optional): Report directory; None uses
                out/report.
            calibrate (str | os.PathLike[str] | None, optional): Calibration output; None uses
                out/calibrate.
            direct (str | os.PathLike[str] | None, optional): Maintained calibration JSON; None
                selects calibration/data/direct.json independently of the output root.
            costs (str | os.PathLike[str] | None, optional): Screen timings; None uses
                out/screen-costs.json.
            task_sources (dict[str, str | os.PathLike[str]] | None, optional): Task definition
                files; None or an empty mapping selects the maintained route and battle data
                files. A nonempty dictionary is copied with each value converted to Path. Keys
                must be strings; dictionary subclasses are accepted.

        Raises:
            TypeError: If a configured path is neither a string nor a string-valued
                path-like object, or task_sources is neither None nor a dictionary
                with string keys.
        """
        if task_sources is not None:
            if not isinstance(cast(object, task_sources), dict):
                raise TypeError("task_sources must be a dictionary or None")
            if any(not isinstance(cast(object, name), str) for name in task_sources):
                raise TypeError("Task source names must be strings")

        self.root = Path(root)
        verification_root = Path(__file__).resolve().parents[1]
        self.task_sources = {
            name: Path(path)
            for name, path in (
                task_sources
                if task_sources is not None and len(task_sources) > 0
                else {
                    "routes": verification_root / "routes/data/routes.json",
                    "battles": verification_root / "battles/data/battles.json",
                }
            ).items()
        }

        # None means no override. A falsey path-like object can still supply
        # a valid path, while False and 0 must reach Path's type validation.
        self.out = Path(self.root / "build/verification" if out is None else out)
        self.results = Path(self.out / "results" if results is None else results)
        self.report = Path(self.out / "report" if report is None else report)
        self.calibrate = Path(self.out / "calibrate" if calibrate is None else calibrate)
        self.direct = Path(
            verification_root / "calibration/data/direct.json" if direct is None else direct
        )
        self.costs = Path(self.out / "screen-costs.json" if costs is None else costs)


DEFAULT = Workspace()


def collect_saved_results(
    tag: Literal["en", "jp"], workspace: Workspace | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return opening/own-task captures and task results from sorted result files.

    Args:
        tag (Literal['en', 'jp']): Result-file language prefix.
        workspace (Workspace | None, optional): Selects the results
            directory to read; None uses DEFAULT.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: Pair (pages, task results);
            opening pages are included once per batch file. The page list contains
            each task's own pages; prerequisite_pages remain attached to its task
            result and are not flattened into that list.

    Raises:
        OSError: If a result file cannot be read.
        ValueError: If a result file contains invalid UTF-8 or JSON.
        KeyError: If a stored batch lacks opening, routes or task pages.
    """
    workspace = workspace or DEFAULT
    pages: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []

    # Merge worker batches in stable filename order, keeping opening captures
    # separate from route records while exposing both in the combined page list.
    for path in sorted(workspace.results.glob(f"{tag}-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pages += data["opening"]

        for r in data["routes"]:
            routes.append(r)
            pages += r["pages"]

    return pages, routes


def infer_task_type(path: pathlib.Path) -> str:
    """
    Infer the task type from path.name, defaulting to routes.

    Args:
        path (pathlib.Path): Stored-result filename containing a -screens,
            -battles, -forced, -direct or -rest marker; otherwise classified as
            routes.

    Returns:
        str: Stored task type inferred from the filename marker, or routes
            when no supported marker is present.
    """
    return next(
        (k for k in ("screens", "battles", "forced", "direct", "rest") if f"-{k}" in path.name),
        "routes",
    )


def record_costs(workspace: Workspace | None = None) -> None:
    """
    Update the costs JSON from saved screen-task durations for both ROMs.

    Start setup costs keep the maximum observed value. Repeated script timings use the last file
    encountered. A language with no script timings retains its existing costs. The destination
    parent must exist; read/write errors propagate.

    Args:
        workspace (Workspace | None, optional): Selects the results input
            directory and costs JSON to update; None uses DEFAULT.

    Raises:
        OSError: If a result or costs file cannot be read, or the costs file cannot
            be written. A failed write can leave a partial costs file.
        ValueError: If an input file contains invalid UTF-8 or JSON.
        KeyError: If a stored screen batch lacks required route or task-key fields.
        TypeError: If stored setup durations cannot be compared as numbers.
    """
    workspace = workspace or DEFAULT
    costs: dict[str, Any] = (
        json.loads(workspace.costs.read_text(encoding="utf-8")) if workspace.costs.exists() else {}
    )
    for tag in ("en", "jp"):
        scripts: dict[str, float] = {}
        starts: dict[str, float] = {}
        for path in workspace.results.glob(f"{tag}-screens*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            scripts.update(
                {r["key"][len("screen:") :]: r["seconds"] for r in data["routes"] if "seconds" in r}
            )

            # Several workers may prepare the same start. Keep the slowest observed
            # setup cost so worker assignment does not underestimate that shared overhead.
            for name, seconds in data.get("start_seconds", {}).items():
                starts[name] = max(starts.get(name, 0), seconds)

        if scripts:
            costs[tag] = dict(scripts=scripts, starts=starts)

    workspace.costs.write_text(json.dumps(costs, indent=1, sort_keys=True), encoding="utf-8")


def replace_batches(batches: Iterable[tuple[Any, ...]], indent: int | None = None) -> None:
    """
    Serialize result batches before replacing their files with rollback on failure.

    Individual replacements are atomic; the group is not atomic across a process or
    operating-system crash. A cleanup failure after successful replacement leaves the new files
    in place. Failed rollback retains recovery files.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        batches (Iterable[tuple[Any, ...]]): Destination Path and JSON-serializable
            replacement value pairs.
        indent (int | None, optional): json.dumps indentation; None emits
            compact JSON.

    Raises:
        TypeError: If indent is a boolean or noninteger, or a replacement value
            cannot be serialized as JSON.
        ValueError: If a replacement contains circular references, destination paths
            repeat, or the destinations have no common parent path.
        OSError: If temporary files or backups cannot be written, a destination
            cannot be replaced, or temporary files cannot be removed. Replacement
            failures trigger rollback; failed rollback retains recovery files.
    """
    if indent is not None and not is_integer(indent):
        raise TypeError("indent must be an integer, not a boolean")
    source_files.replace_many(
        (path, json.dumps(data, indent=indent).encode("utf-8")) for path, data in batches
    )


def prune_selected(
    tag: Literal["en", "jp"],
    kind: str,
    keys: set[str],
    affected: set[str],
    pending_path: pathlib.Path,
    workspace: Workspace | None = None,
) -> None:
    """
    Revoke selected and overlapping direct captures, preserving unrelated records.

    Direct-display targets are pruned even when their page list already lacks the affected
    entries. Updated files are replaced together with rollback on failure.

    Args:
        tag (Literal['en', 'jp']): Result-file language prefix.
        kind (str): Task type: routes, battles, screens, forced or direct.
        keys (set[str]): Task keys whose selected replay results replace the
            previous records.
        affected (set[str]): Entry IDs whose overlapping direct-display
            captures must also be removed.
        pending_path (pathlib.Path): In-progress selected-result file to
            preserve while pruning older batches.
        workspace (Workspace | None, optional): Selects the results
            directory whose overlapping records are replaced; None uses
            DEFAULT.

    Raises:
        OSError: If a stored result cannot be read, or replacement, rollback or
            temporary-file cleanup fails.
        ValueError: If a result file contains invalid UTF-8 or JSON, or replacement
            destinations repeat or have no common parent path.
        KeyError: If a stored batch lacks required routes, task keys or page lists.
    """
    workspace = workspace or DEFAULT
    pruned: list[tuple[Path, bytes]] = []
    for path in workspace.results.glob(f"{tag}-*.json"):
        if path == pending_path:
            continue

        if infer_task_type(path) not in {kind, "rest", "direct"}:
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        kept = [r for r in data["routes"] if r["key"] not in keys]
        changed = len(kept) != len(data["routes"])

        if infer_task_type(path) == "direct" and affected:
            # Direct groups contain several entries: preserve unrelated captures.
            for result in kept:
                pages = [p for p in result["pages"] if p.get("entry") not in affected]
                changed |= len(pages) != len(result["pages"])
                result["pages"] = pages
                if result.get("targets"):
                    targets = [e for e in result["targets"] if e not in affected]
                    changed |= len(targets) != len(result["targets"])
                    result["targets"] = targets

        if changed:
            data["routes"] = kept
            pruned.append((path, json.dumps(data).encode("utf-8")))

    # Replace the affected result files together; a partial update could leave
    # old captures still counting toward coverage for selected tasks.
    source_files.replace_many(pruned)


def finish_selected(
    pending_path: pathlib.Path,
    pending: dict[str, Any],
    batches: Iterable[tuple[Any, ...]],
    keys: set[str],
    decisions: dict[str, Decision],
    missing: set[str],
) -> None:
    """
    Persist evaluated results and leave unfinished task records visible.

    Mutates pending and the supplied parsed batches before attempting their related-file
    replacement. A write failure does not undo those in-memory mutations. Every selected result
    encountered must have a decisions entry.

    Args:
        pending_path (pathlib.Path): In-progress result JSON file to replace
            with still-unfinished records.
        pending (dict[str, Any]): Mutable batch whose routes are reduced to missing task
            keys.
        batches (Iterable[tuple[Any, ...]]): Stored-result Path and parsed JSON pairs to
            update with decisions.
        keys (set[str]): Task keys whose selected replay results replace the
            previous records.
        decisions (dict[str, Decision]): Fresh task evaluation decisions keyed
            by task key.
        missing (set[str]): Task keys without a completed replacement result.

    Raises:
        KeyError: If a selected result has no decision, or a batch lacks required
            routes or task keys.
        TypeError: If updated results cannot be serialized as JSON.
        ValueError: If updated results contain circular references, replacement
            destinations repeat, or they have no common parent path.
        OSError: If replacement, rollback or temporary-file cleanup fails.
    """
    updates: list[tuple[Path, bytes]] = []

    # Keep missing worker results as pending tasks; completion of other workers
    # must not make an unfinished selected task disappear.
    pending["routes"] = [r for r in pending["routes"] if r["key"] in missing]
    updates.append((pending_path, json.dumps(pending).encode("utf-8")))

    for path, data in batches:
        changed = False

        for result in data["routes"]:
            if result["key"] not in keys:
                continue

            # Attach the current decision to each selected result. Execution status
            # remains intact; check_failed records a target/page failure after successful playback.
            result["evaluation"] = decisions[result["key"]].record()
            changed = True
            previous = result.pop("check_failed", None)

            status = result.get("status")
            if decisions[result["key"]].failed and not (
                isinstance(status, str) and status.startswith("error")
            ):
                result["check_failed"] = "Required targets or expected pages were not verified"

            changed |= previous != result.get("check_failed")

        if changed:
            updates.append((path, json.dumps(data).encode("utf-8")))

    # Save the remaining pending tasks and evaluated worker batches together,
    # so an interrupted replacement cannot silently discard unfinished tasks.
    if updates:
        source_files.replace_many(updates)


def save_full_shots(workspace: Workspace | None = None) -> None:
    """
    This function writes report/full-shots.json, an index of full-replay screenshots that quick reports can reuse.

    English captures must match the current ROM and translation-source hashes. Japanese captures
    must identify the protected original ROM. Forced and unidentified captures are omitted.
    Pixel verification is not required for this image lookup; inclusion here does not establish
    translation coverage.

    The English ROM is read from workspace.root/build/Solid Runner (English).sfc. Translation
    sources and build-input records use their maintained locations. The report directory must
    exist. The index is written directly, so a write failure can leave a partial file.

    Args:
        workspace (Workspace | None, optional): Selects the repository root for
            the English ROM, stored results, and the report directory for
            full-shots.json. None uses DEFAULT.

    Raises:
        OSError: If a ROM, translation source, build-input record or result file
            cannot be read, or full-shots.json cannot be written. A failed write
            can leave a partial screenshot index.
        ValueError: If an input file contains invalid UTF-8 or JSON, or translation
            metadata, markup or encoded text is invalid.
        KeyError: If a stored batch lacks required fields or a translation uses
            a character with no game encoding.
    """
    workspace = workspace or DEFAULT
    from core.text import read_translation_files as script_files

    # Use the same English build selected for replay in this workspace.
    rom = (workspace.root / "build/Solid Runner (English).sfc").read_bytes()
    state = evidence.build_capture_identity(rom)

    # Index non-forced screenshots by language and entry, deduplicating
    # filenames across worker batches.
    shots: dict[str, Any] = {}
    for tag in ("en", "jp"):
        for path in workspace.results.glob(f"{tag}-*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            for p in data["opening"] + [p for r in data["routes"] for p in r["pages"]]:
                current = (
                    evidence.is_current_capture(p, state)
                    if tag == "en"
                    else p.get("rom_sha256") == original_rom.SHA256
                )
                if current and p.get("entry") and p.get("shot") and not p.get("forced"):
                    shots.setdefault(tag, {}).setdefault(p["entry"], set()).add(p["shot"])

    # Store the ROM hash and English text with the screenshot index so the
    # report can reject fallback images after the build or translation changes.
    (workspace.report / "full-shots.json").write_text(
        json.dumps(
            dict(
                rom_sha256=state["rom_sha256"],
                en={
                    e["id"]: e["en"]
                    for n in script_files.NAMES
                    for e in script_files.load(n)
                    if e["en"]
                },
                shots={
                    tag: {e: sorted(names) for e, names in by.items()} for tag, by in shots.items()
                },
            )
        ),
        encoding="utf-8",
    )


def write_result(
    path: str | os.PathLike[str], data: dict[str, Any], state: evidence.CaptureIdentity
) -> None:
    """
    Add ROM and translation-source hashes to worker results and save them.

    Args:
        path (str | os.PathLike[str]): Destination worker-result JSON
            file; the parent must exist.
        data (dict[str, Any]): Mutable worker result batch; ROM and translation-source hashes are added
            before the file is written.
        state (evidence.CaptureIdentity): Expected rom_sha256 and per-entry source hashes.

    Raises:
        KeyError: If a task lacks its page list or state lacks required hash fields.
        TypeError: If the result cannot be serialized as JSON.
        ValueError: If the result contains circular references.
        OSError: If replacement, rollback or temporary-file cleanup fails.
    """
    replace_batches([(path, evidence.attach_capture_hashes(data, state))])
