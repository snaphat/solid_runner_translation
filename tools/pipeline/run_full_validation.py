"""
This workflow builds the ROM, runs source and gameplay checks, and saves logs, exit statuses, and reports for each stage.

The workflow rebuilds the English ROM, runs static and unit checks, tests boot and
mission results, replays both languages, audits target requirements and renders
the final report. Each subprocess has its own log and exit status. Input hashes
detect source changes during the run, and archived predecessor bytes allow an
identical-build comparison. Output directories must stay outside maintained
source directories and the gameplay-verification output tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os
    import pathlib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Trees included in input hashing; validation artifacts must not be written here.
SOURCE_FOLDERS = ("translation", "asm", "assets", "tools", "tests", "docs")


def collect_source_manifest(root: pathlib.Path) -> dict[str, str]:
    """
    Hash maintained repository inputs, excluding generated files.

    Args:
        root (pathlib.Path): Repository root containing translation, asm,
            assets, tools, tests, and docs. Only maintained source suffixes and
            the listed root configuration files are included; missing directories
            contribute no files.

    Returns:
        dict[str, str]: Repository-relative source paths mapped to hexadecimal
            SHA-256 digests.

    Raises:
        OSError: A selected source file cannot be read.
    """
    files = [
        p
        for folder in SOURCE_FOLDERS
        for p in (root / folder).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix in (".py", ".asm", ".inc", ".json", ".txt", ".md")
    ]
    files += [
        root / n
        for n in ("README.md", "dependencies.lock.json", "build.bat", "pyproject.toml")
        if (root / n).is_file()
    ]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }


def resolve_validation_output(root: pathlib.Path, value: str | os.PathLike[str]) -> pathlib.Path:
    """
    Resolve an output location outside source and gameplay-verification trees.

    Validation hashes maintained source directories and archives the gameplay
    results and calibration directories. Writing output inside those trees would
    change the inputs being checked or copy an archive into itself. Resolve
    directory links before comparing paths, including separately linked results
    and calibration directories. No files or directories are created here.

    Args:
        root (pathlib.Path): Repository root defining maintained source and
            gameplay-verification directory locations.
        value (str | os.PathLike[str]): Proposed output directory or default
            output parent. Relative paths use the process working directory.

    Returns:
        pathlib.Path: Resolved absolute output location. Existence and write
            permission are checked when the caller creates the directory.

    Raises:
        ValueError: The location is inside a maintained source directory or
            gameplay-verification output tree, including linked archive inputs.
        OSError: A filesystem path cannot be resolved.
    """
    output = Path(value).resolve()
    verification = root / "build/verification"
    protected = [root / folder for folder in SOURCE_FOLDERS]
    protected.extend((verification, verification / "results", verification / "calibrate"))
    for directory in protected:
        if output.is_relative_to(directory.resolve()):
            raise ValueError(f"Validation output must remain outside {directory}")
    return output


def write_json_artifact(path: pathlib.Path, value: Any) -> None:
    """
    Replace one UTF-8 JSON artifact through a sibling temporary file.

    Uses the fixed sibling name <suffix>.tmp; callers must serialize updates
    to the same artifact. Failure before replacement preserves the destination
    but may leave the temporary file.

    Args:
        path (pathlib.Path): JSON artifact to replace; the parent directory
            must exist.
        value (Any): Complete JSON value to publish through a sibling temporary
            file. Must be JSON-serializable.

    Raises:
        TypeError: value is not JSON-serializable.
        OSError: The temporary file cannot be written or replaced.
    """
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, indent=2), encoding="utf-8")
    staging.replace(path)


def run_full_validation(workers: int = 12, out: str | os.PathLike[str] | None = None) -> int:
    """
    Rebuild and execute the complete validation workflow, returning an exit code.

    Archives predecessor outputs and records stage logs and input hashes. Only
    replay calibration may change maintained inputs during validation; other
    source drift fails the run. A failed build stops the workflow; failures
    in later stages are recorded while subsequent stages still run. Predecessor
    outputs are copied, not moved, and the run writes the current-output pointer
    under build/. Interrupted stage execution writes an interrupted summary,
    then re-raises the interruption/error. Output locations are validated before
    creating directories or starting stages.

    Args:
        workers (int, optional): Emulator process count, from 1 through 12.
        out (str | os.PathLike[str] | None, optional): New output
            directory; None creates a fresh directory under build/. An existing
            directory is rejected. Maintained source trees (translation, asm,
            assets, tools, tests, docs), build/verification and separately linked
            results/calibration directories are not permitted. Directory links
            are resolved before validation; relative paths use the current
            working directory. The default build/ parent follows the same rules.

    Returns:
        int: Zero when every stage passes without unexpected input drift;
            nonzero if any stage fails.

    Raises:
        ValueError: Worker count is not an integer from 1 through 12, or an
            output location overlaps source or gameplay-verification trees.
        FileExistsError: The explicitly requested output directory already
            exists.
        OSError: Source/output files cannot be accessed or a stage cannot start.
    """
    if type(workers) is not int or not 1 <= workers <= 12:
        raise ValueError("Full validation requires 1-12 workers")

    # Validate both locations before creating anything. A linked build/ parent
    # must not redirect logs or the current-run pointer into maintained sources.
    build = resolve_validation_output(ROOT, ROOT / "build")
    if out is not None:
        out = resolve_validation_output(ROOT, out)
        out.mkdir(parents=True, exist_ok=False)

    build.mkdir(exist_ok=True)
    if out is None:
        out = Path(tempfile.mkdtemp(prefix="full-validation-", dir=build))

    print(f"Validation output: {out}", flush=True)
    (build / "full-validation-current.txt").write_text(str(out), encoding="utf-8")

    # Record inputs before any stage runs, so source edits during validation
    # are visible in the final result rather than silently accepted.
    before = collect_source_manifest(ROOT)
    write_json_artifact(out / "inputs-before.json", before)
    archived = out / "before"
    archived.mkdir()
    rom = ROOT / "build/Solid Runner (English).sfc"
    original = ROOT / "rom/Solid Runner (Japan).sfc"
    original_before = (
        hashlib.sha256(original.read_bytes()).hexdigest() if original.is_file() else None
    )

    if rom.is_file():
        shutil.copy2(rom, archived / "English.sfc")

    for name in ("results", "calibrate"):
        source = ROOT / "build/verification" / name
        if source.is_dir():
            shutil.copytree(source, archived / name)

    # The build must precede checks; the final report consumes the native replay
    # and calibration results produced by the earlier stages.
    steps = [
        ("build", ["tools/project.py", "build"]),
        ("checks-and-smoke", ["tools/project.py", "check", "--emulator"]),
        (
            "mission-results",
            [
                "-O",
                "tools/verification/checks/check_mission_results.py",
                "--out",
                str(out / "mission-results"),
            ],
        ),
        (
            "full-gameplay",
            ["-u", "tools/verification/verify_gameplay.py", "--full", "--workers", str(workers)],
        ),
        (
            "target-audit",
            ["tools/verification/audit_saved_tasks.py", "--out", str(out / "target-audit.json")],
        ),
        ("final-report", ["tools/verification/verify_gameplay.py", "--report-only"]),
    ]
    rows: list[dict[str, Any]] = []
    built_hash = None
    write_json_artifact(out / "summary.json", dict(status="running", passed=False, results=rows))

    try:
        for name, command in steps:
            print(f"START {name}", flush=True)
            started = time.monotonic()

            # Each stage gets an independent log and exit status. Keep later failures
            # visible without conflating them with subprocess output from another stage.
            with (out / (name + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )

            rows.append(
                dict(
                    step=name,
                    exit_code=result.returncode,
                    seconds=round(time.monotonic() - started, 1),
                )
            )

            if name == "build" and result.returncode == 0:
                built_hash = hashlib.sha256(rom.read_bytes()).hexdigest()

            write_json_artifact(out / "results.json", rows)
            print(f"DONE {name}: exit {result.returncode}", flush=True)

            if name == "build" and result.returncode:
                break  # Never validate a predecessor after a failed rebuild.
    except BaseException as error:
        write_json_artifact(
            out / "summary.json",
            dict(status="interrupted", passed=False, results=rows, error=str(error)),
        )
        raise

    after = collect_source_manifest(ROOT)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))

    # Direct-display calibration is an intentional output of gameplay verification;
    # other source changes invalidate the run's input snapshot.
    unexpected = [k for k in changed if k != "tools/verification/calibration/data/direct.json"]
    predecessor = archived / "English.sfc"
    final_hash = hashlib.sha256(rom.read_bytes()).hexdigest() if rom.is_file() else None
    original_after = (
        hashlib.sha256(original.read_bytes()).hexdigest() if original.is_file() else None
    )

    # Passing requires every stage to finish, stable build bytes and an
    # unchanged original ROM. Matching coverage alone cannot establish these invariants.
    summary: dict[str, Any] = dict(
        status="complete",
        results=rows,
        changed_inputs=changed,
        unexpected_input_changes=unexpected,
        rom_sha256=final_hash,
        original_sha256=original_after,
        rom_unchanged_during_checks=built_hash is not None and built_hash == final_hash,
        original_unchanged=original_before == original_after,
        rebuilt_identically=rom.read_bytes() == predecessor.read_bytes()
        if rom.is_file() and predecessor.is_file()
        else None,
        passed=len(rows) == len(steps)
        and all(r["exit_code"] == 0 for r in rows)
        and not unexpected
        and built_hash is not None
        and built_hash == final_hash
        and original_before == original_after,
    )
    final_report = out / "final-report.log"
    if final_report.is_file():
        match = re.search(
            r"entries: (\d+) verified, (\d+) not verified, (\d+) not reached, (\d+) forced",
            final_report.read_text(encoding="utf-8"),
        )
        if match:
            summary["coverage"] = dict(
                zip(
                    ("verified", "not_verified", "not_reached", "forced_only"),
                    map(int, match.groups()),
                )
            )

    write_json_artifact(out / "summary.json", summary)
    print(
        f"Full validation {'passed' if summary['passed'] else 'FAILED'}: {out / 'summary.json'}",
        flush=True,
    )
    return 0 if summary["passed"] else 1
