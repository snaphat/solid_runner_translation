"""
This module checks supplied English and Japanese task results against task definitions and expected capture hashes.

The caller supplies the task definitions, current English ROM and translation-source hashes, and
required page numbers. audit() returns task failures, missing conditional observations, declared
Japanese-ROM differences, and pages supplied by prerequisites. It does not load task files,
detect missing result records, or modify its inputs.
"""

from __future__ import annotations

from typing import Any, Iterable

from verification.evidence import evaluate_results as evaluation
from verification.evidence import identify_captures as evidence
from verification.tasks import validate_requirements as requirements


def audit(
    english: Iterable[dict[str, Any]],
    japanese: Iterable[dict[str, Any]],
    definitions: dict[str, dict[str, Any]],
    required: dict[str, set[int]],
    state: evidence.CaptureIdentity,
) -> dict[str, Any]:
    """
    Audit supplied records for known, non-forced tasks without mutating them.

    Unknown task keys are skipped. Known definitions are validated before forced tasks are
    excluded; an empty definition still requires a valid execution outcome. This does not detect
    tasks absent from both supplied iterables or check task-definition fingerprints; the calling
    command performs those checks.

    Args:
        english (Iterable[dict[str, Any]]): Stored English task results. Each
            record requires a key used to look up its definition. Evaluated records
            supply status and pages, with optional prerequisite_pages; missing status
            fails execution and missing pages are treated as an empty list.
        japanese (Iterable[dict[str, Any]]): Stored Japanese task results using
            the same record fields as english.
        definitions (dict[str, dict[str, Any]]): Current task definitions keyed by unique
            task key.
        required (dict[str, set[int]]): Current-ROM required page numbers keyed
            by entry ID; pagination takes precedence over saved expectations.
        state (evidence.CaptureIdentity): Expected rom_sha256 and per-entry source hashes.

    Returns:
        dict[str, Any]: failures, conditional_observations,
            expected_original_differences, cached_prerequisites and decisions.

    Raises:
        ValueError: If a known task definition is invalid, including one marked
            forced, or evaluated English required page numbers are not positive
            integers. Booleans are rejected as page numbers.
        KeyError: If a supplied result record has no key.
    """
    failures: list[dict[str, Any]] = []
    conditional: list[dict[str, Any]] = []
    originals: list[dict[str, Any]] = []
    cached: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for language, results in (("en", english), ("jp", japanese)):
        for result in results:
            definition = definitions.get(result["key"])
            if definition is None:
                continue
            # Validate flags before they can exclude a known task from the audit.
            requirements.validate_definition(definition)
            if definition.get("forced") or definition.get("route") == "forced":
                continue

            decision = evaluation.evaluate_task_result(
                result, definition, language == "en", state, required
            )
            decisions.append(dict(language=language, key=result["key"], **decision.record()))

            # Failures affect task validity. Cached prerequisites and conditional
            # observations below explain evidence without making the task fail.
            if decision.failed:
                failure: dict[str, Any] = dict(
                    language=language, key=result["key"], status=result.get("status")
                )
                if language == "en":
                    failure["missing_pages"] = sorted(decision.missing_pages)
                else:
                    failure.update(
                        missing_targets=sorted(decision.missing_targets),
                        unexpected_original_targets=sorted(decision.unexpected_original_targets),
                    )

                if decision.status_failed:
                    failure["status_failed"] = True

                failures.append(failure)

            # Record which required pages came from native prerequisite playback
            # rather than from this task's own interaction.
            if decision.cached_pages:
                cached.append(dict(key=result["key"], pages=sorted(decision.cached_pages)))

            if decision.conditional_entries:
                absent = sorted(decision.conditional_entries)
                conditional.append(
                    dict(
                        key=result["key"],
                        entries=absent,
                        reasons={e: definition["conditional_expect"][e] for e in absent},
                    )
                )

            # Keep expected Japanese-ROM differences visible even when their absence
            # is valid for the original game.
            if decision.original_absent:
                originals.append(
                    dict(
                        key=result["key"],
                        entries=sorted(decision.original_absent),
                        reason=definition.get("original_lacks")
                        or definition.get("original_absent_reason"),
                    )
                )

    return dict(
        failures=failures,
        conditional_observations=conditional,
        expected_original_differences=originals,
        cached_prerequisites=cached,
        decisions=decisions,
    )
