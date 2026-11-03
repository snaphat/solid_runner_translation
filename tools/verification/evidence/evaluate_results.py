"""
This module evaluates task outcomes and captured pages and reports task failures separately from overall translation coverage.

Task success and aggregate entry coverage are independent: another task's good capture cannot
erase a failed route. English requires current pixel-verified pages; Japanese checks
original-ROM target presence with declared differences. Decision records explain missing
pages/targets and classified exceptions.

Forced and peripheral display checks may pass their own requirements without establishing
ordinary native reachability. Reports and saved-result audits recompute these decisions rather
than trusting persisted pass flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Iterable

from core.rom import load_original_rom as original_rom
from core.validation.validate_integers import is_integer
from verification.evidence import identify_captures as evidence
from verification.tasks import validate_requirements as requirements

# These outcomes permit evaluation of the captured pages; they do not establish success
# on their own. Any other outcome fails unless an original-ROM exception applies.
EVIDENCE_GRADED_OUTCOMES = frozenset({"idle", "frame_limit", "repeated_script"})


# Outcomes permitted for the Japanese ROM when a task explicitly explains its
# original_lacks repair difference: either the target or its added handler is absent.
ORIGINAL_LACKS = frozenset({"target_not_reached", "handler_unavailable"})


def is_failure_status(
    status: object, english: bool = True, definition: dict[str, Any] | None = None
) -> bool:
    """
    Classify a task outcome as an execution failure.

    An explained original_lacks difference permits the original ROM to miss a restored target.
    Frame-limit outcomes are graded by captured requirements rather than treated as automatic
    execution failures. Japanese targets_incomplete is also graded through missing-target
    requirements rather than treated as an execution-status failure by itself.

    Boolean options require bool values; numeric zero and one are rejected.

    Args:
        status (object): Recorded route outcome. Missing, non-string and unknown
            values fail rather than being treated as successful execution.
        english (bool, optional): True for the English ROM; False for the
            Japanese original.
            Defaults to True.
        definition (dict[str, Any] | None, optional): Task definition with any
            original_lacks explanation. Japanese missing-target/handler outcomes
            validate the absence fields before applying this exception.

    Returns:
        bool: True for an execution-failure outcome, accounting for explicitly
            declared original-ROM differences.

    Raises:
        TypeError: If english is not a boolean.
        ValueError: If a Japanese missing-target/handler outcome supplies invalid
            absence explanations or explicit absence targets.
    """
    if type(english) is not bool:
        raise TypeError("Boolean options require bool values: english")
    if not isinstance(status, str):
        return True
    if not english and status in ORIGINAL_LACKS and definition is not None:
        # An exception requires a validated explanation, not merely a truthy value.
        requirements.collect_original_absences(definition)
        if definition.get("original_lacks"):
            return False

    if not english and status == "targets_incomplete":
        return False
    return status not in EVIDENCE_GRADED_OUTCOMES


@dataclass
class Decision:
    """
    Decision stores the failure flags and page/entry sets used to evaluate one task.

    status_failed defaults to False. Each omitted set defaults to a separate
    empty set. Sets supplied explicitly are retained by reference. The failed
    property considers status_failed, missing_pages, missing_targets and
    unexpected_original_targets. The remaining sets explain the result without
    making it fail.

    Args:
        status_failed (bool, optional): Whether native task execution failed.
        missing_pages (set[tuple[str, int]], optional): Required (entry ID, page number) pairs
            not verified.
        missing_targets (set[str], optional): Declared entry targets not reached with eligible
            evidence.
        unexpected_original_targets (set[str], optional): Targets observed in Japanese despite a
            declared original-ROM absence.
        cached_pages (set[tuple[str, int]], optional): Required pages supplied only by
            prerequisite captures.
        conditional_entries (set[str], optional): Conditionally expected entries not observed.
        original_absent (set[str], optional): Declared original-ROM target absences not reached
            with eligible native captures. A target can also appear in
            unexpected_original_targets if an uncredited capture contradicts its declared
            absence.
    """

    status_failed: bool = False
    missing_pages: set[tuple[str, int]] = field(default_factory=set[tuple[str, int]])
    missing_targets: set[str] = field(default_factory=set[str])
    unexpected_original_targets: set[str] = field(default_factory=set[str])
    cached_pages: set[tuple[str, int]] = field(default_factory=set[tuple[str, int]])
    conditional_entries: set[str] = field(default_factory=set[str])
    original_absent: set[str] = field(default_factory=set[str])

    @property
    def failed(self) -> bool:
        """
        Check for execution errors, missing requirements or contradicted absences.

        Returns:
            bool: Whether any execution failure, missing requirement or unexpected
                original-ROM target remains in this decision.
        """
        return bool(
            self.status_failed
            or self.missing_pages
            or self.missing_targets
            or self.unexpected_original_targets
        )

    def record(self) -> dict[str, Any]:
        """
        Return JSON-compatible decision fields with sorted evidence lists.

        Returns:
            dict[str, Any]: JSON-compatible decision fields with evidence sets converted to
                sorted lists. The decision itself remains unchanged.
        """
        return dict(
            failed=self.failed,
            status_failed=self.status_failed,
            **{
                name: sorted(getattr(self, name))
                for name in (
                    "missing_pages",
                    "missing_targets",
                    "unexpected_original_targets",
                    "cached_pages",
                    "conditional_entries",
                    "original_absent",
                )
            },
        )


def evaluate_task_result(
    result: dict[str, Any],
    definition: dict[str, Any],
    english: bool,
    state: evidence.CaptureIdentity | None = None,
    required: dict[str, set[int]] | None = None,
) -> Decision:
    """
    Evaluate captured evidence without changing it or loading an emulator.

    English requires current pixel-verified pages. Japanese checks native target presence and
    explained original-ROM differences. Diagnostic and peripheral tasks can pass their own
    checks without establishing ordinary native coverage.

    Boolean options require bool values; numeric zero and one are rejected.

    Args:
        result (dict[str, Any]): Task result with key, status, pages and optional
            prerequisite_pages; read without modification.
        definition (dict[str, Any]): Task requirements, including target(s), expect,
            conditional_expect and original-ROM differences.
        english (bool): True for the English ROM; False for the Japanese
            original.
        state (evidence.CaptureIdentity | None, optional): Expected English ROM and translation-source hashes.
            Required when evaluating English pages; None is suitable for
            Japanese-only evaluation.
        required (dict[str, set[int]] | None, optional): Current-ROM
            pagination. None falls back to nonconditional recorded expect
            entry/page pairs.

    Returns:
        Decision: Execution failure, missing pages/targets and classified
            evidence exceptions.

    Raises:
        TypeError: If english is not a boolean.
        AssertionError: If English evaluation is requested without state.
        ValueError: If the task definition is invalid, or English required pagination
            includes a boolean, noninteger or nonpositive page number.
    """
    if type(english) is not bool:
        raise TypeError("Boolean options require bool values: english")
    requirements.validate_definition(definition)
    decision = Decision(
        status_failed=is_failure_status(result.get("status", ""), english, definition)
    )
    own = result.get("pages", [])
    inherited = result.get("prerequisite_pages", [])
    diagnostic = definition.get("route") == "forced" or bool(definition.get("forced"))
    device = definition.get("route") == "screen" and bool(definition.get("peripheral"))

    # Japanese tasks establish native reachability, not English page counts.
    # Explicit original-ROM absences are checked against what was actually shown.
    if not english:
        seen = {
            p.get("entry")
            for p in own + inherited
            if evidence.is_native_capture(p) and p.get("rom_sha256") == original_rom.SHA256
        }
        absent = requirements.collect_original_absences(definition)

        # Even an uncredited diagnostic capture contradicts a claim of absence.
        decision.unexpected_original_targets = absent & {p.get("entry") for p in own + inherited}
        if not diagnostic and not device:
            decision.missing_targets = (
                requirements.collect_declared_targets(definition) - absent - seen
            )

        decision.original_absent = absent - seen
        return decision

    if state is None:
        raise AssertionError("English result evaluation requires expected ROM and translation-source hashes")

    # Keep captures from this task separate from its prerequisites so a result
    # can report when earlier native playback supplied the required evidence.
    actual = evidence.collect_verified_page_pairs(
        own, state, allow_forced=diagnostic, allow_peripheral=device
    )
    prerequisite = evidence.collect_verified_page_pairs(
        inherited, state, allow_forced=diagnostic, allow_peripheral=device
    )
    expected = (
        requirements.resolve_expected_pages(definition, required)
        if required is not None
        else {
            tuple(p)
            for p in definition.get("expect", [])
            if p[0] not in requirements.validate_conditional_entries(definition)
        }
    )

    # Missing expected pages and missing declared entries are independent
    # failures: capturing one page does not satisfy a multi-page requirement.
    decision.missing_pages = expected - actual - prerequisite
    decision.missing_targets = requirements.collect_declared_targets(definition) - {
        e for e, _ in actual | prerequisite
    }
    decision.cached_pages = expected & (prerequisite - actual)
    decision.conditional_entries = requirements.validate_conditional_entries(definition) - {
        e for e, _ in actual | prerequisite
    }
    return decision


@dataclass
class Coverage:
    """
    Record entry coverage independently of individual task success.

    Args:
        status (Literal['ok', 'bad', 'none', 'forced']): Entry-level result;
            evaluated independently of task execution success.
        missing_pages (set[int]): Required page numbers lacking eligible
            captures.
        category (Literal['native', 'peripheral', 'none']): Evidence category;
            forced captures never establish native coverage.
    """

    status: str
    missing_pages: set[int]
    category: str


def evaluate_coverage(
    pages: list[dict[str, Any]], required: Collection[int], state: evidence.CaptureIdentity
) -> Coverage:
    """
    This function determines whether captures verify one translation entry and all of its required pages.

    Every eligible dialogue capture must pass. Repeated interface-menu draws need at least one
    passing capture per observed page. Required pages must also be present in either case.

    Ordinary captures are used when available. Captures made with the replacement peripheral
    driver are used only if no ordinary captures qualify. Forced displays do not establish
    coverage. These categories come from capture fields; this function does not inspect emulator
    modifications itself.

    A qualifying record has matching ROM and entry-source hashes, a positive page number, a
    boolean verified value, and permission to count toward coverage. Failed pixel checks remain
    in the qualifying records so their failures cannot be hidden by filtering them out.

    Args:
        pages (list[dict[str, Any]]): Captures for one translation entry, including
            repeated observations from different tasks. Group entries before
            calling; page numbers alone do not distinguish different entries.
        required (Collection[int]): Required page numbers for this single entry.
        state (evidence.CaptureIdentity): Expected rom_sha256 and per-entry source hashes.

    Returns:
        Coverage: Entry status, missing required page numbers and evidence
            category.

    Raises:
        ValueError: If a required page is a boolean, noninteger or not positive.
    """
    if any(not is_integer(page) or page < 1 for page in required):
        raise ValueError("Required page numbers must be positive integers, not booleans")
    eligible = [p for p in pages if evidence.is_eligible_capture(p, state, allow_peripheral=True)]

    # Select the preferred group by capture metadata. Only use records labelled
    # as peripheral-driver replacements when the preferred group is empty.
    native = [p for p in eligible if evidence.is_native_capture(p)]
    credited = native or [p for p in eligible if evidence.is_peripheral_capture(p)]

    # Count pages only from the selected eligible records. A failed pixel check
    # still identifies an observed page; its failure is evaluated below.
    missing = set(required) - {p.get("page") for p in credited}
    category = "native" if native else "peripheral" if credited else "none"

    if not credited:
        # Initialization-only traces are retained for inspection, but do not
        # change the category of an independently captured forced display.
        displays = [p for p in pages if p.get("credit", True) is True]
        status = (
            ("forced" if displays and all(p.get("forced") for p in displays) else "bad")
            if pages
            else "none"
        )
    elif not required or missing:
        status = "bad"
    else:
        dialogue = [p for p in credited if not str(p.get("phase", "")).startswith("screen:")]

        # Repeated screen captures may include transient redraws: one verified
        # capture satisfies that screen page. Dialogue captures must all verify.
        screens: dict[int | None, bool] = {}
        for page in credited:
            if str(page.get("phase", "")).startswith("screen:"):
                number = page.get("page")
                screens[number] = screens.get(number, False) or bool(page.get("verified"))

        status = (
            "ok" if all(p.get("verified") for p in dialogue) and all(screens.values()) else "bad"
        )

    return Coverage(status, missing, category)


def evaluate_native_result(
    result: dict[str, Any],
    definition: dict[str, Any],
    english: bool,
    state: evidence.CaptureIdentity,
    required: dict[str, set[int]],
    quick: bool = False,
) -> None:
    """
    This function adds input hashes, task requirements, and the current evaluation to a newly
    recorded task result.

    The supplied result dictionary is modified in place. Saving it is a separate operation.
    english and quick require bool values; numeric zero and one are not accepted.

    Args:
        result (dict[str, Any]): Task result updated in place with ROM and translation-source
            hashes, requirements and evaluation.
        definition (dict[str, Any]): Task requirements, including target(s), expect,
            conditional_expect and original-ROM differences.
        english (bool): True for the English ROM; False for the Japanese original.
        state (evidence.CaptureIdentity): Expected rom_sha256 and per-entry source hashes.
        required (dict[str, set[int]]): Current-ROM required page numbers keyed by entry ID;
            pagination takes precedence over saved expectations.
        quick (bool, optional): Label attached requirements quick-native rather than native;
            this function does not execute playback. Defaults to False.

    Raises:
        TypeError: If english or quick is not a boolean.
        ValueError: If the task definition or required page numbers fail validation.
        KeyError: If a task result lacks pages or the supplied capture identity lacks a required
            hash field.
    """
    if type(english) is not bool or type(quick) is not bool:
        raise TypeError("Boolean options require bool values: english, quick")
    requirements.validate_definition(definition)
    if any(not is_integer(page) or page < 1 for pages in required.values() for page in pages):
        raise ValueError("Required page numbers must be positive integers, not booleans")
    evidence.attach_capture_hashes(dict(routes=[result]), state)
    decision = evaluate_task_result(result, definition, english, state, required)
    result["evaluation"] = decision.record()
    result["requirement_scope"] = "quick-native" if quick else "native"
    result["requirements"] = requirements.resolve_task_requirements(definition)

    # Re-evaluation replaces a stored verdict; do not retain a failure from an
    # earlier evaluation when the current evidence now satisfies the task.
    result.pop("check_failed", None)

    if decision.failed:
        result["check_failed"] = "Required targets or expected pages were not verified"


def has_selected_task_failure(
    result: dict[str, Any],
    definition: dict[str, Any],
    english: bool,
    state: evidence.CaptureIdentity | None = None,
    required_pages: dict[str, set[int]] | None = None,
) -> bool:
    """
    This function returns whether a saved task result fails its execution or required-text checks.

    It calls evaluate_task_result() with the expected ROM and source hashes and required page
    numbers. The result is not modified. The complete Decision from that evaluator provides the
    failure reasons when callers need them.

    english requires a bool value; numeric zero and one are not accepted.

    Args:
        result (dict[str, Any]): Recorded status, pages and optional prerequisite_pages;
            read without mutation.
        definition (dict[str, Any]): Requirements for evaluation, including
            targets, expected pages and declared original-ROM absences.
        english (bool): True checks English pixel evidence; False checks
            Japanese target presence.
        state (evidence.CaptureIdentity | None, optional): English ROM and translation-source hashes; None
            loads the current hashes for English checks. Ignored for Japanese
            checks.
        required_pages (dict[str, set[int]] | None, optional): Current page
            numbers by entry ID; None uses nonconditional expect pairs.

    Returns:
        bool: Whether evaluation reports an execution or required-evidence failure.

    Raises:
        TypeError: If english is not a boolean.
        ValueError: If the task definition or English required page numbers are
            invalid. Also raised if loading the English capture identity encounters
            invalid build-record JSON, source text, metadata, markup or text tokens.
        OSError: If state is omitted for an English check and the current ROM,
            build-input record or translation sources cannot be read.
        KeyError: If loading the English capture identity encounters a source
            character with no game encoding.
    """
    if type(english) is not bool:
        raise TypeError("Boolean options require bool values: english")
    if english and state is None:
        state = evidence.build_capture_identity()

    return evaluate_task_result(result, definition, english, state, required_pages).failed


def get_coverage_exit_code(counts: dict[str, int], partial: bool = False) -> int:
    """
    Convert aggregate entry coverage into a command exit status.

    Invalid captures always fail. Unreached entries also fail unless the caller explicitly
    permits partial coverage for a selected subset. Both bad and none counters must be present
    in the supplied mapping.

    Boolean options require bool values; numeric zero and one are rejected.

    Args:
        counts (dict[str, int]): Coverage counts with bad and none keys, representing invalid
            and unreached entries.
        partial (bool, optional): When True, unreached entries do not fail the selected subset;
            bad entries still fail. Defaults to False.

    Returns:
        int: One for failed entry coverage or disallowed unreached entries; otherwise zero.
            Partial runs permit unrelated unreached entries.

    Raises:
        TypeError: If partial is not a boolean.
        ValueError: If a count is a boolean, noninteger or negative.
        KeyError: If bad is absent, or bad is zero and none is absent.
    """
    if type(partial) is not bool:
        raise TypeError("Boolean options require bool values: partial")
    if any(not is_integer(value) or value < 0 for value in counts.values()):
        raise ValueError("Coverage counts must be nonnegative integers, not booleans")
    return int(bool(counts["bad"] or (counts["none"] and not partial)))


def collect_execution_errors(
    english: Iterable[dict[str, Any]], japanese: Iterable[dict[str, Any]]
) -> list[tuple[str, str, str]]:
    """
    Collect (language, task key, error status) tuples from both result lists.

    Args:
        english (Iterable[dict[str, Any]]): English task results; records with an error-prefixed
            status contribute key and status to the output.
        japanese (Iterable[dict[str, Any]]): Japanese task results filtered by the same
            error-status rule.

    Returns:
        list[tuple[str, str, str]]: Language tag, task key and failed execution status for each
            record whose status starts with error.

    Raises:
        KeyError: If a result with an error-prefixed status lacks its task key.
    """
    return [
        (tag, r["key"], r["status"])
        for tag, routes in (("en", english), ("jp", japanese))
        for r in routes
        if isinstance(r.get("status"), str) and r["status"].startswith("error")
    ]
