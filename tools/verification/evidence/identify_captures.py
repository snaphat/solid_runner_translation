"""
This module identifies the ROM and translation sources behind a capture and decides whether it can count toward current verification coverage.

CaptureIdentity contains the ROM SHA-256 and a sources mapping from translation IDs to source
hashes. Page records carry both hashes. Only entries matching the compiled ROM's input record
receive a current source hash, preventing newer source text from being verified against an older
build.

Category checks distinguish ordinary native captures, forced diagnostics and peripheral
fixtures. Eligibility checks the credit flag, matching hashes and permitted categories;
page-pair collection additionally requires successful pixel verification.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, TypedDict

from core.files.validate_binary_buffers import require_byte_buffer
from core.text import read_translation_files as script_files
from core.validation.validate_integers import is_integer
from pipeline.compilation import record_build_inputs as build_state


class CaptureIdentity(TypedDict):
    """
    ROM hash and per-entry translation-source hashes matching its build-input record.

    Attributes:
        rom_sha256 (str): SHA-256 of the complete ROM image used for captures.
        sources (dict[str, str]): Entry IDs mapped to hashes of English text,
            layout mode and width, after matching their compilation inputs to
            this ROM's build record.
    """

    rom_sha256: str
    sources: dict[str, str]


def _has_valid_category_flags(page: dict[str, Any]) -> bool:
    """
    Return whether category flags use supported boolean or explanation values.

    Args:
        page (dict[str, Any]): Capture with optional forced and peripheral flags.
            Missing values and None mean no flag; nonempty strings retain
            diagnostic explanations. Numeric values are invalid.

    Returns:
        bool: Whether both flags are absent, None, booleans or strings.
    """
    return all(page.get(name) is None or isinstance(page.get(name), (bool, str))
               for name in ("forced", "peripheral"))


def is_peripheral_capture(page: dict[str, Any]) -> bool:
    """
    Return whether a capture uses a peripheral fixture.

    The function recognizes a true peripheral flag, a nonempty explanation string,
    or the Turbo File fixture label. Numeric flags do not identify a peripheral.

    Args:
        page (dict[str, Any]): Capture whose peripheral flag or fixture label identifies
            a stand-in device.

    Returns:
        bool: Whether the capture records an explicit peripheral flag or the
            supported Turbo File fixture label.
    """
    flag = page.get("peripheral")
    return (
        flag is True
        or isinstance(flag, str) and bool(flag)
        or "Turbo File stand-in driver" in str(page.get("fixture", ""))
    )


def is_native_capture(page: dict[str, Any]) -> bool:
    """
    Return whether a capture is credited, non-forced and non-peripheral.

    Matching ROM and translation hashes and pixel verification are checked separately.

    Args:
        page (dict[str, Any]): Capture with credit, forced, peripheral and fixture
            fields; missing credit permits ordinary coverage.

    Returns:
        bool: Whether ordinary native coverage is permitted by the capture's
            credit and evidence-category fields. Numeric category flags are rejected.
            ROM hashes, translation hashes and
            pixel placement are checked separately.
    """
    return (
        _has_valid_category_flags(page)
        and page.get("credit", True) is True
        and not page.get("forced")
        and not is_peripheral_capture(page)
    )


def hash_capture_source(entry: Mapping[str, object]) -> str:
    """
    Hash an entry's English text and layout settings.

    This identifies the text and layout settings used by a capture. It does not
    prove that an entry was compiled into a ROM; build_capture_identity checks the broader
    compilation-input record before including this hash.

    Args:
        entry (Mapping[str, object]): Translation entry; only en, mode and w are hashed, with
            missing values represented as null.

    Returns:
        str: Hexadecimal SHA-256 of the English text, layout mode and width.

    Raises:
        TypeError: If a selected entry field is not JSON-serializable.
        ValueError: If a selected entry field contains a JSON cycle.
    """
    fields = {k: entry.get(k) for k in ("en", "mode", "w")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


def build_capture_identity(rom: bytes | bytearray | None = None) -> CaptureIdentity:
    """
    Compute ROM and translation-source hashes for entries matching the build-input record.

    Only entries matching the build-input JSON record for this ROM hash are included. Pixel
    verification can therefore reject captures after source edits even when their ROM digest
    still matches the present build. Entries and their system-name dependencies are rechecked
    after the saved-record lookup; entries skipped during that second read are excluded.

    Args:
        rom (bytes | bytearray | None, optional): ROM bytes to identify; None reads the
            current English build.

    Returns:
        CaptureIdentity: rom_sha256 and sources mapping entry IDs to capture-source
            hashes. A missing build-input JSON record produces an empty sources mapping,
            not presumed matches to the current translations.

    Raises:
        OSError: If the default ROM, build-input record, or a translation source cannot
            be read.
        ValueError: If the record JSON/shape, source UTF-8, metadata, markup or
            encoded text tokens are invalid.
        KeyError: If a source character has no game encoding.
        TypeError: If rom is neither bytes nor bytearray.
    """
    if rom is None:
        rom = (script_files.ROOT / "build/Solid Runner (English).sfc").read_bytes()

    require_byte_buffer(rom, (bytes, bytearray))
    digest = hashlib.sha256(rom).hexdigest()

    # Read again after matching the saved record. An intervening edit can change
    # an entry's text, skip flag or the names used to reserve its layout width.
    built = build_state.find_matching_entries(digest)
    collections = script_files.load_all()
    names_hash = build_state.hash_name_inputs(collections.get("system", []))
    return {
        "rom_sha256": digest,
        "sources": {
            entry["id"]: hash_capture_source(entry)
            for entries in collections.values()
            for entry in entries
            if not entry.get("skip")
            and built.get(entry["id"])
            == build_state.hash_entry_inputs(entry, names_hash=names_hash)
        },
    }


def is_current_capture(page: dict[str, Any], state: CaptureIdentity) -> bool:
    """
    Return whether ROM and nonmissing entry-source hashes match state.

    Args:
        page (dict[str, Any]): Capture carrying entry, rom_sha256 and source_sha256; a missing
            source hash cannot match.
        state (CaptureIdentity): Expected rom_sha256 and sources mapping entry IDs to source
            hashes.

    Returns:
        bool: Whether both the ROM digest and a present entry-source digest match the expected
            ROM and translation-source hashes.

    Raises:
        KeyError: If capture comparison reaches a required rom_sha256 or sources lookup but that
            field is absent from state.
    """
    entry = page.get("entry")
    if not isinstance(entry, str) or not entry:
        return False

    # Require both ROM and translation-source hashes; matching two missing hashes must
    # never make an unidentified capture current.
    expected_source = state["sources"].get(entry)
    return (
        page.get("rom_sha256") == state["rom_sha256"]
        and page.get("source_sha256") == expected_source
        and page.get("source_sha256") is not None
    )


def is_eligible_capture(
    page: dict[str, Any],
    state: CaptureIdentity,
    *,
    allow_forced: bool = False,
    allow_peripheral: bool = False,
) -> bool:
    """
    Check that a capture matches the current ROM and translation and uses an allowed capture
    category.

    Boolean options require bool values; numeric zero and one are rejected.

    Args:
        page (dict[str, Any]): Capture record.
        state (CaptureIdentity): Expected ROM and per-entry translation-source hashes from
            build_capture_identity().
        allow_forced (bool, optional): Permit forced diagnostic captures. Defaults to False.
        allow_peripheral (bool, optional): Permit peripheral-fixture captures. Defaults to
            False.

    Returns:
        bool: True for a credited, current capture with a positive integer page and boolean
            verdict in an allowed category. Numeric category flags are rejected. Pixel success
            is checked separately.

    Raises:
        TypeError: If allow_forced or allow_peripheral is not a boolean.
        KeyError: If capture comparison reaches a required rom_sha256 or sources lookup but that
            field is absent from state.
    """
    if type(allow_forced) is not bool or type(allow_peripheral) is not bool:
        raise TypeError("Boolean options require bool values: allow_forced, allow_peripheral")
    return (
        _has_valid_category_flags(page)
        and (is_integer(page.get("page")))
        and page["page"] > 0
        and type(page.get("verified")) is bool
        and page.get("credit", True) is True
        and (allow_forced or not page.get("forced"))
        and (allow_peripheral or not is_peripheral_capture(page))
        and is_current_capture(page, state)
    )


def collect_verified_page_pairs(
    pages: Iterable[dict[str, Any]],
    state: CaptureIdentity,
    *,
    allow_forced: bool = False,
    allow_peripheral: bool = False,
) -> set[tuple[str, int]]:
    """
    Collect current, pixel-verified (entry ID, one-based page) pairs.

    Boolean options require bool values; numeric zero and one are rejected.

    Args:
        pages (Iterable[dict[str, Any]]): Capture records, including possible duplicates.
        state (CaptureIdentity): Expected ROM and per-entry translation-source hashes from
            build_capture_identity().
        allow_forced (bool, optional): Permit forced diagnostic captures. Defaults to False.
        allow_peripheral (bool, optional): Permit peripheral-fixture captures. Defaults to
            False.

    Returns:
        set[tuple[str, int]]: Distinct pairs passing is_eligible_capture() and verified checks.

    Raises:
        TypeError: If allow_forced or allow_peripheral is not a boolean.
        KeyError: If capture comparison reaches a required rom_sha256 or sources lookup but that
            field is absent from state.
    """
    if type(allow_forced) is not bool or type(allow_peripheral) is not bool:
        raise TypeError("Boolean options require bool values: allow_forced, allow_peripheral")
    return {
        (p["entry"], p["page"])
        for p in pages
        if p.get("entry")
        and p.get("page")
        and p.get("verified") is True
        and is_eligible_capture(
            p, state, allow_forced=allow_forced, allow_peripheral=allow_peripheral
        )
    }


def attach_capture_hashes(data: dict[str, Any], state: CaptureIdentity) -> dict[str, Any]:
    """
    This function labels newly recorded captures with their ROM and translation-source hashes,
    modifying the supplied result batch.

    It overwrites hashes on opening, task, and prerequisite captures. The supplied state must
    describe the inputs used to record those captures; applying current hashes to older captures
    would not validate them. Entries absent from state.sources receive source_sha256=None and
    cannot match current sources.

    Args:
        data (dict[str, Any]): Mutable result batch with opening and routes; capture rom_sha256
            and source_sha256 fields are added in place.
        state (CaptureIdentity): Expected rom_sha256 and per-entry source hashes.

    Returns:
        dict[str, Any]: The same batch object with ROM and translation-source hashes attached to
            capture records.

    Raises:
        KeyError: If a route lacks pages, or a capture needs a hash field absent from state.
    """
    # Attach hashes to opening, task and prerequisite captures consistently; prerequisites
    # are reused across tasks and need the same stale-evidence protection.
    for page in data.get("opening", []) + [
        p for r in data.get("routes", []) for p in r["pages"] + r.get("prerequisite_pages", [])
    ]:
        page["rom_sha256"] = state["rom_sha256"]
        entry = page.get("entry")
        page["source_sha256"] = state["sources"].get(entry) if entry is not None else None

    return data
