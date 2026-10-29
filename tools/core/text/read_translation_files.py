"""
This module loads and saves translation entries using editable text files paired with JSON storage and layout metadata.

load() reads translation/<name>.txt and translation/meta/<name>.json, checks
that their entry IDs agree, validates the metadata fields, and returns entry
dictionaries combining Japanese, English, notes, and storage/layout settings.
The collections are events, system, and additions. save() formats both files
before replacing either and uses replace_many() to restore earlier contents
if a later replacement fails. Formatting helpers also return text without
writing files.

The readable file stores one block per original-ROM text offset:

    #### 085EC9
    #JP 「敵生物、活動を停止したぞ。
    #NOTE optional reviewer note
    The enemy creature is inactive.

English occupies the lines after the initial # lines. Blocks are separated
by a blank line before the next #### header. A trailing | protects whitespace
or a literal final | and is removed on load. The source writes page breaks
as \f, carriage returns as \r, and literal backslashes as \\.

JSON entries use the same IDs and hold fields such as end, mode, w, ptrs, imm,
and skip. These specify the original text slot, layout, references to update,
and whether the entry is omitted from translation.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Iterable,
    Literal,
    NotRequired,
    TypeAlias,
    TypedDict,
    cast,
)

if TYPE_CHECKING:
    import os
import json
import re
from pathlib import Path

from core.files import replace_related_files as source_files

# Text compiler mode codes, shared with translation metadata:
# v = wrapped dialogue with three-row pages; p = wrapped prose without pages;
# n = single-line name record; f = fixed text with authored row breaks;
# m = labels at explicit menu columns; c = banner centered within 240 pixels.
TextMode: TypeAlias = Literal["v", "p", "n", "f", "m", "c"]


class TextBlock(TypedDict):
    """
    Source lines collected under one translation entry header.

    Attributes:
        id (str): Entry identifier taken from the header.
        lines (list[str]): Following source lines, including the blank separator.
        line (int): One-based header line number used in parse errors.
    """

    id: str
    lines: list[str]
    line: int


class TextReferences(TypedDict):
    """
    Original text location and the references that can redirect it.

    Attributes:
        id (str): Hexadecimal original-ROM text offset.
        ptrs (list[str]): Optional hexadecimal offsets of 16-bit text pointers.
        imm (list[str]): Optional three distinct low/high/bank address-operand file offsets.
    """

    id: str
    ptrs: NotRequired[list[str]]
    imm: NotRequired[list[str]]


class TranslationEntry(TextReferences):
    """
    Source text and technical metadata consumed by translation validation.

    Required fields identify the original storage range and both language texts.
    Optional fields describe layout and intentional departures from source rules.
    Other collection metadata may coexist with these fields in loaded entries.

    Attributes:
        id (str): Hexadecimal original-ROM text offset, without a prefix.
        end (str): Hexadecimal exclusive end of the original text slot.
        jp (str): Japanese source text; empty for patch-authored entries.
        en (str): English text, including any command or menu-column markup.
        note (str): Optional translator-facing explanation stored with the text.
        mode (TextMode): Layout strategy; omission selects dialogue mode v.
        w (int): Explicit width in pixels, or character cells for menu mode.
        max_lines (int): Visible row allowance for prose or centered banners.
        skip (bool | str): Whether to omit the entry, optionally with a reason.
        menu_columns (list[list[int]]): Expected column positions for each row.
        fixed_line_bytes (list[int]): Maximum encoded byte count for each row.
        reviewed_controls (list[str]): Expected English brace-delimited commands,
            including duplicates; overrides comparison with Japanese commands.
        max_name_bytes (int): Encoded name capacity, overriding eight bytes.
        name_spaces (bool): Whether a name may contain spaces.
        ptrs (list[str]): Hexadecimal offsets of retargetable 16-bit pointers.
        imm (list[str]): Three distinct hexadecimal offsets of address bytes
            stored as CPU instruction operands, in low/high/bank order.
    """

    end: str
    jp: str
    en: str
    note: NotRequired[str]
    mode: NotRequired[TextMode]
    w: NotRequired[int]
    max_lines: NotRequired[int]
    skip: NotRequired[bool | str]
    menu_columns: NotRequired[list[list[int]]]
    fixed_line_bytes: NotRequired[list[int]]
    reviewed_controls: NotRequired[list[str]]
    max_name_bytes: NotRequired[int]
    name_spaces: NotRequired[bool]


# Repository root for the default translation directory and built-ROM
# lookups used by the text tooling.
ROOT = Path(__file__).resolve().parents[3]

# Collection basenames used to pair translation/<name>.txt with meta/<name>.json.
NAMES: tuple[Literal["events", "system", "additions"], ...] = ("events", "system", "additions")

# Human-readable fields stored in the text file; other entry fields go into metadata.
TEXT_FIELDS = ("jp", "en", "note")

# Entry-block prefix followed by the text's hexadecimal original-ROM offset.
HEADER = "#### "

# Collection descriptions written into readable translation-file headers.
TITLES = {
    "events": "story dialogue",
    "system": "menus, names, descriptions and other system text",
    "additions": "patch-authored lines (not original Japanese)",
}


def resolve_collection_paths(
    name: Literal["events", "system", "additions"], folder: str | os.PathLike[str] | None = None
) -> tuple[Path, Path]:
    """
    Locate a collection's readable text and technical metadata.

    Args:
        name (Literal['events', 'system', 'additions']): Collection basename
            without an extension.
        folder (str | os.PathLike[str] | None, optional): Translation
            root containing text files and meta/. None uses the repository
            translation directory.

    Returns:
        tuple[Path, Path]: Readable text path and its meta/<name>.json
            path.

    Raises:
        ValueError: The collection name is not events, system or additions.
    """
    if name not in NAMES:
        raise ValueError(f"Unknown translation collection: {name!r}")
    folder = Path(folder or ROOT / "translation")
    return folder / f"{name}.txt", folder / "meta" / f"{name}.json"


def _protect(line: str) -> str:
    """
    Append an escape marker when trailing whitespace or a literal bar needs protection.

    Args:
        line (str): One source line without its newline; an empty line is
            allowed.

    Returns:
        str: Original line, plus one final | when protection is needed.
    """
    return line + "|" if line[-1:].isspace() or line.endswith("|") else line


def _unprotect(line: str) -> str:
    """
    Remove one final escape marker.

    Args:
        line (str): Serialized source line; a final | is removed even when no
            whitespace precedes it.

    Returns:
        str: Decoded line with at most one final marker removed.
    """
    return line[:-1] if line.endswith("|") else line


def _encode_source_lines(text: str) -> list[str]:
    """
    Escape backslashes, carriage returns and page breaks, then split physical lines.

    Args:
        text (str): Decoded text whose newlines remain physical line breaks;
            backslashes, carriage returns and form feeds are escaped.

    Returns:
        list[str]: Source lines with doubled backslashes, backslash-r carriage
            returns and backslash-f page breaks.
    """
    return text.replace("\\", "\\\\").replace("\r", "\\r").replace("\f", "\\f").split("\n")


def _decode_source_lines(lines: Iterable[str]) -> str:
    """
    Decode protected readable lines into dialogue text.

    Args:
        lines (Iterable[str]): Serialized lines with optional trailing | markers
            and escaped backslashes, carriage returns and page breaks.

    Returns:
        str: Unprotected lines joined by newlines, with the supported escapes
            decoded. Other backslash sequences are retained literally.
    """
    # Decode each escape once. A doubled backslash before f represents the two
    # literal characters, not a backslash followed by another page-break escape.
    escapes = {"\\": "\\", "r": "\r", "f": "\f"}
    return re.sub(
        r"\\([\\fr])",
        lambda match: escapes[match.group(1)],
        "\n".join(_unprotect(l) for l in lines),
    )


def format_text(
    name: Literal["events", "system", "additions"], entries: Iterable[TranslationEntry]
) -> str:
    """
    Serialize editable translation blocks without changing controls.

    Args:
        name (Literal['events', 'system', 'additions']): Collection basename
            without an extension.
        entries (Iterable[TranslationEntry]): Translation entries with id, jp, en, optional
            note and technical metadata.

    Returns:
        str: Editable text blocks, ready for the caller to encode as UTF-8.

    Raises:
        AssertionError: An English line starts with # and would conflict with
            the readable file format.
        ValueError: An active translation is blank and could not be loaded again.
    """
    out = [
        f"# Solid Runner English script: {TITLES[name]}.",
        '# Edit only the English under each "#### ID". #JP lines show the original Japanese;',
        '# #NOTE lines are reviewer notes. A trailing "|" protects trailing spaces. See docs/EDITING.md.',
        "",
    ]

    # Keep source and notes in tagged lines, followed by the editable English
    # payload. The empty line after each entry is its required separator.
    for e in entries:
        if not e.get("skip") and not e["en"].strip():
            raise ValueError(f"{e['id']}: English translation must not be blank")
        out.append(HEADER + e["id"])
        out += ["#JP " + _protect(l) if l else "#JP" for l in _encode_source_lines(e["jp"])]

        if "note" in e:
            out += [
                "#NOTE " + _protect(l) if l else "#NOTE" for l in _encode_source_lines(e["note"])
            ]

        skipped = e.get("skip")
        if skipped:
            reason = skipped if isinstance(skipped, str) else "not text"
            # A reason lives in metadata; its readable copy is only a comment.
            # Prefix every physical line so embedded headers cannot create blocks.
            out.extend(f"# skipped, not translatable: {line}" for line in reason.splitlines())

        # Protect trailing spaces and structural markers without changing the
        # in-game text they represent.
        en = _encode_source_lines(e["en"])
        if not (not any(l.startswith("#") for l in en)):
            raise AssertionError((e["id"], "English line starts with #"))

        out += [_protect(l) for l in en]
        out.append("")

    return "\n".join(out) + "\n"


def parse_text(text: str, where: str = "script") -> list[dict[str, str]]:
    """
    Parse readable blocks into id/jp/en/note dictionaries.

    Args:
        text (str): Readable translation-file contents containing ID, JP, EN
            and optional NOTE blocks.
        where (str, optional): Source label included in malformed-block
            diagnostics.

    Returns:
        list[dict[str, str]]: Parsed text fields in source order. The caller joins
            technical metadata separately using each entry ID.

    Raises:
        ValueError: Content precedes the first ID header or an entry header
            lacks its required blank separator.
    """
    # Normalize physical file line endings before parsing; encoded in-game
    # line and page breaks are decoded later from each text field.
    text = text.replace("\r\n", "\n")
    if text.endswith("\n"):
        text = text[:-1]

    blocks: list[TextBlock] = []
    current: TextBlock | None = None

    # Collect entry blocks before interpreting their #JP/#NOTE fields. A leading
    # # outside a block is a file comment, not translated text.
    for number, line in enumerate(text.split("\n"), 1):
        if line.startswith(HEADER):
            current = {"id": line[len(HEADER) :].strip(), "lines": [], "line": number}
            blocks.append(current)
        elif current is None:
            if line and not line.startswith("#"):
                raise ValueError(f'{where}:{number}: text before the first "#### ID" header')
        else:
            current["lines"].append(line)

    # Parse collected blocks independently so header locations remain
    # available for errors and each entry keeps its own English payload.
    entries: list[dict[str, str]] = []
    for b in blocks:
        lines = b["lines"]

        # Remove exactly the required block separator. Further blank lines belong
        # to the translation and must survive the round trip.
        if lines and lines[-1] == "":
            lines = lines[:-1]
        elif b is not blocks[-1]:
            raise ValueError(
                f"{where}:{b['line']}: entry {b['id']} must be followed by a blank line"
            )

        fields: dict[str, list[str]] = {"jp": [], "note": []}
        k = 0

        # Only the leading tagged lines describe Japanese text and translator notes;
        # the remaining lines form the English payload.
        while k < len(lines) and lines[k].startswith("#"):
            for tag, field in (("#JP", "jp"), ("#NOTE", "note")):
                if lines[k] == tag or lines[k].startswith(tag + " "):
                    fields[field].append(lines[k][len(tag) + 1 :])

            k += 1

        # Decode protected text only after separating tagged source fields from
        # English, so protected markers are not mistaken for file syntax.
        entry: dict[str, str] = {
            "id": b["id"],
            "jp": _decode_source_lines(fields["jp"]),
            "en": _decode_source_lines(lines[k:]),
        }
        if fields["note"]:
            entry["note"] = _decode_source_lines(fields["note"])

        entries.append(entry)

    return entries


def format_meta(entries: Iterable[TranslationEntry]) -> str:
    """
    Serialize technical entry fields in their original order.

    Args:
        entries (Iterable[TranslationEntry]): Translation entries with id, jp, en, optional
            note and technical metadata.

    Returns:
        str: JSON object text mapping entry IDs to technical metadata, ready
            for the caller to encode as UTF-8.

    Raises:
        ValueError: Entries repeat an ID or contain invalid technical metadata.
    """
    rows: list[str] = []
    seen: set[int] = set()
    for e in entries:
        meta = {k: v for k, v in e.items() if k not in TEXT_FIELDS and k != "id"}
        _validate_metadata(e["id"], meta)
        offset = int(e["id"], 16)
        if offset in seen:
            raise ValueError(f"Duplicate translation entry offset: {e['id']}")
        seen.add(offset)
        rows.append(f"{json.dumps(e['id'])}: {json.dumps(meta, ensure_ascii=False)}")

    return "{\n" + ",\n".join(rows) + "\n}\n"


def _read_unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """
    Decode a JSON object without silently replacing repeated fields.

    Args:
        pairs (list[tuple[str, object]]): Object members in JSON source order.

    Returns:
        dict[str, object]: Members indexed by their unique names.

    Raises:
        ValueError: An object repeats a member name.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata key: {key}")
        result[key] = value
    return result


def _validate_metadata(entry_id: str, metadata: dict[str, object]) -> None:
    """
    Validate storage and layout fields before they can direct compiler writes.

    Unknown supplemental fields remain available to other tools. Readable text
    and the entry ID belong to the text file and cannot be overridden here.

    Args:
        entry_id (str): Hexadecimal unheadered ROM offset from the text header.
        metadata (dict[str, object]): Technical fields from the matching JSON object.

    Raises:
        ValueError: A known field has an invalid type, range or structure.
    """

    def offset(value: object) -> int:
        """
        Parse a hexadecimal file offset within the four-megabyte LoROM image.

        Args:
            value (object): JSON value expected to contain unprefixed hexadecimal digits.

        Returns:
            int: Checked file offset.

        Raises:
            ValueError: The value is not a hexadecimal ROM offset.
        """
        if (
            not isinstance(value, str)
            or not value
            or any(c not in "0123456789abcdefABCDEF" for c in value)
        ):
            raise ValueError(f"{entry_id}: invalid hexadecimal offset {value!r}")
        result = int(value, 16)
        if result >= 0x400000:
            raise ValueError(f"{entry_id}: offset outside the four-megabyte ROM")
        return result

    start = offset(entry_id)
    end = offset(metadata.get("end"))
    if end < start:
        raise ValueError(f"{entry_id}: end precedes the entry start")
    if any(key in metadata for key in ("id", *TEXT_FIELDS)):
        raise ValueError(f"{entry_id}: metadata cannot override text fields or the entry ID")
    if "mode" in metadata and metadata["mode"] not in ("v", "p", "n", "f", "m", "c"):
        raise ValueError(f"{entry_id}: unsupported text mode")

    # bool is an int subclass; accepting it as a width would create a one-pixel
    # layout instead of reporting the malformed JSON value.
    for key in ("w", "max_lines", "max_name_bytes"):
        if key in metadata and (type(metadata[key]) is not int or cast(int, metadata[key]) <= 0):
            raise ValueError(f"{entry_id}: {key} must be a positive integer")
    if "skip" in metadata and not isinstance(metadata["skip"], (bool, str)):
        raise ValueError(f"{entry_id}: skip must be a boolean or reason string")
    if "name_spaces" in metadata and type(metadata["name_spaces"]) is not bool:
        raise ValueError(f"{entry_id}: name_spaces must be a boolean")

    for key in ("ptrs", "imm", "reviewed_controls", "fixed_line_bytes", "menu_columns"):
        if key not in metadata:
            continue
        raw = metadata[key]
        if not isinstance(raw, list):
            raise ValueError(f"{entry_id}: {key} must be a list")
        values = cast(list[object], raw)
        if key in ("ptrs", "imm"):
            if key == "imm" and len(values) not in (0, 3):
                raise ValueError(f"{entry_id}: imm must contain low/high/bank operand offsets")
            for value in values:
                at = offset(value)
                if key == "ptrs" and at == 0x3FFFFF:
                    raise ValueError(f"{entry_id}: two-byte pointer extends beyond the ROM")
            if key == "imm" and values and len({offset(value) for value in values}) != 3:
                raise ValueError(f"{entry_id}: imm must identify three distinct operand bytes")
        elif key == "reviewed_controls":
            if any(not isinstance(value, str) for value in values):
                raise ValueError(f"{entry_id}: reviewed_controls must contain strings")
        else:
            rows: list[list[object]] = [values]
            if key == "menu_columns":
                if any(not isinstance(row, list) for row in values):
                    raise ValueError(f"{entry_id}: menu_columns must contain row lists")
                rows = cast(list[list[object]], values)
            if any(type(value) is not int or value < 0 for row in rows for value in row):
                raise ValueError(f"{entry_id}: {key} must contain nonnegative integers")


def load(
    name: Literal["events", "system", "additions"], folder: str | os.PathLike[str] | None = None
) -> list[TranslationEntry]:
    """
    Read and merge a collection's UTF-8 text and technical metadata.

    Args:
        name (Literal['events', 'system', 'additions']): Collection basename
            without an extension.
        folder (str | os.PathLike[str] | None, optional): Translation
            root containing text files and meta/. None uses the repository
            translation directory.

    Returns:
        list[TranslationEntry]: Entries in source order, with technical metadata joined
            to readable text by entry ID.

    Raises:
        ValueError: Text IDs or metadata fields are malformed, duplicated or inconsistent, or an active translation is blank.
    """
    text_path, meta_path = resolve_collection_paths(name, folder)
    texts = parse_text(text_path.read_text(encoding="utf-8"), text_path.name)
    raw: object = json.loads(
        meta_path.read_text(encoding="utf-8"), object_pairs_hook=_read_unique_object
    )
    if not isinstance(raw, dict):
        raise ValueError(f"{meta_path.name}: metadata must be an object keyed by entry ID")
    meta = cast(dict[str, object], raw)
    ids = [e["id"] for e in texts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{text_path.name}: duplicate entry IDs")

    # Readable text and technical metadata must identify exactly the same entries;
    # otherwise joining them could silently lose text or attach the wrong layout.
    if set(ids) != set(meta):
        missing, extra = sorted(set(meta) - set(ids)), sorted(set(ids) - set(meta))
        raise ValueError(
            f"{text_path.name}: IDs differ from {meta_path.name}; missing {missing[:5]}, unknown {extra[:5]}"
        )

    checked: dict[str, dict[str, object]] = {}
    for entry_id, value in meta.items():
        if not isinstance(value, dict):
            raise ValueError(f"{entry_id}: metadata must be an object")
        fields = cast(dict[str, object], value)
        _validate_metadata(entry_id, fields)
        checked[entry_id] = fields

    offsets = [int(entry_id, 16) for entry_id in ids]
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"{text_path.name}: entry IDs alias the same ROM offset")

    # Join on stable entry IDs, allowing text fields to come from the editable
    # text file while layout and pointer fields come from its metadata file.
    entries = cast(
        list[TranslationEntry],
        [
            {"id": e["id"], **checked[e["id"]], **{k: e[k] for k in TEXT_FIELDS if k in e}}
            for e in texts
        ],
    )
    for e in entries:
        if not e.get("skip") and not e["en"].strip():
            raise ValueError(
                f"{text_path.name} #### {e['id']}: English translation must not be blank"
            )

    return entries


def load_all(folder: str | os.PathLike[str] | None = None) -> dict[str, list[TranslationEntry]]:
    """
    Read every configured translation collection and require unique ROM entry offsets.

    Args:
        folder (str | os.PathLike[str] | None, optional): Translation
            root containing text files and meta/. None uses the repository
            translation directory.

    Returns:
        dict[str, list[TranslationEntry]]: Ordered entries for every configured collection.

    Raises:
        ValueError: Two collections assign entries to the same ROM offset.
    """
    collections: dict[str, list[TranslationEntry]] = {name: load(name, folder) for name in NAMES}
    owners: dict[int, str] = {}
    for name, entries in collections.items():
        for entry in entries:
            offset = int(entry["id"], 16)
            if offset in owners:
                raise ValueError(f"Entry {entry['id']} appears in both {owners[offset]} and {name}")
            owners[offset] = name
    return collections


def save(
    name: Literal["events", "system", "additions"],
    entries: Iterable[TranslationEntry],
    folder: str | os.PathLike[str] | None = None,
) -> None:
    """
    Write readable text and metadata together through the file transaction helper.

    Formats both files before either is replaced. Failed writes follow
    replace_many rollback and recovery behavior.

    Args:
        name (Literal['events', 'system', 'additions']): Collection basename
            without an extension.
        entries (Iterable[TranslationEntry]): Translation entries with id, jp, en, optional
            note and technical metadata.
        folder (str | os.PathLike[str] | None, optional): Translation
            root containing text files and meta/. None uses the repository
            translation directory.

    Raises:
        OSError: Staging or replacement fails; the related-file transaction
            attempts rollback.
        ValueError: An active translation is blank, or its technical metadata is invalid.
        AssertionError: An English line starts with the reserved # comment prefix.
    """
    # Both formatters must see the same entries, including one-shot iterators.
    entries = list(entries)
    contents = (format_text(name, entries).encode("utf-8"), format_meta(entries).encode("utf-8"))
    text_path, meta_path = resolve_collection_paths(name, folder)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    source_files.replace_many(zip((text_path, meta_path), contents))


def hash_collection_files(
    name: Literal["events", "system", "additions"], folder: str | os.PathLike[str] | None = None
) -> str:
    """
    Fingerprint both files that together define a translation collection.

    The readable text bytes precede the technical JSON bytes in the SHA-256 input.
    This detects changes to source formatting as well as entry content.

    Args:
        name (Literal['events', 'system', 'additions']): Collection basename
            selecting the text and meta JSON files.
        folder (str | os.PathLike[str] | None, optional): Directory
            containing <name>.txt and meta/<name>.json; None uses the
            repository translation directory.

    Returns:
        str: Hexadecimal SHA-256 of readable text bytes followed by metadata bytes.
    """
    import hashlib

    return hashlib.sha256(
        b"".join(p.read_bytes() for p in resolve_collection_paths(name, folder))
    ).hexdigest()


def load_names_for_dialogue_layout(folder: str | os.PathLike[str] | None = None) -> list[str]:
    """
    Load translated names used to reserve space for names inserted into dialogue.

    Some dialogue commands insert an item or opponent name during gameplay.
    The chosen name can vary, so word wrapping needs room for the widest name
    that could appear. This function supplies the names for that calculation.

    Read system.txt and meta/system.json from the translation directory. Select
    every entry marked mode n, which identifies a name record. Exclude entries
    whose skip field is true or contains an exclusion reason, and entries with
    no English text. Every remaining name record contributes to the candidate
    list. The complete source collection is validated before this selection.

    Pass the returned strings to measure_text.measure_name_width(rom, names).
    Pass that pixel width as name_width to the layout or insertion function.
    This function reads source files; pixel measurement uses the candidate ROM's
    font through measure_name_width().

    Args:
        folder (str | os.PathLike[str] | None, optional): Directory containing
            system.txt and meta/system.json. None uses the repository's
            translation/ directory.

    Returns:
        list[str]: English text from the selected name records, in source order.
            Duplicate strings are retained. An empty list means no records met
            the selection rules; measure_name_width() rejects that empty list.

    Raises:
        ValueError: Readable text or JSON metadata is malformed, their entry IDs
            disagree, or a non-skipped translation has blank English text.
        OSError: Either source file cannot be read.
    """
    # Name records share the system collection with prose and interface text.
    # Mode n selects the strings copied as names. Retain every eligible name
    # because the item/opponent chosen during gameplay is not yet known.
    return [
        entry["en"]
        for entry in load("system", folder)
        if entry.get("mode") == "n" and entry.get("en") and not entry.get("skip")
    ]
