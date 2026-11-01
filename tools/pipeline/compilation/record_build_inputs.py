"""
This module saves translation-input hashes for a compiled ROM and finds entries whose current inputs still match.

Records under build/inputs are named by the ROM's SHA-256. Entry hashes cover translation text
and insertion metadata. Entries containing item/opponent-name commands also include a hash of
the active system-name strings, because the widest name determines their reserved layout width.
Translator notes are omitted.

The caller supplies the inputs used for the build. These records do not prove that the ROM
contains those translations or check current font and compiler sources; they compare translation
inputs with the mapping supplied for that ROM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping, cast

if TYPE_CHECKING:
    import pathlib
import hashlib
import json
import os
import re
import tempfile

from core.files.validate_binary_buffers import require_byte_buffer
from core.text import read_translation_files as script_files
from core.text.encode_text import encode_text
from core.text.read_text_tokens import parse_encoded_text


def _validate_digest(value: object, label: str) -> str:
    """
    Validate and lowercase one SHA-256 digest before using it as a filename or hash input.

    Args:
        value (object): Value that must be a string of 64 ASCII hexadecimal digits.
        label (str): Description included in validation errors.

    Returns:
        str: Digest normalized to lowercase hexadecimal.

    Raises:
        TypeError: If value is not a string.
        ValueError: If value is not exactly 64 ASCII hexadecimal digits.
    """
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(f"{label} must be a 64-digit hexadecimal SHA-256 digest")
    return value.lower()


def _validate_input_mapping(value: object) -> dict[str, str]:
    """
    This function checks that a build-input record maps string entry IDs to string hashes and returns a new dictionary.

    Args:
        value (object): Dictionary whose entry IDs and recorded hashes must be strings.
            ID spelling and digest contents are not checked here; matching compares
            the recorded values with hashes computed from loaded source entries.

    Returns:
        dict[str, str]: Independent dictionary with the validated string pairs.

    Raises:
        ValueError: If value is not a dictionary or a key or value is not a string.
    """
    if not isinstance(value, dict):
        raise ValueError("Build-input record must be a JSON object mapping entry IDs to hash strings")
    mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in mapping.items()):
        raise ValueError("Build-input record must be a JSON object mapping entry IDs to hash strings")
    return cast(dict[str, str], dict(mapping))


def hash_name_inputs(entries: Iterable[Mapping[str, object]]) -> str:
    """
    Hash the active system-name strings used to reserve space for runtime insertions.

    The selection contains non-skipped mode-n records with nonempty English text, as in
    the name-width loader. Sorting unique strings makes source order and duplicate
    copies irrelevant: neither changes the widest name measured with a given ROM font.
    This function hashes the strings without measuring them or reading files.

    Args:
        entries (Iterable[Mapping[str, object]]): Records from the system collection,
            all taken from the same loaded source snapshot as the entries being hashed.

    Returns:
        str: SHA-256 of the sorted unique name strings; an empty selection is allowed.

    Raises:
        TypeError: If a selected mode-n record has a non-string en value.
    """
    names: set[str] = set()
    for entry in entries:
        if entry.get("skip") or entry.get("mode") != "n":
            continue
        name = entry.get("en")
        if not isinstance(name, str):
            raise TypeError("System-name English text must be a string")
        if name:
            names.add(name)
    return hashlib.sha256(json.dumps(sorted(names)).encode("utf-8")).hexdigest()


def _uses_system_name(entry: Mapping[str, object]) -> bool:
    """
    Detect item/opponent-name commands without mistaking command operands for opcodes.

    Args:
        entry (Mapping[str, object]): Entry with English source text and an optional
            mode. Menu column markers are removed before encoding; source page
            breaks become line breaks for token parsing. Absent or non-string
            English values contain no detectable commands.

    Returns:
        bool: Whether the encoded text contains a complete F8/12 or F8/18 token.

    Raises:
        KeyError: If a source character has no game encoding.
        ValueError: If brace markup or an encoded token is malformed.
    """
    text = entry.get("en")
    if not isinstance(text, str):
        return False
    if entry.get("mode") == "m":
        text = re.sub(r"@\d+:", "", text)
    encoded = encode_text(text.replace("\f", "\n"))
    return any(token in (b"\xf8\x12", b"\xf8\x18") for token in parse_encoded_text(encoded))


def hash_entry_inputs(entry: Mapping[str, object], *, names_hash: str | None = None) -> str:
    """
    Hash one entry's translation fields and any dependency on runtime system names.

    Args:
        entry (Mapping[str, object]): JSON-serializable entry fields. en, jp, mode, w,
            end, ptrs and imm contribute; missing fields become JSON null. The entry
            ID is the record's mapping key, and translator notes are omitted.
        names_hash (str | None, optional): Digest from hash_name_inputs() for the same
            loaded collections. Required for F8/12 or F8/18 commands; otherwise not
            included in the entry hash. A supplied digest is validated even if unused.

    Returns:
        str: SHA-256 of the entry fields and, where needed, system_names_sha256.

    Raises:
        TypeError: If names_hash is not a string or a contributing field is not
            JSON-serializable.
        ValueError: If a supplied digest is malformed, a runtime-name command lacks
            names_hash, markup/tokens are malformed, or a field contains a JSON cycle.
        KeyError: If a source character has no game encoding.
    """
    if names_hash is not None:
        names_hash = _validate_digest(names_hash, "System-name hash")
    fields = {key: entry.get(key) for key in ("en", "jp", "mode", "w", "end", "ptrs", "imm")}
    if _uses_system_name(entry):
        if names_hash is None:
            raise ValueError("Runtime-name commands require a system-name hash from the same sources")
        fields["system_names_sha256"] = names_hash
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


def hash_active_entries() -> dict[str, str]:
    """
    Hash active entries and their system-name dependencies from one loaded set of collections.

    Returns:
        dict[str, str]: Non-skipped entry IDs mapped to compilation-input SHA-256
            digests. Dependent entries include the shared system-name hash.

    Raises:
        OSError: If a translation source cannot be read.
        ValueError: If source text, UTF-8, metadata, markup or encoded tokens are invalid.
        KeyError: If a source character has no game encoding.
    """
    collections = script_files.load_all()
    names_hash = hash_name_inputs(collections.get("system", []))
    return {
        entry["id"]: hash_entry_inputs(entry, names_hash=names_hash)
        for entries in collections.values()
        for entry in entries
        if not entry.get("skip")
    }


def resolve_input_record_path(rom_hash: str) -> pathlib.Path:
    """
    Return the JSON record path for a compiled ROM's SHA-256.

    Args:
        rom_hash (str): Exactly 64 ASCII hexadecimal digits; case is normalized.

    Returns:
        pathlib.Path: build/inputs/<lowercase ROM hash>.json beneath the repository.

    Raises:
        TypeError: If rom_hash is not a string.
        ValueError: If rom_hash is not exactly 64 ASCII hexadecimal digits.
    """
    return script_files.ROOT / "build/inputs" / (_validate_digest(rom_hash, "ROM hash") + ".json")


def record_build_inputs(rom: bytes | bytearray, inputs: dict[str, str]) -> None:
    """
    Validate and atomically save an entry-hash mapping under the ROM bytes' SHA-256.

    Validation and copying finish before directory creation or temporary writes. The writer
    checks the mapping's shape, not whether it describes the supplied ROM. The caller must
    supply the hashes used to build that image.

    Args:
        rom (bytes | bytearray): Compiled ROM bytes determining the record filename.
        inputs (dict[str, str]): Entry IDs mapped to recorded hash strings, as returned
            by hash_active_entries(). Only string keys and values are accepted.

    Raises:
        TypeError: If rom is not bytes or bytearray.
        ValueError: If inputs is not a dictionary with string keys and values.
        OSError: If the output directory cannot be created, the record cannot be saved,
            or a remaining temporary file cannot be removed.
    """
    require_byte_buffer(rom, (bytes, bytearray))
    validated = _validate_input_mapping(inputs)
    target = resolve_input_record_path(hashlib.sha256(rom).hexdigest())
    target.parent.mkdir(parents=True, exist_ok=True)

    # Replace on the same filesystem so readers cannot observe a partial JSON file.
    scratch: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as stream:
            scratch = stream.name
            json.dump(validated, stream, sort_keys=True)
        os.replace(scratch, target)
    finally:
        # Failures before replacement preserve the existing destination. A leftover
        # temporary file is removed after serialization or replacement errors.
        if scratch is not None and os.path.exists(scratch):
            os.unlink(scratch)


def find_matching_entries(rom_hash: str) -> dict[str, str]:
    """
    Find active entries whose current translation inputs match a ROM's saved record.

    A missing record returns an empty mapping. Invalid JSON and record shapes raise errors.
    Older records without system-name dependencies cannot match dependent entries, so they
    cannot validate those entries using incomplete inputs.

    Args:
        rom_hash (str): SHA-256 of the compiled ROM, as 64 hexadecimal digits.

    Returns:
        dict[str, str]: Current non-skipped entries whose input hashes match the record.

    Raises:
        TypeError: If rom_hash is not a string.
        ValueError: If the ROM digest, record JSON/shape, source UTF-8, metadata,
            markup or encoded tokens are invalid.
        KeyError: If a source character has no game encoding.
        OSError: If an existing record or a translation source cannot be read.
    """
    target = resolve_input_record_path(rom_hash)
    try:
        contents = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    built = _validate_input_mapping(json.loads(contents))
    return {
        entry: value for entry, value in hash_active_entries().items() if built.get(entry) == value
    }
