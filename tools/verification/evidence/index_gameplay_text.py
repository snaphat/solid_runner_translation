"""
This module builds lookup tables that identify translation entries and pages from captured gameplay text addresses.

Page-start and terminator indexes associate compiled-ROM addresses with entry
IDs and one-based page numbers. Slot ranges identify pointers inside original
translation slots; the item-name index identifies dynamic item-name insertions.
The builders load active translation records, and item lookup also reads the
hash-checked Japanese ROM to recover the original item-name references.

Binary text decoding and reference resolution belong to
core.text.read_compiled_text. These helpers build capture-identification indexes;
they do not emulate the game, verify screenshot pixels, or change ROM bytes.
"""

from __future__ import annotations

from core.rom import load_original_rom as original_rom
from core.rom.convert_addresses import resolve_bank_pointer
from core.text import read_translation_files as script_files
from core.text.read_compiled_text import (
    decode_literal_glyphs,
    find_entry_text_starts,
    iter_dialogue_pages,
    resolve_text_start,
)


def _index_entry_starts(rom: bytes | bytearray) -> dict[int, str]:
    """
    Assign original text slots before adding reference aliases.

    Load active entries from every translation collection. Register all original
    slot starts first so an alias cannot take another entry's address, regardless
    of source order. Aliases to otherwise unowned addresses retain the first
    referring entry in collection order. Both page indexes use this ownership map.

    Args:
        rom (bytes | bytearray): Compiled ROM used to resolve original slot
            starts and stored pointer/immediate references.

    Returns:
        dict[int, str]: Text-start file offsets mapped to their owning entry IDs.

    Raises:
        OSError: A translation source cannot be read.
        ValueError: Source metadata or a stored reference cannot be resolved.
        IndexError: An original slot or reference operand lies outside rom.
    """
    entries = [
        e
        for name in script_files.NAMES
        for e in script_files.load(name)
        if not e.get("skip") and e["en"]
    ]
    # Reserve every actual slot before processing aliases, including the
    # computer slot whose leading show-text opcode is not part of its text.
    owners = {resolve_text_start(rom, int(e["id"], 16)): e["id"] for e in entries}
    for entry in entries:
        for first in find_entry_text_starts(rom, entry):
            owners.setdefault(first, entry["id"])
    return owners


def index_page_starts(rom: bytes | bytearray) -> dict[int, tuple[str, int]]:
    """
    Map compiled text-start offsets to their owning entry and dialogue page.

    The index includes original slots and retargeted pointer/immediate starts,
    following relocated text and page continuations. Gameplay trace readers use
    it to identify which translated page the native text engine has started.
    Active source records are loaded from every configured collection on each
    call. Original slot ownership takes priority over aliases, and a separately
    registered entry start retains page number one.

    Args:
        rom (bytes | bytearray): Compiled ROM whose dialogue chains are
            indexed.

    Returns:
        dict[int, tuple[str, int]]: Text-start offset to (entry ID, one-based page
            number).

    Raises:
        OSError: A translation source file cannot be read.
        ValueError: Source metadata or a decoded text chain is invalid.
        IndexError: An entry, operand, text token, or jump target is outside rom.
        KeyError: A decoded page contains an unsupported F8 command.
    """
    own = _index_entry_starts(rom)
    starts = {at: (i, 1) for at, i in own.items()}
    for at, entry in own.items():
        # A registered entry start owns itself, even when another page flows into it.
        for page, first, _ in iter_dialogue_pages(rom, at, own, entry):
            # A separately registered alias still starts at page one; the page
            # count of an earlier chain must not overwrite that start's identity.
            if first == at or first not in own:
                starts[first] = (entry, page)

    return starts


def index_item_names(rom: bytes | bytearray) -> dict[int, tuple[str, list[int]]]:
    """
    Index translated item names through their original item records.

    Reads the 124 original item records at $83:8FE5, each 14 bytes long, from
    the hash-checked Japanese ROM. Only names with an active page-one source
    entry are included. Glyphs come from the supplied compiled ROM; runtime
    insertions are not expanded.

    Args:
        rom (bytes | bytearray): Compiled ROM containing the translated
            item-name strings.

    Returns:
        dict[int, tuple[str, list[int]]]: Zero-based item number (0 through 123)
            to (entry ID, literal glyph indices), omitting unregistered names.

    Raises:
        OSError: The original ROM or a translation source cannot be read.
        ValueError: The original-ROM hash, source metadata, or a text chain is invalid.
        IndexError: A referenced entry or text token lies outside the compiled ROM.
        KeyError: A decoded name contains an unsupported F8 command.
    """
    jp = original_rom.read_original_rom()
    starts = set(index_page_starts(rom).values())
    out: dict[int, tuple[str, list[int]]] = {}

    # The first two bytes of each 14-byte item record hold its bank-local
    # name pointer in the original ROM.
    for item in range(124):
        record = 0x18FE5 + 14 * item
        entry = f"{resolve_bank_pointer(int.from_bytes(jp[record : record + 2], 'little'), 0x18000):06X}"
        if (entry, 1) in starts:
            out[item] = (
                entry,
                decode_literal_glyphs(rom, resolve_text_start(rom, int(entry, 16)))[0],
            )

    return out


def get_text_slot_ranges() -> list[tuple[int, int, str]]:
    """
    Build an offset-ordered lookup of the original slots owned by active translations.

    Gameplay capture identification uses these ranges to map a pointer inside an
    original slot back to its translation entry.

    Returns:
        list[tuple[int, int, str]]: Sorted (start offset, exclusive end offset,
            entry ID) records for non-skipped entries with English text.

    Raises:
        OSError: A translation source cannot be read.
        ValueError: A collection fails source or metadata validation.
    """
    # Slot ownership excludes skipped and empty translations; sorting permits
    # address lookup even when translation collections use a different entry order.
    out: list[tuple[int, int, str]] = []
    for name in script_files.NAMES:
        for e in script_files.load(name):
            if not e.get("skip") and e["en"]:
                out.append((int(e["id"], 16), int(e["end"], 16), e["id"]))

    return sorted(out)


def index_page_terminators(rom: bytes | bytearray) -> dict[int, tuple[str, int, int]]:
    """
    Index translated page terminators and their owning entries.

    Load active records from all translation collections and resolve their
    original and retargeted starts. Original slots take priority over reference
    aliases, using the same ownership map as index_page_starts(). walk_pages()
    keeps the first owner of a shared terminator and retains valid pages before
    a malformed segment.
    Source-load and reference-resolution failures still propagate.

    Args:
        rom (bytes | bytearray): Compiled ROM whose translated page
            terminators are indexed.

    Returns:
        dict[int, tuple[str, int, int]]: Terminator offset to (entry ID, one-based page
            number, text start).

    Raises:
        OSError: A translation source cannot be read.
        ValueError: Source metadata or a reference cannot be resolved.
        IndexError: An original slot or reference operand lies outside rom.
    """
    out: dict[int, tuple[str, int, int]] = {}
    starts = _index_entry_starts(rom)

    # Different references may reach the same terminator; walk_pages preserves
    # the first owner and records only pages it can decode safely.
    for first, entry in starts.items():
        walk_pages(rom, entry, first, out, starts)

    return out


def walk_pages(
    rom: bytes | bytearray,
    entry: str,
    at: int,
    out: dict[int, tuple[str, int, int]],
    owners: dict[int, str] | None = None,
) -> None:
    """
    Record the entry and page number owning each terminator in a text chain.

    Preserve existing owners in out. ValueError, IndexError, and KeyError from
    page traversal stop indexing without propagating; pages added before the
    malformed segment remain in the index. Type errors are not suppressed.

    Args:
        rom (bytes | bytearray): ROM containing the page chain to add to
            out.
        entry (str): Entry ID assigned to discovered pages.
        at (int): First page file offset.
        out (dict[int, tuple[str, int, int]]): Mutable terminator-offset to
            (entry ID, page number, start offset) index.
        owners (dict[int, str] | None, optional): Text-start ownership map;
            prevents a page continuation from claiming another entry. None uses
            no registered boundaries, so all decodable continuations are included.

    Raises:
        TypeError: rom is not bytes or bytearray, or at is not an integer
            or is a boolean.
    """
    try:
        for page, start, end in iter_dialogue_pages(rom, at, owners or {}, entry):
            # Several references can reach one page; preserve its first owner.
            out.setdefault(end, (entry, page, start))
    except (ValueError, IndexError, KeyError):
        # Capture indexing is best effort; Reader reports malformed chains separately.
        return
