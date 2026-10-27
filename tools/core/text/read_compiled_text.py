"""
This module reads text from a compiled ROM and checks its jumps, commands, and terminator locations.

The reader follows text jumps in the supplied ROM buffer and reports malformed
commands, out-of-range reads, cycles, and incorrect terminator locations.
It checks that paginated dialogue segments end where both the event scan and
text reader expect them to end. Names are checked separately because the game
copies them with stricter rules than ordinary dialogue.

verify_compiled_text() loads active translation entries, locates their strings
in a supplied 4 MiB ROM, prints structural failures, and returns zero or one.
Other helpers decode literal glyphs and build indexes relating ROM text offsets,
page terminators, and item names to translation entry IDs. Capture tools use
those indexes to identify displayed text. This module does not compare English
meaning with Japanese or check screenshot pixels, and does not modify the ROM.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from core.rom import load_original_rom as original_rom
from core.rom.convert_addresses import convert_pointer_bytes_to_file, resolve_bank_pointer
from core.text import read_translation_files as script_files
from core.text.read_text_tokens import decode_glyph_index, walk_text_tokens
from core.text.read_translation_files import TextReferences

# Original report-slot file offset: the repair uses its first byte for show-text,
# so resolve_message_start() skips that opcode when locating the repaired text payload.
COMPUTER = 0x149B2A


def resolve_message_start(offset: int) -> int:
    """
    Locate the insertion start in the patched layout for an original text slot.

    The computer-report slot at $149B2A reserves its first byte for event
    opcode 18, so its text begins at the following byte. Other offsets are
    unchanged. This function assumes that layout without reading the ROM.

    Args:
        offset (int): Zero-based byte offset in the unheadered ROM.

    Returns:
        int: Unheadered ROM file offset.

    Raises:
        TypeError: If the offset is not an integer or is a boolean.
        ValueError: If the offset lies outside the four-megabyte LoROM address range.
    """
    if type(offset) is not int:
        raise TypeError("Message offset must be an integer; booleans are not offsets")
    if not 0 <= offset < 0x400000:
        raise ValueError("Message offset must lie within the four-megabyte ROM")
    return offset + int(offset == COMPUTER)


def resolve_text_start(rom: bytes | bytearray, offset: int) -> int:
    """
    Locate text after the computer slot's show-text opcode when it is present.

    At $149B2A only, a leading event opcode 18 moves the text start forward
    one byte. Other offsets are returned unchanged. The supplied ROM is read
    to establish whether that opcode is present.

    Args:
        rom (bytes | bytearray): ROM inspected for the repaired computer
            show-text opcode.
        offset (int): Zero-based byte offset in the unheadered ROM.

    Returns:
        int: Unheadered ROM file offset.

    Raises:
        TypeError: If the offset is not an integer or is a boolean.
        IndexError: If the offset or repaired text start lies outside the ROM.
    """
    # Validate before adding the repair adjustment: arithmetic would otherwise
    # turn a boolean into an ordinary integer and bypass the token reader's check.
    if type(offset) is not int:
        raise TypeError("Text offset must be an integer; booleans are not offsets")
    if not 0 <= offset < len(rom):
        raise IndexError(f"outside ROM at {offset:06X}")
    start = offset + int(offset == COMPUTER and rom[offset] == 0x18)
    if start >= len(rom):
        raise IndexError(f"outside ROM at {start:06X}")
    return start


class Reader:
    """
    Reader checks compiled text jumps, name records, and dialogue segment boundaries.

    The reader retains the supplied ROM buffer without changing it. Traversal failures normally
    become returned diagnostics; invalid API arguments can raise exceptions instead.

    Attributes:
        rom (Union[bytes, bytearray]): Caller-owned ROM buffer retained by reference; keep it
            unchanged during a read.
    """

    def __init__(self, rom: bytes | bytearray) -> None:
        """
        Retain the supplied buffer by reference.

        Args:
            rom (bytes | bytearray): Text buffer retained by reference; it
                must remain unchanged while reading.
        """
        self.rom = rom

    def trace_text_chain(self, i: int, follow_jumps: bool = True) -> tuple[int | None, str | None]:
        """
        Follow a compiled text chain to locate its terminator or explain a malformed chain.

        The text reader follows relocation commands, while the event interpreter's
        initial scan stops at its own command boundaries. The follow_jumps option
        models these two paths so dialogue verification can check their agreement.

        Args:
            i (int): Text start as an unheadered ROM file offset.
            follow_jumps (bool, optional): Follow text jumps when True; False
                models the event interpreter scan.
                Defaults to True.

        Returns:
            tuple[int | None, str | None]: Terminator offset and error text:
                (offset, None) on success, or (None, message) on failure.

        Raises:
            TypeError: If the ROM is not bytes or bytearray, i is not an integer or is a
                boolean, or follow_jumps is not a boolean.
        """
        # Apply the computer-slot adjustment before walking; the generic token reader
        # does not know that this repaired slot begins with a show-text event opcode.
        try:
            for offset, _ in walk_text_tokens(
                self.rom, resolve_text_start(self.rom, i), follow_jumps, max_tokens=4000
            ):
                if self.rom[offset] >= 0xFD:
                    return offset, None
        # Keep the reader diagnostic API while using the shared parser. Unknown
        # commands carry both the subopcode and its failing file offset.
        except KeyError as error:
            return None, "bad F8 %02X at %06X" % error.args
        except (IndexError, ValueError) as error:
            return None, str(error)

        raise AssertionError("Text walk ended without a terminator")

    def validate_name_chain(self, i: int) -> str | None:
        """
        Validate a name against the game's raw-copy restrictions.

        A valid name has at most one leading F8/06 jump, followed by glyphs and an FF
        terminator. The copy loops reject other controls, embedded newlines and glyph operands
        containing FF.

        Args:
            i (int): Text start as an unheadered ROM file offset.

        Returns:
            str | None: Validation diagnostic, or None when the name contains only permitted
                glyphs, an optional leading F8/06 jump and a final FF terminator.

        Raises:
            TypeError: If i is not an integer or is a boolean.
        """
        if type(i) is not int:
            raise TypeError("Name offset must be an integer; booleans are not offsets")
        if not 0 <= i < len(self.rom):
            return "outside ROM at %06X" % i

        # Native name-copy loops follow at most one initial relocation jump.
        # After it, only glyph bytes ending in FF are valid.
        if self.rom[i] == 0xF8:
            if i + 1 >= len(self.rom):
                return "truncated F8 at %06X" % i

            if self.rom[i + 1] == 0x06:
                if i + 5 > len(self.rom):
                    return "truncated name jump at %06X" % i

                i = convert_pointer_bytes_to_file(self.rom[i + 2], self.rom[i + 3], self.rom[i + 4])

        # This 64-byte scan bound detects missing terminators; it is not the
        # allowed name length. Source-name byte limits are validated separately.
        start = i
        while i - start < 64:
            if not 0 <= i < len(self.rom):
                return "outside ROM at %06X" % i

            c = self.rom[i]
            if c == 0xFF:
                return None

            # Dialogue terminators and text controls cannot be copied as name
            # glyphs; names must terminate specifically with FF.
            if c in (0xFD, 0xFE):
                return "expected FF, found %02X at %06X" % (c, i)

            if c == 0xF8 and i + 1 < len(self.rom) and self.rom[i + 1] == 0x06:
                return "name has F8 06 at %06X: a second jump (copy loops follow only one)" % i

            if c in (0xF8, 0x00):
                return "name has %02X at %06X (copy loops take only glyphs up to FF)" % (c, i)

            if 0xF9 <= c <= 0xFC:
                if i + 1 >= len(self.rom):
                    return "truncated name glyph at %06X" % i

                # The raw copy loop stops on FF even inside a two-byte glyph, so that
                # operand would truncate the copied name before rendering.
                if self.rom[i + 1] == 0xFF:
                    return "name glyph contains FF at %06X (copy loops stop before its operand)" % i

                i += 2
            else:
                i += 1

        return "name has no FF within 64 bytes"

    def validate_text_chain(self, i: int, end: int, mode: str) -> str | None:
        """
        Check that a compiled entry reaches its required terminator safely.

        Names are checked against native raw-copy restrictions. Other modes use the
        text reader's jump chain and require either the original slot terminator or
        a standalone FF terminator, as selected by end.

        Args:
            i (int): Text start as an unheadered ROM file offset.
            end (int): Expected original terminator file offset; a negative value
                requires a standalone FF terminator. Ignored in name mode n,
                whose native copy loop always requires FF.
            mode (str): Translation layout mode; n selects name-copy restrictions.

        Returns:
            str | None: Validation diagnostic, or None when the text passes the name-copy checks
                or reaches the terminator required by end.

        Raises:
            TypeError: If i is not an integer or is a boolean, or ordinary text traversal
                receives a ROM that is not bytes or bytearray.
        """
        if mode not in ("v", "p", "n", "f", "m", "c"):
            return f"unsupported text mode: {mode!r}"
        if mode == "n":
            return self.validate_name_chain(i)

        t, err = self.trace_text_chain(i)
        if err:
            return err

        if t is None:
            raise AssertionError("Text chain has no terminator")

        # A separately allocated string ends with its own FF byte. Text reached
        # through the original slot must return to that slot's existing terminator.
        if end < 0:
            return (
                None
                if self.rom[t] == 0xFF
                else "expected FF, found %02X at %06X" % (self.rom[t], t)
            )

        return None if t == end else "terminator at %06X, expected %06X" % (t, end)

    def validate_dialogue_segments(self, a: int, end: int) -> str | None:
        """
        Check agreement between dialogue scanning and text reading.

        Inner segments end in FD/18. The final segment must reach the original
        terminator, and interpreter scans must agree with text-reader jump chains
        at every segment.

        Args:
            a (int): First dialogue segment file offset.
            end (int): Required final original terminator file offset.

        Returns:
            str | None: Validation diagnostic, or None when the dialogue ends at the original
                terminator and every page boundary agrees with the event scan.

        Raises:
            TypeError: If the ROM is not bytes or bytearray, or a is not an integer or is a
                boolean.
        """
        # The event interpreter and text renderer traverse relocation differently.
        # Each dialogue segment must bring both back to the same native terminator.
        pos = a
        while True:
            t, err = self.trace_text_chain(pos, follow_jumps=False)
            if err:
                return "interpreter at %06X: %s" % (pos, err)

            r, err = self.trace_text_chain(pos)
            if err:
                return "text at %06X: %s" % (pos, err)

            if t is None or r is None:
                raise AssertionError("Text chain has no terminator")

            if t != r:
                return "segment at %06X: interpreter stops at %06X, text at %06X" % (pos, t, r)

            if t == end:
                return None

            # Only FD followed by event opcode 18 continues a dialogue page chain.
            # Any other boundary must already be the entry's final terminator.
            if self.rom[t] != 0xFD or t + 1 >= len(self.rom) or self.rom[t + 1] != 0x18 or t > end:
                return "segment ends at %06X without FD 18" % t

            pos = t + 2


def verify_compiled_text(rom: bytes | bytearray) -> int:
    """
    Validate all active translations in a compiled 4 MiB ROM.

    Args:
        rom (bytes | bytearray): Compiled English ROM to validate; must
            be exactly 4 MiB.

    Returns:
        int: Zero when every active entry validates; one on a ROM-size or
            text-chain failure.

    Raises:
        OSError: If a translation source cannot be read.
        ValueError: If source metadata or a stored reference cannot be resolved.
        IndexError: If a stored reference operand lies outside the ROM.
    """
    reader = Reader(rom)

    # Compiled text may live in expansion banks, so structural validation requires
    # the complete 4 MiB English image rather than the original 3 MiB cartridge.
    if len(rom) != 0x400000:
        print(f"Invalid English ROM size: {len(rom)} bytes; expected {0x400000}")
        print("bad 1")
        return 1

    bad = 0

    for f in script_files.NAMES:
        for e in script_files.load(f):
            if not e["en"] or e.get("skip"):
                continue

            mode = e.get("mode", "v")
            if e.get("ptrs") or e.get("imm"):
                # Repointed pointers/immediates lead to strings with their own FF terminator.
                refs = read_entry_references(rom, e)

                for t in refs:
                    err = reader.validate_text_chain(t, -1, mode)
                    if err:
                        bad += 1
                        print(f, e["id"], "via pointer", err, repr(e["en"][:40]))

                continue

            # Entries without retargeted references must return to their original slot
            # terminator. Dialogue also checks agreement with the event interpreter scan.
            a, b = int(e["id"], 16), int(e["end"], 16)
            err = (
                reader.validate_dialogue_segments(a, b)
                if mode == "v"
                else reader.validate_text_chain(a, b, mode)
            )
            if err:
                bad += 1
                print(f, e["id"], mode, err, repr(e["en"][:40]))

    print("bad", bad)
    return 1 if bad else 0


def decode_literal_glyphs(rom: bytes | bytearray, start: int) -> tuple[list[int], int]:
    """
    Decode literal glyphs through the first terminator, following text jumps.

    Newlines and all F8 commands contribute no literal glyphs. Runtime names are not expanded.
    The returned terminator may be FD, FE, or FF; it is not included in the glyph list. No
    computer-slot adjustment is applied here.

    Args:
        rom (bytes | bytearray): ROM containing the encoded page and any
            jump targets.
        start (int): Unheadered ROM file offset at which text reading starts.

    Returns:
        tuple[list[int], int]: Literal zero-based glyph indices and terminator file offset;
            dynamic name insertions are excluded.

    Raises:
        TypeError: If rom is not bytes or bytearray, or start is not an integer or is a boolean.
        ValueError: If the text jump chain is cyclic.
        KeyError: If an F8 command has no supported operand-length definition.
        IndexError: If a glyph, command or followed jump reads beyond the ROM buffer.
    """
    out: list[int] = []
    try:
        for offset, _ in walk_text_tokens(rom, start):
            code = rom[offset]
            if code >= 0xFD:
                return out, offset

            # Dynamic insertions and newline codes draw no literal font glyph here.
            if code not in (0, 0xF8):
                out.append(decode_glyph_index(rom, offset))
    except ValueError as error:
        raise ValueError(f"cyclic text at {start:06X}") from error

    raise AssertionError("Text walk ended without a terminator")


def index_page_starts(rom: bytes | bytearray) -> dict[int, tuple[str, int]]:
    """
    Map compiled text-start offsets to translation entry IDs and dialogue page numbers.

    The index includes original slots and retargeted pointer/immediate starts, following
    relocated text and page continuations. Gameplay trace readers use it to identify which
    translated page the native text engine has started.

    Args:
        rom (bytes | bytearray): Compiled ROM whose dialogue chains are indexed.

    Returns:
        dict[int, tuple[str, int]]: Text-start offset to (entry ID, one-based page number).

    Raises:
        OSError: If a translation source cannot be read.
        ValueError: If source metadata or a stored reference is invalid.
        IndexError: If an original slot or reference operand lies outside the ROM.
        KeyError: If a decoded page contains an unsupported F8 command.
    """
    entries = [
        e
        for name in script_files.NAMES
        for e in script_files.load(name)
        if not e.get("skip") and e["en"]
    ]
    # Keep the original entry ID when a reference points into an
    # independently registered entry. Register every boundary before walking.
    own = {resolve_text_start(rom, int(e["id"], 16)): e["id"] for e in entries}
    for e in entries:
        for first in read_entry_references(rom, e):
            own.setdefault(first, e["id"])
    starts = {at: (i, 1) for at, i in own.items()}
    for at, entry in own.items():
        # Preserve each registered start's entry ID when another page chain reaches it.
        for page, first, _ in iter_dialogue_pages(rom, at, own, entry):
            # A separately registered alias still starts at page one; the page
            # count from an earlier chain must not change its entry ID or page number.
            if first == at or first not in own:
                starts[first] = (entry, page)

    return starts


def index_item_names(rom: bytes | bytearray) -> dict[int, tuple[str, list[int]]]:
    """
    Index translated item names through their original item records.

    The function reads the 124 original item records at $83:8FE5, each 14 bytes long, from the
    hash-checked Japanese ROM. Only names with an active page-one source entry are included.
    Glyphs come from the supplied compiled ROM; runtime insertions are not expanded.

    Args:
        rom (bytes | bytearray): Compiled ROM containing the translated item-name strings.

    Returns:
        dict[int, tuple[str, list[int]]]: Zero-based item number (0 through 123) to (entry ID,
            literal glyph indices), omitting unregistered names.

    Raises:
        OSError: If the original ROM or a translation source cannot be read.
        ValueError: If the original-ROM hash, source metadata, or a text chain is invalid.
        IndexError: If a referenced entry or text token lies outside the compiled ROM.
        KeyError: If a decoded name contains an unsupported F8 command.
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
    List original text-slot ranges and their active translation entry IDs in offset order.

    Gameplay capture identification uses these ranges to map a pointer inside an original slot
    back to its translation entry.

    Returns:
        list[tuple[int, int, str]]: Sorted (start offset, exclusive end offset,
            entry ID) records for non-skipped entries with English text.

    Raises:
        OSError: If a translation source cannot be read.
        ValueError: If a collection fails source or metadata validation.
    """
    # Exclude skipped and empty translations; sorting permits
    # address lookup even when translation collections use a different entry order.
    out: list[tuple[int, int, str]] = []
    for name in script_files.NAMES:
        for e in script_files.load(name):
            if not e.get("skip") and e["en"]:
                out.append((int(e["id"], 16), int(e["end"], 16), e["id"]))

    return sorted(out)


def find_entry_text_starts(rom: bytes | bytearray, e: TextReferences) -> dict[int, None]:
    """
    Find an entry's slot and retargeted pointer/immediate starts.

    The ptrs metadata contains same-bank 16-bit references. The imm metadata contains separate
    low, high and bank-byte instruction operands. The original slot is included first, even when
    compilation redirects its references elsewhere. Duplicate target offsets appear only once.

    Args:
        rom (bytes | bytearray): Compiled ROM containing the entry slot
            and retargeted references.
        e (TextReferences): Original slot ID and optional ptrs/imm operand offsets
            in the hexadecimal form checked by read_translation_files.load().

    Returns:
        dict[int, None]: Insertion-ordered unique text-start offsets as keys, with None
            values.

    Raises:
        ValueError: If an ID or operand offset is malformed, or nonempty imm metadata does not
            contain three distinct offsets.
        IndexError: If the original slot or a reference operand lies outside rom.
    """
    starts = [resolve_text_start(rom, int(e["id"], 16)), *read_entry_references(rom, e)]
    return dict.fromkeys(starts)


def read_entry_references(rom: bytes | bytearray, entry: TextReferences) -> list[int]:
    """
    Resolve an entry's retargeted references without including its original slot.

    Args:
        rom (bytes | bytearray): Compiled ROM containing the pointer operands.
        entry (TextReferences): Original slot ID, optional ptrs operand offsets whose targets
            use the original slot's bank, and optional three distinct imm offsets in
            low/high/bank order.

    Returns:
        list[int]: Referenced text starts in metadata order, retaining duplicates.

    Raises:
        ValueError: If a hexadecimal ID or operand offset is malformed, a bank-local reference
            uses a slot bank outside LoROM capacity, or nonempty imm metadata does not contain
            exactly three distinct offsets.
        IndexError: If a reference operand falls outside the ROM.
    """
    slot = int(entry["id"], 16)
    starts: list[int] = []

    # ptrs lists the ROM locations of 16-bit operands. Their missing bank comes
    # from the original text slot, not from the operand location.
    for pointer in entry.get("ptrs") or []:
        offset = int(pointer, 16)
        if not 0 <= offset < len(rom) - 1:
            raise IndexError("Two-byte text pointer lies outside the ROM")
        address = rom[offset] | rom[offset + 1] << 8
        starts.append(resolve_bank_pointer(address, slot))

    # imm lists three separate instruction operands in low/high/bank order.
    # They need not be adjacent, so read each recorded offset independently.
    immediates = entry.get("imm")
    if immediates:
        if len(immediates) != 3:
            raise ValueError("imm metadata requires low, high and bank offsets")
        offsets = [int(offset, 16) for offset in immediates]
        if len(set(offsets)) != 3:
            raise ValueError("imm metadata requires three distinct operand offsets")
        if any(not 0 <= offset < len(rom) for offset in offsets):
            raise IndexError("Text address operand lies outside the ROM")
        lo, hi, bank = (rom[offset] for offset in offsets)
        starts.append(convert_pointer_bytes_to_file(lo, hi, bank))
    return starts


def iter_dialogue_pages(
    rom: bytes | bytearray, start: int, boundaries: Mapping[int, str], owner: str | None = None
) -> Iterator[tuple[int, int, int]]:
    """
    Follow FD/18 page continuations until a boundary, terminator or repeated start.

    Args:
        rom (bytes | bytearray): ROM containing the complete dialogue chain.
        start (int): First page's file offset.
        boundaries (Mapping[int, str]): Registered starts mapped to entry IDs.
            The initial page is always read.
        owner (str | None, optional): Permit continuations into starts belonging
            to this entry; None stops at every registered start.

    Yields:
        tuple[int, int, int]: One-based page number, start offset and terminator offset.

    Raises:
        TypeError: If rom is not bytes or bytearray, or a page start is not an integer or is a
            boolean.
        ValueError: If a page contains a cyclic relocation chain.
        IndexError: If a page token or relocation target is outside the ROM.
        KeyError: If a page contains an unsupported F8 command.
    """
    page = 1
    seen: set[int] = set()
    while start not in seen:
        seen.add(start)
        _, end = decode_literal_glyphs(rom, start)
        yield page, start, end

        # A final FD may end the buffer; it is not a continuation without opcode 18.
        if rom[end : end + 2] != b"\xfd\x18":
            return
        # A continuation can run straight into another registered entry. Stop before
        # claiming its pages, unless its owner is explicitly permitted by the caller.
        if end + 2 in boundaries and (owner is None or boundaries[end + 2] != owner):
            return
        start, page = end + 2, page + 1


def index_page_terminators(rom: bytes | bytearray) -> dict[int, tuple[str, int, int]]:
    """
    Map translated page terminator offsets to translation entry IDs.

    Args:
        rom (bytes | bytearray): Compiled ROM whose translated page terminators are indexed.

    Returns:
        dict[int, tuple[str, int, int]]: Terminator offset to (entry ID, one-based page number,
            text start).

    Raises:
        OSError: If a translation source cannot be read.
        ValueError: If source metadata or a stored reference is invalid.
        IndexError: If an original slot or reference operand lies outside the ROM.
    """
    out: dict[int, tuple[str, int, int]] = {}
    starts: dict[int, str] = {}

    # Register original and retargeted starts before traversal so that a chain
    # cannot assign its entry ID to a start belonging to an entry loaded later.
    for name in script_files.NAMES:
        for e in script_files.load(name):
            if e.get("skip") or not e["en"]:
                continue

            for first in find_entry_text_starts(rom, e):
                starts[first] = e["id"]

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
    Map each terminator in a text chain to its entry ID and page number.

    Existing entry IDs in out are preserved. Traversal stops on ValueError,
    IndexError, or KeyError, retaining pages indexed before the malformed segment.
    TypeError is not suppressed.

    Args:
        rom (bytes | bytearray): ROM containing the page chain to add to
            out.
        entry (str): Entry ID assigned to discovered pages.
        at (int): First page file offset.
        out (dict[int, tuple[str, int, int]]): Mutable terminator-offset to
            (entry ID, page number, start offset) index.
        owners (dict[int, str] | None, optional): Text-start offsets mapped to entry IDs.
            A continuation stops before a start mapped to another entry ID. None
            disables these boundaries. Supplied dictionaries are used directly,
            without copying or modification.

    Raises:
        TypeError: If rom is not bytes or bytearray, or at is not an integer or is a boolean.
    """
    try:
        for page, start, end in iter_dialogue_pages(
            rom, at, {} if owners is None else owners, entry
        ):
            # Several references can reach one page; preserve its first owner.
            out.setdefault(end, (entry, page, start))
    except (ValueError, IndexError, KeyError):
        # Capture indexing is best effort; Reader reports malformed chains separately.
        return
