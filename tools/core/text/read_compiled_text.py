"""
This module reads text from a compiled ROM and checks its jumps, commands, and terminator locations.

Reader follows text jumps in the supplied ROM buffer and reports malformed
commands, out-of-range reads, cycles, and incorrect terminator locations.
It checks that paginated dialogue segments end where both the event scan and
text reader expect them to end. Names are checked separately because the game
copies them with stricter rules than ordinary dialogue.

Helpers decode literal glyphs, read an entry's stored references, and iterate
its dialogue pages without loading translation collections or the original ROM.
This module does not compare English meaning with Japanese or check screenshot
pixels, and does not modify the supplied ROM.

Source-entry structural validation is provided by core.text.validate_compiled_text.
Gameplay capture indexes that assign entry IDs to addresses, page terminators,
and item names are provided by verification.evidence.index_gameplay_text.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from core.rom.convert_addresses import convert_pointer_bytes_to_file, resolve_bank_pointer
from core.text.read_text_tokens import decode_glyph_index, walk_text_tokens
from core.text.read_translation_files import TextReferences

# Original report-slot file offset: the repair uses its first byte for show-text,
# so resolve_message_start() skips that opcode when locating the repaired text payload.
COMPUTER = 0x149B2A


def resolve_message_start(offset: int) -> int:
    """
    Locate the insertion start in the patched layout for an original text slot.

    At COMPUTER ($149B2A), the patch reserves the slot's first byte for event
    opcode 18, so return the following byte. All other offsets are unchanged.
    This helper assumes the patched layout without reading a ROM; use
    resolve_text_start() when the opcode must be checked in a supplied image.

    Args:
        offset (int): Zero-based byte offset in the unheadered ROM.

    Returns:
        int: Unheadered ROM file offset.

    Raises:
        TypeError: The offset is not an integer or is a boolean.
        ValueError: The offset lies outside the four-megabyte LoROM address range.
    """
    if type(offset) is not int:
        raise TypeError("Message offset must be an integer; booleans are not offsets")
    if not 0 <= offset < 0x400000:
        raise ValueError("Message offset must lie within the four-megabyte ROM")
    return offset + int(offset == COMPUTER)


def resolve_text_start(rom: bytes | bytearray, offset: int) -> int:
    """
    Locate text after the computer slot's show-text opcode when it is present.

    Only COMPUTER ($149B2A) receives special treatment, and only when its first
    byte is event opcode 18. All other valid offsets are returned unchanged.
    Unlike resolve_message_start(), this works without assuming the patch
    has already been applied.

    Args:
        rom (bytes | bytearray): ROM inspected for the repaired computer
            show-text opcode.
        offset (int): Zero-based byte offset in the unheadered ROM.

    Returns:
        int: Unheadered ROM file offset.

    Raises:
        TypeError: The offset is not an integer or is a boolean.
        IndexError: The offset or repaired text start lies outside the ROM.
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
    Inspect compiled text chains in one caller-owned ROM buffer.

    Use trace_text_chain() for a terminator and traversal diagnostic,
    validate_text_chain() for an entry's required end, or
    validate_dialogue_segments() to compare renderer jumps with event scanning.
    Malformed game text normally becomes a returned diagnostic; programmer
    input errors such as a noninteger start are raised.

    Attributes:
        rom (Union[bytes, bytearray]): Caller-owned ROM buffer retained by
            reference; keep it unchanged during a read.
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

        Returns:
            tuple[int | None, str | None]: Terminator offset and error text:
                (offset, None) on success, or (None, message) on failure.

        Raises:
            TypeError: The ROM is not bytes or bytearray, i is not an integer
                or is a boolean, or follow_jumps is not a boolean.
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

        Allows at most one leading F8/06 jump, followed by glyphs and an FF
        terminator. Copy loops reject other controls, embedded newlines and glyph
        operands containing FF.

        Args:
            i (int): Text start as an unheadered ROM file offset.

        Returns:
            str | None: Validation diagnostic, or None when the text chain
                satisfies the reader contract.

        Raises:
            TypeError: i is not an integer or is a boolean.
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
            mode (str): One of v, p, n, f, m, or c. n selects name-copy
                restrictions; the other modes share ordinary chain traversal.
                An unsupported mode produces a diagnostic, not an exception.

        Returns:
            str | None: Validation diagnostic, or None when the text chain
                satisfies the reader contract.

        Raises:
            TypeError: i is not an integer or is a boolean, or ordinary text
                traversal receives a ROM that is not bytes or bytearray.
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

        # A retargeted standalone string owns its FF terminator. Text reached
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
            a (int): First dialogue segment file offset, at the text payload
                after its show-text opcode (except for the special computer-slot
                adjustment handled by resolve_text_start()).
            end (int): Required final original terminator file offset.

        Returns:
            str | None: Validation diagnostic, or None when the text chain
                satisfies the reader contract.

        Raises:
            TypeError: The ROM is not bytes or bytearray, or a is not an integer
                or is a boolean.
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


def decode_literal_glyphs(rom: bytes | bytearray, start: int) -> tuple[list[int], int]:
    """
    Decode literal glyphs through the first terminator, following text jumps.

    Newlines and all F8 commands contribute no literal glyphs. Runtime names
    are not expanded. The returned terminator may be FD, FE, or FF; it is not
    included in the glyph list. No computer-slot adjustment is applied here.

    Args:
        rom (bytes | bytearray): ROM containing the encoded page and any
            jump targets.
        start (int): Unheadered ROM file offset at which text reading starts.

    Returns:
        tuple[list[int], int]: Literal zero-based glyph indices and terminator file offset;
            dynamic name insertions are excluded.

    Raises:
        TypeError: rom is not bytes or bytearray, or start is not an integer
            or is a boolean.
        ValueError: The text jump chain is cyclic.
        KeyError: An F8 command has no supported operand-length definition.
        IndexError: A glyph, command or followed jump reads beyond the ROM buffer.
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


def find_entry_text_starts(rom: bytes | bytearray, e: TextReferences) -> dict[int, None]:
    """
    Find an entry's slot and retargeted pointer/immediate starts.

    The ptrs metadata contains same-bank 16-bit references. The imm metadata
    contains separate low, high and bank-byte instruction operands.
    The original slot is included first, even when compilation redirects its
    references elsewhere. Duplicate target offsets appear only once.

    Args:
        rom (bytes | bytearray): Compiled ROM containing the entry slot
            and retargeted references.
        e (TextReferences): Original slot ID and optional ptrs/imm operand offsets
            in the hexadecimal form checked by read_translation_files.load().

    Returns:
        dict[int, None]: Insertion-ordered unique text-start offsets as keys, with None
            values.

    Raises:
        ValueError: An ID or operand offset is malformed, or nonempty imm
            metadata does not contain three distinct offsets.
        IndexError: The original slot or a reference operand lies outside rom.
    """
    starts = [resolve_text_start(rom, int(e["id"], 16)), *read_entry_references(rom, e)]
    return dict.fromkeys(starts)


def read_entry_references(rom: bytes | bytearray, entry: TextReferences) -> list[int]:
    """
    Resolve an entry's retargeted references without including its original slot.

    Args:
        rom (bytes | bytearray): Compiled ROM containing the pointer operands.
        entry (TextReferences): Original slot ID, optional ptrs operand offsets
            whose targets use the original slot's bank, and optional three
            distinct imm offsets in low/high/bank order.

    Returns:
        list[int]: Referenced text starts in metadata order, retaining duplicates.

    Raises:
        ValueError: A hexadecimal ID or operand offset is malformed, the slot
            bank is outside LoROM capacity, or nonempty imm metadata does not
            contain exactly three distinct offsets.
        IndexError: A reference operand falls outside the ROM.
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
        TypeError: rom is not bytes or bytearray, or a page start is not an
            integer or is a boolean.
        ValueError: A page contains a cyclic relocation chain.
        IndexError: A page token or relocation target is outside the ROM.
        KeyError: A page contains an unsupported F8 command.
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
