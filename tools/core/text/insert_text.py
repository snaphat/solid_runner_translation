"""
This module lays out English translations and writes their encoded bytes and updated references into a supplied ROM buffer.

insert_translations() takes a mutable 4 MiB ROM and records containing each
translation's original start/end offsets, English string, layout mode, and
optional width or pointer locations. It encodes and lays out the text, allocates
space for strings that need relocation, writes them, and updates text jumps,
16-bit pointers, or address operands so the game can read the new strings.
It modifies the supplied buffer and returns the dialogue allocation cursor;
it does not read translation files or save the finished ROM. ROM-space
reservations are managed by core.rom.allocate_space.

The original writable slot is [start, end); end identifies the existing
terminator and is excluded. Ordinary relocated dialogue jumps back to that
terminator. Names and strings reached through retargeted references can have
their own FF terminator. Fixed records, menus, centered text, and short slots
use separate encoding or placement rules.

Width calculations use core.text.measure_text. Layout helpers wrap dialogue
into pages of at most three lines and encode menu-column padding. Newline characters
force line breaks and form feeds force page breaks. Dialogue page boundaries
use FD followed by event opcode 18, with a segment left in the original slot
for each page.
"""

from __future__ import annotations

import re
from typing import Iterable, TypeAlias

from core.rom import allocate_space
from core.rom.convert_addresses import convert_file_offset_to_snes
from core.text import measure_text
from core.text.encode_text import encode_text
from core.text.read_text_tokens import split_dialogue_pages

MAX_W = 236  # usable pixels per line (30 tiles = 240, keep a margin)
LINES = 3  # Dialogue rows per page before an automatic page break.
PAGE = b"\xfd\x18"  # FD waits/ends the page; event opcode 18 starts the next text.


# A complete compiler record includes explicit pointer and address-operand lists.
CompleteInsertionRecord: TypeAlias = tuple[int, int, str, str, int | None, list[int], list[int]]

# Insertion records may omit trailing options: mode, width, pointers and
# immediate-address operands. Bounds are unheadered ROM file offsets.
InsertionRecord: TypeAlias = (
    tuple[int, int, str]
    | tuple[int, int, str, str]
    | tuple[int, int, str, str, int | None]
    | tuple[int, int, str, str, int | None, list[int] | None]
    | tuple[int, int, str, str, int | None, list[int] | None, list[int] | None]
)


def encode_text_jump(off: int) -> bytes:
    """
    Return the F8 06 long-text-jump bytes targeting the supplied file offset.

    Args:
        off (int): Zero-based byte offset in the unheadered ROM.

    Returns:
        bytes: Five-byte F8/06 long-text jump.
    """
    # F8 06 is a text jump followed by the low, high and bank address bytes.
    a = convert_file_offset_to_snes(off)
    return bytes([0xF8, 0x06, a & 0xFF, (a >> 8) & 0xFF, a >> 16])


def wrap_text(
    rom: bytes | bytearray, text: str, max_w: int = MAX_W, paginate: bool = True
) -> bytes:
    """
    Wrap English text to its pixel limit and optionally split it into dialogue pages.

    Uses 00 for newlines and, when pagination is enabled, FD/18 breaks after
    three lines.

    Args:
        rom (bytes | bytearray): Font tables used to fit words within
            max_w.
        text (str): English translation text, including supported
            brace-delimited control markup.
        max_w (int, optional): Maximum line width in pixels.
        paginate (bool, optional): Insert FD/18 page breaks after three lines
            when True.

    Returns:
        bytes: Encoded text with layout controls; no final FF terminator is
            added.

    Raises:
        ValueError: The line width is not positive.
        AssertionError: A wrapped line still exceeds its pixel-width allowance.
    """
    if max_w <= 0:
        raise ValueError("Text width must be positive")

    # Spaces and line breaks inside raw hexadecimal markup separate bytes, not
    # words or paragraphs. Encode complete tokens before splitting layout text
    # so wrapping cannot cut a valid command in half.
    text = re.sub(r"\{[^{}]*\}", lambda match: "{" + encode_text(match.group()).hex() + "}", text)
    pages: list[list[str]] = []
    insert_sizes: dict[bytes, int] = {}

    # An explicit form feed starts a new page before word wrapping. Explicit
    # newlines remain paragraph boundaries inside that page.
    for page_src in text.split("\f"):
        lines: list[str] = []
        for para in page_src.split("\n"):
            # Leading spaces position text in the original window. Keep them on
            # this explicit line; wrapped continuations start at the normal margin.
            body = para.lstrip(" ")
            words = body.split(" ")
            cur = para[: len(para) - len(body)] + words[0] if body else ""
            for word in words[1:]:
                cand = word if not cur else cur + " " + word
                if (
                    cur
                    and measure_text.measure_text_width(rom, encode_vwf_line(cand), insert_sizes)
                    > max_w
                ):
                    lines.append(cur)
                    cur = word
                else:
                    cur = cand

            lines.append(cur)

        if not paginate or len(lines) <= LINES:
            pages.append(lines)
            continue

        # Distribute wrapped lines evenly across pages (4 → 2+2, 5 → 3+2,
        # 7 → 3+2+2) rather than leaving a single-line final page.
        n = -(-len(lines) // LINES)
        k = 0
        for i in range(n):
            size = len(lines) // n + (1 if i < len(lines) % n else 0)
            pages.append(lines[k : k + size])
            k += size

    for p in pages:
        for line in p:
            w = measure_text.measure_text_width(rom, encode_vwf_line(line), insert_sizes)
            if not (w <= max_w):
                raise AssertionError(f"word too wide ({w}px): {line!r}")

    return PAGE.join(b"\x00".join(encode_vwf_line(l) for l in p) for p in pages)


def encode_vwf_line(line: str) -> bytes:
    """
    Encode one wrapped line with VWF setup before a leading runtime insertion.

    A leading name or number insertion would otherwise select the fixed-width
    font. A blank one-pixel FCF3 glyph switches the renderer to variable-width
    mode before the inserted text is drawn.

    Args:
        line (str): One decoded text line without a newline terminator.

    Returns:
        bytes: Encoded text with layout controls; no final FF terminator is
            added.
    """
    b = encode_text(line)
    if b[:1] == b"\xf8" and b[1:2] in (b"\x0c", b"\x12", b"\x18", b"\x04"):
        b = b"\xfc\xf3" + b

    return b


def encode_menu_layout(
    rom: bytes | bytearray, text: str, cells_total: int | None = None
) -> bytes | bytearray:
    """
    Encode menu labels at their assigned fixed-width cell positions.

    For example, `@0:Sound@4:Text Speed@12:Controller` positions labels at 12
    pixels per cell. VWF spaces pad each label to its assigned column.

    Args:
        rom (bytes | bytearray): Font tables used to fit labels between
            menu columns.
        text (str): English translation text, including supported
            brace-delimited control markup.
        cells_total (int | None, optional): Fixed-width cells reserved per
            menu row. None uses a 99-cell final-label bound; earlier labels
            remain bounded by the following column marker.

    Returns:
        bytes | bytearray: Encoded menu rows with explicit cell positioning,
            joined by 00 newline bytes. No final FF is added.

    Raises:
        ValueError: The row width is not positive or menu column markup is malformed.
        AssertionError: A label overlaps its predecessor or exceeds its
            assigned pixel width.
    """
    if cells_total is not None and cells_total <= 0:
        raise ValueError("Menu row width must be positive")
    if "\n" in text:  # several menu rows: lay out each, joined by 00 newlines
        return b"\x00".join(encode_menu_layout(rom, line, cells_total) for line in text.split("\n"))

    # Require markers to cover the entire row. Text outside a marker would
    # otherwise have no native column and could shift the selectable labels.
    parts = re.findall(r"@(\d+):([^@]*)", text)
    if not parts or "".join("@" + cell + ":" + item for cell, item in parts) != text:
        raise ValueError("Menu rows require @column:label markers with no text outside them")

    columns = [int(cell) for cell, _ in parts]
    if columns != sorted(set(columns)):
        raise ValueError("Menu columns must be unique and increasing")

    if any(not item.strip() for _, item in parts):
        raise ValueError("Menu labels must not be blank")

    out = b""
    x = 0
    for k, (cell, item) in enumerate(parts):
        target = int(cell) * 12
        if not (x <= target):
            raise AssertionError(f"menu item overlaps previous: {text!r}")

        # Pad to the marker's pixel position using four-pixel spaces, then
        # one-pixel FCF3 spacers for the remainder.
        n, r = divmod(target - x, measure_text.SPACEW)
        out += bytes([1]) * n + bytes([0xFC, 0xF3]) * r  # FCF3 = 1px space (see asm/vwf.asm)
        x = target
        b = encode_text(item)
        x += measure_text.measure_text_width(rom, b)
        out += b

        # Each label ends before the next marker, or before the supplied row
        # width. The comparison permits two pixels beyond that boundary; an
        # unspecified final width uses a 99-cell fallback.
        limit = int(parts[k + 1][0]) * 12 if k + 1 < len(parts) else (cells_total or 99) * 12
        if not (x <= limit + 2):
            raise AssertionError(
                f"menu item too wide ({x - target}px > {limit - target}): {item!r}"
            )

    return out


def insert_translations(
    rom: bytearray,
    messages: Iterable[InsertionRecord],
    bank_free: dict[int, list[int]] | None = None,
) -> int:
    """
    Insert translated text and retarget its native references.

    Layout modes:

        v: Wrap dialogue into pages of at most three rows and preserve event-script
            continuations.

        p: Wrap prose to the supplied pixel width without adding page breaks.

        n: Wrap using the supplied width and store an FF-terminated name. Source
            validation separately requires a name to remain on one line.

        f: Encode fixed text directly, preserving authored breaks and using
            fixed-font storage when relocation is necessary.

        m: Place labels at explicit menu-column markers measured in 12-pixel cells.

        c: Encode a banner directly and add spaces to center it within 240 pixels.

    Args:
        rom (bytearray): Mutable 4 MiB candidate ROM receiving translated text,
            jumps and retargeted pointers.
        messages (Iterable[InsertionRecord]): Tuples beginning with original start
            offset, terminator offset and English text. Optional trailing fields
            are mode, width, pointer locations and immediate-address locations,
            in that order. Offsets are unheadered ROM file positions. Mode defaults
            to v. Wrapped modes use pixels: an omitted width selects 236 for v
            and 216 for p/n, while explicit None selects 236. Mode m uses cells
            with a 99-cell final-label bound when width is omitted or None.
            Modes f/c do not use this width field. Pointer lists identify
            16-bit bank-local references; immediate lists identify low/high/bank
            instruction operands. The original terminator is outside the writable
            slot [start, end). Immediate operands must occupy three distinct bytes.
        bank_free (dict[int, list[int]] | None, optional): Shared bank-tail
            [next-free, end] cursors; None starts an independent allocation
            pass.

    Returns:
        int: Final VWF allocator file offset; other allocator cursors are not
            included.

    Raises:
        ValueError: The candidate is not a 4 MiB ROM, or a slot, layout mode,
            width or reference operand is invalid.
        AssertionError: Allocation, original-slot capacity or native pagination
            constraints prevent safe insertion. Paginated dialogue requires room
            for native segments in its original slot and cannot use a standalone
            payload reached through retargeted references.
    """
    # Use separate allocators for proportional text, fixed text and short
    # jump targets because the reader chooses its font and address rules by bank.
    # Allocation regions refer to the expanded image. Python slice assignments
    # beyond a short buffer would append bytes at the wrong address instead.
    if len(rom) != 0x400000:
        raise ValueError("Translation insertion requires a 4 MiB candidate ROM")
    messages = list(messages)
    # Reject malformed storage references before any message can alter the ROM.
    for msg in messages:
        start, end, _ = msg[:3]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end < len(rom)
            or end >= 0x400000
        ):
            raise ValueError("Text slot must lie within the ROM and exclude its terminator")
        if len(msg) > 3 and msg[3] not in ("v", "p", "n", "f", "m", "c"):
            raise ValueError(f"Unsupported text mode: {msg[3]!r}")
        if len(msg) > 4 and msg[4] is not None and (type(msg[4]) is not int or msg[4] <= 0):
            raise ValueError("Text width must be positive")
        if (
            len(msg) > 5
            and msg[5]
            and any(type(at) is not int or not 0 <= at < len(rom) - 1 for at in msg[5])
        ):
            raise ValueError("Text pointer extends beyond the ROM")
        if len(msg) > 6 and msg[6]:
            if len(msg[6]) != 3 or any(
                type(at) is not int or not 0 <= at < len(rom) for at in msg[6]
            ):
                raise ValueError("Immediate references must identify three ROM operand bytes")
            if len(set(msg[6])) != 3:
                raise ValueError("Immediate references must identify three distinct operand bytes")
    bank_free = {} if bank_free is None else bank_free
    vwf = allocate_space.Space(allocate_space.FREE_REGIONS)
    fixed = allocate_space.Space(allocate_space.FIXED_REGIONS)
    short = allocate_space.Space(allocate_space.SHORT_REGIONS)
    for msg in messages:
        start, end, text = msg[:3]
        mode = msg[3] if len(msg) > 3 else "v"
        width = msg[4] if len(msg) > 4 else (MAX_W if mode == "v" else 216)
        laid: bytes | bytearray

        if mode == "f":
            laid = encode_text(text)
        elif mode == "c":  # centered banner (location names): center within 240px
            b = encode_text(text)
            pad = max(0, (240 - measure_text.measure_text_width(rom, b)) // 2)
            n, r = divmod(pad, measure_text.SPACEW)
            laid = bytes([1]) * n + bytes([0xFC, 0xF3]) * r + b
        elif mode == "m":
            laid = encode_menu_layout(rom, text, msg[4] if len(msg) > 4 else None)
        else:
            laid = wrap_text(
                rom, text, width if width is not None else MAX_W, paginate=(mode == "v")
            )  # 'p' and 'n': no pages

        # Choose how native references reach the payload. The original slot
        # excludes its terminator; ptrs holds 16-bit pointer locations and imm
        # holds the low/high/bank instruction operands of a complete address.
        slot = end - start
        ptrs = msg[5] if len(msg) > 5 else None
        imm = msg[6] if len(msg) > 6 else None
        pages = split_dialogue_pages(laid) if mode == "v" else []
        if len(pages) > 1 and (slot < 5 or imm or (ptrs and len(laid) > slot)):
            # The event interpreter resumes in the original script after each
            # page. A standalone payload or four-byte jump cannot supply the
            # required native wait/show-text segments.
            raise AssertionError(
                f"{start:06X}: paginated dialogue requires native segments in the original slot"
            )
        if imm:
            # Full-address instruction operands can reach the payload directly.
            # Any bank-local pointers need a same-bank stub to reach that same
            # payload; neither reference type supersedes the other.
            body = laid + bytes([0xFF])
            at = vwf.allocate(len(body))
            stub_at: int | None = None
            stub = b""
            if ptrs:
                stub = encode_text_jump(at) + b"\xff"
                stub_at = allocate_space.allocate_bank_tail(rom, start >> 15, len(stub), bank_free)
            rom[at : at + len(body)] = body
            a = convert_file_offset_to_snes(at)
            for o, v in zip(imm, (a & 0xFF, (a >> 8) & 0xFF, a >> 16)):
                rom[o] = v

            if stub_at is not None:
                rom[stub_at : stub_at + len(stub)] = stub
                for p in ptrs or []:
                    rom[p : p + 2] = (0x8000 | (stub_at & 0x7FFF)).to_bytes(2, "little")

            continue

        if ptrs and (len(laid) > slot or mode == "n" or (slot < 5 and mode in "vpc")):
            # Repoint oversized text and names through the original bank's free tail.
            # Short dialogue slots also use this path when they have native pointers.
            body = laid + bytes([0xFF])
            if mode in "vpcn":
                # VWF text: keep it in VWF space, and point at a jump stub in the bank (names too: a
                # string read from a non-VWF bank draws in the tile font, see the Data Library list)
                vat = vwf.allocate(len(body))
                rom[vat : vat + len(body)] = body
                body = encode_text_jump(vat) + bytes([0xFF])

            at = allocate_space.allocate_bank_tail(rom, start >> 15, len(body), bank_free)
            rom[at : at + len(body)] = body
            for p in ptrs:
                rom[p : p + 2] = (0x8000 | (at & 0x7FFF)).to_bytes(2, "little")

            continue

        if mode == "n" and slot < 5 and len(laid) <= slot:
            # names fit in place: terminate with FF (readers stop there; never pad with spaces,
            # which the 8px tile-font lists would draw as stray tiles)
            rom[start : start + len(laid)] = laid
            if len(laid) < slot:
                rom[start + len(laid)] = 0xFF

            continue

        if slot == 4 and mode in "vpc":
            # too short for an F8 06 jump: F8 20 lo hi jumps into bank $FD instead. The script
            # interpreter skips text by scanning for FD-FF, so keep those out of lo
            if (short.free & 0xFF) >= 0xFD:
                short.free = (short.free | 0xFF) + 1

            body = laid + encode_text_jump(end)
            at = short.allocate(len(body))
            rom[at : at + len(body)] = body
            a = 0x8000 | (at & 0x7FFF)
            if not (at >> 15 == 0x7D and (a & 0xFF) < 0xFD):
                raise AssertionError()

            rom[start:end] = bytes([0xF8, 0x20, a & 0xFF, a >> 8])
            continue

        if slot < 5 or (mode == "f" and len(laid) <= slot):
            # Keep payloads that cannot use relocation in their native fixed-width slot.
            # Code 01 pads unused bytes with display spaces.
            if not (len(laid) <= slot):
                raise AssertionError(f"{start:06X}: {text!r} does not fit its {slot}-byte slot")

            rom[start:end] = laid + bytes([1]) * (slot - len(laid))
            continue

        if len(pages) > 1:
            # Several pages. The event interpreter does not follow F8 06: after opcode 18 it scans
            # the original slot to the first FD/FE/FF and continues right after it, so a page break
            # inside the relocated text would drop every later page. Instead, chain real script
            # segments in the slot, the way the Japanese chains short texts:  F8 06 →page1, FD, 18
            # F8 06 →page2, FD, ... 18 F8 06 →last, 01.., <original terminator at end>. Each page
            # jumps back to its own terminator.
            if start == 0:
                raise AssertionError("Paginated text requires a preceding show-text opcode")
            op = rom[start - 1]
            if not (op == 0x18):
                raise AssertionError(
                    f"{start:06X}: paginated text not started by opcode 18 ({op:02X})"
                )

            # For each page, leave a native segment in the original slot and put
            # the visible payload in proportional-font storage. Its return jump
            # lands on that segment's own terminator, not the next page's text.
            # Five jump bytes per page, plus a wait and show-text opcode between
            # pages. Check the entire chain before writing its first payload.
            if 7 * len(pages) - 2 > slot:
                raise AssertionError(
                    f"{start:06X}: {len(pages)} pages do not fit the {slot}-byte slot"
                )
            pos = start
            for k, page in enumerate(pages):
                last = k == len(pages) - 1
                head = b"" if k == 0 else bytes([op])
                term = end if last else pos + len(head) + 5
                vat = vwf.allocate(len(page) + 5)
                rom[vat : vat + len(page) + 5] = page + encode_text_jump(term)
                seg = head + encode_text_jump(vat) + (b"" if last else b"\xfd")
                if not (pos + len(seg) <= end):
                    raise AssertionError(
                        f"{start:06X}: {len(pages)} pages do not fit the {slot}-byte slot"
                    )

                rom[pos : pos + len(seg)] = seg
                pos += len(seg)

            rom[pos:end] = b"\x01" * (end - pos)
            continue

        # General relocation uses a five-byte F8 06 jump in the original slot.
        # Names terminate with FF for raw-copy readers; other payloads jump back
        # to the original terminator so event execution resumes at the right byte.
        body = laid + (bytes([0xFF]) if mode == "n" else encode_text_jump(end))
        at = (fixed if mode == "f" else vwf).allocate(len(body))
        rom[at : at + len(body)] = body
        rom[start:end] = encode_text_jump(at) + b"\x01" * (slot - 5)

    return vwf.free
