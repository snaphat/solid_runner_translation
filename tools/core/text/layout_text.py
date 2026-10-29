"""
This module wraps translated text into encoded lines, dialogue pages, and menu columns.

wrap_text() measures encoded words against a pixel limit, preserves explicit
line and page breaks, and optionally distributes lines across dialogue pages
of at most three rows. encode_vwf_line() selects proportional rendering before
a leading runtime insertion. encode_menu_layout() places labels at authored
12-pixel cell positions using encoded spacing.

The functions return encoded text without a final FF terminator. They use
caller-supplied ROM font bytes for measurement, leave the ROM unchanged, and
do not allocate storage or update game pointers.
"""

from __future__ import annotations

import re

from core.text import measure_text
from core.text.encode_text import encode_text

MAX_W = 236  # usable pixels per line (30 tiles = 240, keep a margin)
LINES = 3  # Dialogue rows per page before an automatic page break.
PAGE = b"\xfd\x18"  # FD waits/ends the page; event opcode 18 starts the next text.


def wrap_text(
    rom: bytes | bytearray,
    text: str,
    max_w: int = MAX_W,
    paginate: bool = True,
    *,
    name_width: int | None = None,
) -> bytes:
    """
    Wrap English text to its pixel limit and optionally split it into dialogue pages.

    Uses 00 for newlines and FD/18 between pages. Automatic pagination
    distributes lines evenly across pages of at most three rows: four lines
    become two pages of two. Explicit form feeds create page breaks even when
    automatic pagination is disabled.

    Args:
        rom (bytes | bytearray): Font tables used to fit words within
            max_w.
        text (str): English translation text, including supported
            brace-delimited control markup. Literal newlines end authored lines;
            form feeds start new pages. Ordinary spaces supply wrap points, and
            words are never split. Leading spaces on nonblank authored lines
            are retained; a whitespace-only line becomes an empty line.
        max_w (int, optional): Positive integer maximum line width in pixels;
            booleans and other nonintegers are rejected.
        paginate (bool, optional): Enable automatic pagination when True;
            explicit page breaks are preserved in either mode. Must be a boolean.
        name_width (int | None, optional): Pixel space reserved for each item or
            opponent name inserted while the game displays a line. Word-fit
            decisions and final line checks use this same reservation. Supply
            the nonnegative result of measure_text.measure_name_width(). None
            is allowed when the text contains no F8/12 or F8/18 name commands.

    Returns:
        bytes: Encoded text with layout controls; no final FF terminator is
            added.

    Raises:
        ValueError: max_w is not a positive integer, or a required system-name
            width is absent or invalid, a raw brace token is malformed, or a
            measured command or font record is invalid. Booleans are not valid widths.
        KeyError: A source character has no game encoding.
        TypeError: paginate is not a boolean.
        AssertionError: A wrapped line still exceeds its pixel-width allowance.
    """
    if type(max_w) is not int or max_w <= 0:
        raise ValueError("Text width must be a positive integer")
    if type(paginate) is not bool:
        raise TypeError("paginate must be a boolean")

    # Spaces and line breaks inside raw hexadecimal markup separate bytes, not
    # words or paragraphs. Encode complete tokens before splitting layout text
    # so wrapping cannot cut a valid command in half.
    text = re.sub(r"\{[^{}]*\}", lambda match: "{" + encode_text(match.group()).hex() + "}", text)
    pages: list[list[str]] = []
    # Every candidate line uses the same name reservation. Reusing insertion
    # measurements keeps word-fit decisions and final line checks consistent.
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
                    and measure_text.measure_text_width(
                        rom, encode_vwf_line(cand), insert_sizes, name_width=name_width
                    )
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
            w = measure_text.measure_text_width(
                rom, encode_vwf_line(line), insert_sizes, name_width=name_width
            )
            if not (w <= max_w):
                raise AssertionError(f"word too wide ({w}px): {line!r}")

    return PAGE.join(b"\x00".join(encode_vwf_line(l) for l in p) for p in pages)


def encode_vwf_line(line: str) -> bytes:
    """
    Encode one wrapped line with VWF setup before a leading runtime insertion.

    A leading name or number insertion would otherwise select the fixed-width
    font. A blank one-pixel FCF3 glyph switches the renderer to variable-width
    mode before the inserted text is drawn.

    This encoder does not validate raw token structure or enforce line width;
    wrap_text() performs those checks while laying out the complete text.

    Args:
        line (str): One decoded text line without a newline terminator.

    Returns:
        bytes: Encoded text with layout controls; no final FF terminator is
            added.

    Raises:
        KeyError: A source character has no game encoding.
        ValueError: A raw hexadecimal brace token is malformed.
    """
    b = encode_text(line)
    if b[:1] == b"\xf8" and b[1:2] in (b"\x0c", b"\x12", b"\x18", b"\x04"):
        b = b"\xfc\xf3" + b

    return b


def encode_menu_layout(
    rom: bytes | bytearray,
    text: str,
    cells_total: int | None = None,
    *,
    name_width: int | None = None,
) -> bytes | bytearray:
    """
    Encode menu labels at their assigned fixed-width cell positions.

    For example, `@0:Sound@4:Text Speed@12:Controller` positions labels at 12
    pixels per cell. VWF spaces pad each label to its assigned column.

    Args:
        rom (bytes | bytearray): Font tables used to fit labels between
            menu columns.
        text (str): One or more newline-separated rows of @column:label pairs.
            Each row must begin with a marker and contain no text outside its
            labels. Columns are nonnegative, unique, and increasing; labels must
            be nonblank. Labels may contain brace-delimited game commands.
        cells_total (int | None, optional): Positive integer number of fixed-width
            cells reserved per menu row; booleans and other nonintegers are
            rejected. None uses a 99-cell final-label bound; earlier labels
            remain bounded by the following column marker.
        name_width (int | None, optional): Pixel space reserved when a menu label
            contains an F8/12 or F8/18 item/opponent-name command. It contributes
            to the label's width before positioning the next column. Supply a
            nonnegative result from measure_text.measure_name_width(). None is
            allowed when the labels contain no such name commands.

    Returns:
        bytes | bytearray: Encoded menu rows with explicit cell positioning,
            joined by 00 newline bytes. No final FF is added.

    Raises:
        ValueError: cells_total is neither None nor a positive integer (booleans
            are rejected), menu column markup is malformed, or a required
            system-name width is absent or invalid, or a raw token/font record
            fails encoding or measurement.
        KeyError: A label character has no game encoding.
        AssertionError: A label overlaps its predecessor or exceeds its
            assigned pixel width.
    """
    if cells_total is not None and (type(cells_total) is not int or cells_total <= 0):
        raise ValueError("Menu row width must be a positive integer")
    if "\n" in text:  # several menu rows: lay out each, joined by 00 newlines
        return b"\x00".join(
            encode_menu_layout(rom, line, cells_total, name_width=name_width)
            for line in text.split("\n")
        )

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
        # A runtime name in a label consumes its reserved pixel width before
        # padding or overlap checks place the next selectable label.
        x += measure_text.measure_text_width(rom, b, name_width=name_width)
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
