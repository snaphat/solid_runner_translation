"""
This module reads encoded text buffers as glyphs and commands and follows the addresses stored in text-jump commands.

The functions take bytes or bytearray buffers. They determine how many bytes
each token occupies, convert glyph codes to font indexes, and walk a string
through its first FD, FE, or FF terminator, optionally following relocation
jumps. Truncated tokens, unsupported F8 commands, and cyclic jumps raise errors.

parse_encoded_text() splits an entire encoded buffer without following jumps
or stopping at terminators. split_dialogue_pages() uses those tokens to split
FD/18 page boundaries without confusing command operands with page markers.
Layout and compiled-text readers share these parsing rules. This module does
not load translation files, change buffers, or enforce name-copy restrictions.
"""

from collections.abc import Iterator

from core.rom.convert_addresses import convert_pointer_bytes_to_file

# Operand counts exclude the F8 prefix and subopcode. The native event
# interpreter reads these counts at bus address $84:AA3B + 2 * subopcode.
# F8 06 carries a full address. The F8 20 extension in asm/vwf.asm carries
# a two-byte address within CPU bank $FD and consumes two operands.
F8_OPERAND_LENGTHS = {
    0x00: 0,
    0x02: 2,
    0x04: 4,
    0x06: 3,
    0x08: 3,
    0x0A: 1,
    0x0C: 1,
    0x0E: 0,
    0x10: 1,
    0x12: 0,
    0x14: 0,
    0x16: 1,
    0x18: 0,
    0x1A: 1,
    0x1C: 1,
    0x1E: 0,
    0x20: 2,
}


def _validate_text_buffer(data: object) -> None:
    """
    Require a byte buffer before interpreting values as native text codes.

    Args:
        data (object): Input that must be a bytes or bytearray text/ROM buffer.

    Raises:
        TypeError: data is neither bytes nor bytearray.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("Text data must be bytes or bytearray")


def read_text_token_size(data: bytes | bytearray, offset: int) -> int:
    """
    Read one complete token's byte length without following a text jump.

    Args:
        data (bytes | bytearray): Encoded text or a ROM containing it.
        offset (int): Token boundary as an integer byte offset into data;
            booleans and other noninteger values are rejected.

    Returns:
        int: Length including the prefix, subopcode and operands.

    Raises:
        TypeError: data is not bytes or bytearray, or offset is not an integer
            or is a boolean.
        IndexError: The offset is outside data or a token is truncated.
        KeyError: An F8 subopcode is unsupported; args contains (subopcode, offset).
    """
    _validate_text_buffer(data)
    if type(offset) is not int:
        raise TypeError("Token offset must be an integer; booleans are not offsets")
    if not 0 <= offset < len(data):
        raise IndexError(f"outside ROM at {offset:06X}")

    # Ordinary glyphs, newline 00 and terminators FD-FF occupy one byte.
    # F8 commands and F9-FC glyphs carry operands that must stay with them.
    code = data[offset]
    size = 1
    if code == 0xF8:
        if offset + 1 >= len(data):
            raise IndexError(f"truncated F8 at {offset:06X}")
        opcode = data[offset + 1]

        # An unknown command has no trustworthy length; guessing would
        # let its operands become unrelated glyphs or terminators.
        if opcode not in F8_OPERAND_LENGTHS:
            raise KeyError(opcode, offset)
        size = 2 + F8_OPERAND_LENGTHS[opcode]
    elif 0xF9 <= code <= 0xFC:
        size = 2

    # Check the full token before callers slice it: Python slices would
    # silently accept missing operands at the end of the buffer.
    if offset + size > len(data):
        raise IndexError(f"truncated code at {offset:06X}")
    return size


def decode_glyph_index(data: bytes | bytearray, offset: int) -> int:
    """
    Convert a one-based glyph code to a zero-based font index.

    Codes 01-F7 select indices 0-246. Prefixes F9-FC extend the glyph code:
    ((prefix - F8) * 256 + operand) - 1 gives the font index. For example,
    F9 01 selects index 256. The operand is part of the glyph even when it
    equals a control or terminator byte. Native name-copy restrictions are
    checked separately.

    Args:
        data (bytes | bytearray): Encoded text containing the complete glyph.
        offset (int): Nonnegative integer byte offset at a glyph boundary,
            within data; booleans and other noninteger values are rejected.
            The newline code 00 is also accepted and produces the layout sentinel -1.

    Returns:
        int: Font index, or -1 for newline.

    Raises:
        TypeError: data is not bytes or bytearray, or offset is not an integer
            or is a boolean.
        IndexError: offset is outside data or an extended glyph lacks its operand.
        ValueError: The byte at offset is the F8 command prefix or an FD-FF terminator.
    """
    _validate_text_buffer(data)
    if type(offset) is not int:
        raise TypeError("Glyph offset must be an integer; booleans are not offsets")
    if not 0 <= offset < len(data):
        raise IndexError(f"Glyph offset outside data: {offset}")

    # These reserved first bytes control text flow rather than select
    # font artwork. Newline 00 remains valid as the layout sentinel -1.
    code = data[offset]
    if code == 0xF8 or code >= 0xFD:
        raise ValueError(f"Text control {code:02X} at {offset} is not a glyph")

    if 0xF9 <= code <= 0xFC:
        if offset + 1 >= len(data):
            raise IndexError(f"Truncated extended glyph at {offset}")
        # The prefix selects a 256-code group and the operand selects
        # its member. Do not interpret reserved values inside that operand.
        return (((code - 0xF8) << 8) | data[offset + 1]) - 1

    # Font tables are zero-based; encoded character codes are one-based.
    return code - 1


def walk_text_tokens(
    data: bytes | bytearray,
    start: int,
    follow_jumps: bool = True,
    max_tokens: int | None = None,
) -> Iterator[tuple[int, int]]:
    """
    Yield token offsets and lengths through the first FD, FE or FF terminator.

    Validation occurs as the iterator advances. A yielded jump command does
    not establish that its destination is valid; advancing again checks that
    destination. Mutable data must remain unchanged during iteration.

    Args:
        data (bytes | bytearray): ROM containing text and any relocation targets.
        start (int): First token's integer file offset; booleans and other
            noninteger values are rejected. No repair-specific adjustment is applied.
        follow_jumps (bool, optional): Follow F8 06 and F8 20 addresses when true;
            False skips their operands as the event interpreter does. Only
            boolean values are accepted.
        max_tokens (int | None, optional): Nonnegative integer token limit,
            including the terminator; booleans and noninteger values are rejected.
            Zero permits no tokens and immediately exhausts the limit. None
            imposes no limit beyond bounds and cycle detection.

    Yields:
        tuple[int, int]: Token file offset and complete byte length, including
            the final terminator and any followed jump commands.

    Raises:
        TypeError: data is not bytes or bytearray, start is not an integer or
            is a boolean, or follow_jumps is not a boolean.
        ValueError: max_tokens is neither None nor a nonnegative integer, a token
            boundary repeats, or the limit is exhausted before reaching a terminator.
        IndexError: A token or jump target is outside data or truncated.
        KeyError: An F8 subopcode is unsupported.
    """
    _validate_text_buffer(data)
    if type(start) is not int:
        raise TypeError("Text start must be an integer; booleans are not offsets")
    if type(follow_jumps) is not bool:
        raise TypeError("follow_jumps must be a boolean")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 0):
        raise ValueError(f"max_tokens must be a nonnegative integer or None, got {max_tokens!r}")

    offset = start
    seen: set[int] = set()
    # Count token boundaries, not bytes: operands do not consume extra
    # budget, and revisiting a boundary means the jump chain is cyclic.
    while max_tokens is None or len(seen) < max_tokens:
        if offset in seen:
            raise ValueError(f"cyclic jump at {offset:06X}")
        seen.add(offset)

        size = read_text_token_size(data, offset)
        yield offset, size

        # Yield the terminator itself so callers can identify the native
        # event boundary, then stop before interpreting following event bytes.
        code = data[offset]
        if code >= 0xFD:
            return

        # F8 06 supplies low/high/bank bytes; F8 20 omits the bank and
        # always targets $FD. Interpreter scans skip both commands
        # without following their targets when follow_jumps is false.
        if code == 0xF8 and follow_jumps and data[offset + 1] in (0x06, 0x20):
            bank = data[offset + 4] if data[offset + 1] == 0x06 else 0xFD
            offset = convert_pointer_bytes_to_file(data[offset + 2], data[offset + 3], bank)
        else:
            offset += size

    raise ValueError("runaway")


def parse_encoded_text(data: bytes | bytearray) -> list[bytes]:
    """
    Split encoded text into complete glyph and command tokens.

    Uses the shared binary token lengths so malformed raw markup cannot
    borrow missing operands from relocation bytes appended during insertion.

    Args:
        data (bytes | bytearray): Encoded text payload. Terminators and relocation
            commands are retained as tokens; parsing continues through the whole
            buffer without following jump targets.

    Returns:
        list[bytes]: Complete tokens in source order, retaining command operands.

    Raises:
        TypeError: data is neither bytes nor bytearray.
        ValueError: An F8 subopcode is unsupported, or a command or two-byte
            glyph is missing required bytes.
    """
    _validate_text_buffer(data)
    tokens: list[bytes] = []
    offset = 0
    # Source validation consumes the entire payload in storage order.
    # Unlike a ROM walk, it neither follows jumps nor stops at terminators.
    while offset < len(data):
        try:
            size = read_text_token_size(data, offset)

        # Present malformed source as ValueError with its payload offset;
        # ROM traversal retains the lower-level errors for reader diagnostics.
        except KeyError as error:
            raise ValueError(
                f"Unsupported F8 subopcode {error.args[0]:02X} at encoded byte {offset}"
            ) from error
        except IndexError as error:
            raise ValueError(f"Truncated text token at encoded byte {offset}") from error

        tokens.append(bytes(data[offset : offset + size]))
        offset += size

    return tokens


def split_dialogue_pages(data: bytes | bytearray) -> list[bytes]:
    """
    Split encoded dialogue at FD/18 boundaries without splitting token operands.

    Args:
        data (bytes | bytearray): Encoded dialogue before relocation jumps are
            appended. Empty pages are retained, including a final empty page.

    Returns:
        list[bytes]: Page payloads in order, excluding their FD/18 separators.
            At least one page is returned, even for empty input.

    Raises:
        TypeError: data is neither bytes nor bytearray.
        ValueError: A glyph or command is truncated, or an F8 subopcode is unsupported.
    """
    tokens = parse_encoded_text(data)
    pages = [bytearray()]
    index = 0
    while index < len(tokens):
        # FD is the wait/segment terminator; 18 starts the next segment. The
        # same bytes inside an address or extended glyph do not control paging.
        if tokens[index] == b"\xfd" and tokens[index + 1 : index + 2] == [b"\x18"]:
            pages.append(bytearray())
            index += 2
        else:
            pages[-1].extend(tokens[index])
            index += 1
    return [bytes(page) for page in pages]
