"""
This module converts English translation strings into the glyph codes and command bytes the game reads.

encode_text() maps letters and punctuation to the Latin font's byte codes.
It turns {SHU} and {ERSKINE} into commands that insert the player's saved given
and family names, and accepts hexadecimal brace tokens for raw bytes. Newlines
become line-break bytes, three dots become one ellipsis glyph, and ~ becomes
the native wave-dash glyph. Unsupported characters or invalid brace tokens fail.

The returned bytes have no added terminator, word wrapping, or automatic page
breaks; insert_text supplies layout and ROM placement. The shared M table also
contains {il}, {ki}, and {ne} ligature codes used by patch_naming.encode_name();
encode_text() does not accept those ligature spellings.
"""

from __future__ import annotations

# Source character/token -> native font code. Code 00 is a line break;
# brace-delimited ligatures occupy one glyph code in fixed-size name buffers.
# encode_text() treats braces as commands; patch_naming.encode_name() uses the
# ligature entries directly when encoding default names.
M = {
    " ": 0x01,
    "?": 0xAE,
    "!": 0xAF,
    "'": 0xB0,
    '"': 0xB3,
    "-": 0xEE,
    ",": 0xB7,
    ".": 0xB8,
    "(": 0xB5,
    ")": 0xB6,
    "+": 0xED,
    "=": 0xEF,
    "{il}": 0xF0,
    "%": 0xF1,
    "/": 0xF2,
    "{ki}": 0xF3,
    "{ne}": 0xF7,
    ":": 0xF4,
    "&": 0xB4,
    "\n": 0x00,
}

# Digits occupy ten consecutive codes beginning at A3.
for i in range(10):
    M[str(i)] = 0xA3 + i

# Uppercase and lowercase letters each occupy a separate contiguous range.
for i in range(26):
    M[chr(65 + i)] = 0xB9 + i
    M[chr(97 + i)] = 0xD3 + i

# F8 0C inserts the saved hero name: selector 01 given name, 00 family name.
NAMES = {"{SHU}": "F80C01", "{ERSKINE}": "F80C00"}


def encode_text(s: str) -> bytes:
    """
    Encode English text and supported name/raw-hex tokens into game bytes.

    Encoding does not wrap or paginate text.

    Args:
        s (str): English text with {SHU}, {ERSKINE} or raw hexadecimal brace
            tokens containing at least one byte; newlines are encoded without
            wrapping. Name ligatures {il}, {ki} and {ne} are not accepted here.

    Returns:
        bytes: Encoded glyphs and control bytes, without an added text
            terminator.

    Raises:
        KeyError: An input character has no game encoding.
        ValueError: A brace token is empty, unterminated or contains invalid
            hexadecimal data.
    """
    # Expand symbolic name insertions to the same raw command syntax as hex tokens.
    for k, v in NAMES.items():
        s = s.replace(k, "{" + v + "}")

    # Scan the expanded source left to right. A token consumes all of its
    # markup; ordinary characters consume one source character.
    out = bytearray()
    i = 0
    while i < len(s):
        # Brace contents are raw bytes, not visible characters. Structural validation
        # of those commands belongs to validate_translation.
        if s[i] == "{":
            j = s.index("}", i)
            token = bytes.fromhex(s[i + 1 : j])
            # Empty markup must not silently disappear from the compiled text.
            if not token:
                raise ValueError("A hexadecimal brace token must contain at least one byte")
            out += token
            i = j + 1
            continue

        # The wave dash uses the native two-byte glyph code F9 F4.
        if s[i] == "~":
            out += b"\xf9\xf4"
            i += 1
            continue

        # Three source dots occupy one native ellipsis glyph, not three periods.
        if s.startswith("...", i):
            out += b"\xf9\xc9"
            i += 3
            continue

        # Remaining characters use the one-byte Latin/punctuation mapping.
        out.append(M[s[i]])
        i += 1

    return bytes(out)
