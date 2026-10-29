"""
This module loads the editable Latin font bitmaps from assets/latin-font.json and validates their dimensions.

Importing this module reads the JSON file and checks for duplicate keys,
required sections, and valid bitmap dimensions before filling G and MAIN_PUNCT.
G maps character or ligature names to eleven bitmap rows. MAIN_PUNCT stores
multiline punctuation bitmaps; # marks ink and other characters are background.

load_font_artwork() can also validate another file and return its artwork
without changing those tables. register_glyph_artwork() adds or replaces one
bitmap in G. Other modules measure or install these bitmaps; this module does
not write them into a ROM. The font_tables module defines the ROM byte offsets
used to install or measure the two font tables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict, cast


class FontArtwork(TypedDict):
    """
    Validated artwork from the editable Latin font JSON file.

    Attributes:
        glyphs (dict[str, list[str]]): Character or ligature keys mapped to eleven
            bitmap row strings, each at most sixteen pixels wide.
        punctuation (dict[str, str]): Fixed-width punctuation keys mapped to
            LF-separated bitmap strings with the same row limits.
    """

    glyphs: dict[str, list[str]]
    punctuation: dict[str, str]


def _read_unique_artwork_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """
    Decode a JSON object without discarding duplicate artwork definitions.

    Args:
        pairs (list[tuple[str, object]]): JSON object members in file order.

    Returns:
        dict[str, object]: Members indexed by their unique keys.

    Raises:
        ValueError: An object repeats a key.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate font artwork key: {key!r}")
        result[key] = value
    return result


def _validate_artwork_rows(value: object, label: str) -> list[str]:
    """
    Validate and copy the eleven rows of one JSON glyph bitmap.

    Args:
        value (object): JSON value expected to be an array of row strings.
        label (str): Section and character key included in validation errors.

    Returns:
        list[str]: Eleven independent row entries, at most sixteen characters each.

    Raises:
        ValueError: The value is not an eleven-row array, or a row is not a
            single-line string of at most sixteen characters.
    """
    if not isinstance(value, list):
        raise ValueError(f"{label}: artwork must be an array of eleven row strings")
    rows = cast(list[object], value)
    if len(rows) != 11:
        raise ValueError(f"{label}: artwork must be an array of eleven row strings")
    if any(not isinstance(row, str) or len(row) > 16 or "\n" in row or "\r" in row for row in rows):
        raise ValueError(f"{label}: each bitmap row must be a string of at most sixteen pixels")
    return cast(list[str], rows).copy()


def load_font_artwork(path: str | os.PathLike[str]) -> FontArtwork:
    """
    Read both font sections and validate all artwork before returning it.

    Args:
        path (str | os.PathLike[str]): UTF-8 font JSON file containing glyphs
            as arrays of row strings and punctuation as multiline strings.

    Returns:
        FontArtwork: Checked glyph and punctuation dictionaries. Loading does
            not change the module's registered artwork. CRLF line endings in
            punctuation strings are normalized to LF.

    Raises:
        OSError: The input file cannot be read.
        ValueError: JSON is malformed, keys repeat, sections are not objects,
            keys are empty, or artwork values have invalid types or bitmap shapes.
    """
    raw: object = json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=_read_unique_artwork_object
    )
    if not isinstance(raw, dict):
        raise ValueError("Font artwork must be an object with glyphs and punctuation sections")
    sections = cast(dict[str, object], raw)
    glyphs: dict[str, list[str]] = {}
    punctuation: dict[str, str] = {}
    for section in ("glyphs", "punctuation"):
        entries = sections.get(section)
        if not isinstance(entries, dict):
            raise ValueError(f"Font {section} section must be an object")
        for key, value in cast(dict[str, object], entries).items():
            if not key:
                raise ValueError(f"Font {section} keys must not be empty")
            label = f"{section}[{key!r}]"
            if section == "glyphs":
                glyphs[key] = _validate_artwork_rows(value, label)
            else:
                if not isinstance(value, str):
                    raise ValueError(f"{label}: punctuation artwork must be a multiline string")
                value = value.replace("\r\n", "\n")
                _validate_artwork_rows(value.strip("\n").split("\n"), label)
                punctuation[key] = value
    return {"glyphs": glyphs, "punctuation": punctuation}


# Character or ligature token -> eleven bitmap rows; # marks ink and .
# marks background in the external font artwork.
G: dict[str, list[str]] = {}


def register_glyph_artwork(ch: str, art: str) -> None:
    """
    Register eleven artwork rows under ch in G.

    Args:
        ch (str): Artwork character key stored in the module-level G table.
        art (str): Eleven LF- or CRLF-separated bitmap rows; # is ink and other
            characters are background. Exactly eleven rows preserve empty edge
            rows. Extra surrounding empty lines are treated as string framing,
            removing paired outer lines first, then trailing and leading extras.

    Raises:
        AssertionError: The artwork does not contain exactly eleven rows,
            contains a bare carriage return, or a row exceeds the renderer's
            16-pixel bitmap width.
    """
    rows = art.replace("\r\n", "\n").split("\n")

    # An empty row is a blank bitmap scanline, not automatically string framing.
    # Remove outer blank lines only while the string has more than eleven rows.
    first, last = 0, len(rows)
    while last - first > 12 and rows[first] == rows[last - 1] == "":
        first += 1
        last -= 1
    while last - first > 11 and rows[last - 1] == "":
        last -= 1
    while last - first > 11 and rows[first] == "":
        first += 1
    rows = rows[first:last]

    # The renderer reads exactly eleven bitmap rows per glyph; malformed
    # artwork would otherwise omit rows or overwrite the next glyph's bytes.
    if not (len(rows) == 11):
        raise AssertionError((ch, len(rows)))
    if any(len(row) > 16 for row in rows):
        raise AssertionError((ch, "Artwork rows must fit sixteen bitmap pixels"))
    if any("\r" in row for row in rows):
        raise AssertionError((ch, "Artwork must use LF or CRLF line endings"))

    # Store artwork by source character/token. Installation resolves that
    # key through the encoder rather than assuming JSON order equals glyph order.
    G[ch] = rows


_ART = load_font_artwork(Path(__file__).resolve().parents[3] / "assets/latin-font.json")
# Both sections have passed validation before the registry changes. Copy row
# arrays directly so an empty first or last row remains a legitimate blank row.
G.update({character: rows.copy() for character, rows in _ART["glyphs"].items()})

# Punctuation bitmap rows installed into the native fixed-width font.
MAIN_PUNCT: dict[str, str] = _ART["punctuation"]
