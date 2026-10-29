"""
This module checks an English translation entry for encoding and layout problems and returns diagnostic messages.

validate_translation_entry() takes Japanese and English text, the entry's
storage/layout metadata, and ROM font bytes. It returns a list of diagnostics
for missing text, invalid markup, changed command counts, omitted model codes,
byte-capacity violations, and text that does not fit its display rules. It uses
the compiler's wrapping and menu layout helpers without changing the entry or ROM.

Brace commands insert saved names, pause text, or emit raw bytes. Comparing
Japanese and English command types and counts helps catch commands dropped or
duplicated during editing. reviewed_controls metadata supplies an explicit
English command list where intentional changes are needed, such as replacing
a repeated name with "you" or adding alignment spaces.

The checks use known window limits and entry metadata. They do not judge
translation meaning or inspect rendered gameplay screenshots.
"""

import collections
import re
import unicodedata
from typing import TypedDict

from core.text import encode_text as text_encoding
from core.text import get_text_limits as text_limits
from core.text import layout_text, measure_text
from core.text.read_text_tokens import parse_encoded_text as parse_encoded_text
from core.text.read_text_tokens import split_dialogue_pages
from core.text.read_translation_files import TextMode, TranslationEntry


class EntryLayout(TypedDict):
    """
    Resolved storage and display rules used when validating a source entry.

    Created by derive_entry_layout_constraints() and consumed by
    encode_entry_layout(). The latter uses the compiler's layout helpers, but
    this record does not allocate ROM space or direct insertion itself.

    Attributes:
        mode (TextMode): Source mode code; see derive_entry_layout_constraints
            for the behavior of each code.
        width (int | None): Width in pixels, or cells for menus; None when no
            width is derived for the mode.
        lines (int | None): Visible row allowance; None when no limit is known.
        slot (int): Original storage capacity in bytes, excluding its terminator.
        category (str | None): Known text-window category, or None when unclassified.
    """

    mode: TextMode
    width: int | None
    lines: int | None
    slot: int
    category: str | None


# Japanese hero-name tokens -> equivalent English tokens. Normalizing these
# spellings allows name insertions to be compared across the two source formats:
# {名}/{SHU} inserts the saved given name; {姓}/{ERSKINE} inserts the family name.
NAME_TOKENS: dict[str, str] = {"{名}": "{SHU}", "{姓}": "{ERSKINE}"}

# Entry IDs requiring an explicit English command list in reviewed_controls
# metadata, rather than deriving it from Japanese. IDs are hexadecimal original-ROM
# text offsets. For example, 0F5D78 replaces one name insertion with "you", while
# 03DAFA adds {01} space bytes to align its poster attribution.
# English command types and counts must match that list; an empty
# list allows none. Membership makes missing metadata an error, even for entries
# whose English commands now match Japanese. It does not exempt them from validation.
KNOWN_CONTROL_DIFFERENCES: set[str] = set(
    """09E944 0B110E 0F5D78 0F5DB2 12A86B 16E0D8 03DAFA 03DBB6 03DC2F 03DCA9 03DD46
03DDBD 03DE6E 03DEE9 03DF59 03DFE4 03E0A3 03E157 03E21E 03E29C 03E342 03E3C1 03E472 03E530 03E5A9 03E643
03E6BF 03E91E 09C016""".split()
)

# Name-entry IDs requiring max_name_bytes metadata because their English names
# exceed the default eight encoded bytes: 03EF00 = Sr.Rachel (9), 03EF28 = Weissheit
# (9), 03EF34 = Anna Kelly (10). The metadata sets each byte limit and separately
# allows Anna Kelly's space through name_spaces. Membership makes a missing limit
# an error. Names must still fit that limit, obey the spacing rule and occupy one line.
KNOWN_LONG_NAMES: set[str] = {"03EF00", "03EF28", "03EF34"}


def missing_name_identifiers(japanese: str, english: str) -> list[str]:
    """
    Find literal model codes/numbers omitted from translated name records.

    Kana can be translated or romanized, but source Latin identifiers and model
    numbers must survive. Letters may remain attached to romanized names and
    separators may change, but each digit sequence must match in full.
    Full-width letters and numbers are normalized before comparison.

    Args:
        japanese (str): Source name containing literal identifiers and kana.
        english (str): Translated name to check without changing its spelling.

    Returns:
        list[str]: Source identifiers or model numbers missing from English.
    """
    japanese = unicodedata.normalize("NFKC", japanese)
    english = unicodedata.normalize("NFKC", english)
    codes = re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-ー−][A-Za-z0-9]+)*|\d+", japanese)
    numbers = re.findall(r"\d+", english)
    missing: list[str] = []
    for code in codes:
        if code.isdigit():
            present = code in numbers
        else:
            # Separators may occur between letters or digit groups, but a model
            # number cannot match only the prefix of a larger number. Keeping
            # digit groups intact also distinguishes RX-78-2 from RX-782.
            parts = re.findall(r"[A-Z]|\d+", code.upper())
            pattern = r"[^A-Z0-9]*".join(
                rf"(?<![0-9]){part}(?![0-9])" if part.isdigit() else part for part in parts
            )
            present = re.search(pattern, english.upper()) is not None
        if not present:
            missing.append(code)
    return missing


def extract_text_commands(s: str) -> list[str]:
    """
    Extract embedded command markup so translations can be checked for lost commands.

    The validator compares these strings between Japanese and English, or
    against an explicit English command list. For example, "Hi {SHU}{F81A3C}"
    produces ["{SHU}", "{F81A3C}"]. Repeated commands remain repeated.

    Args:
        s (str): Japanese or English source text containing brace-delimited
            markup, ordinary dialogue, or both.

    Returns:
        list[str]: Matching nonempty {...} substrings in source order,
            including braces. This extraction does not establish that commands
            are supported or that all braces in the input are balanced.
    """
    return re.findall(r"\{[^{}]+\}", s)


def derive_entry_layout_constraints(e: TranslationEntry) -> EntryLayout:
    """
    Derive an entry's layout, slot and window constraints.

    Source metadata uses these mode codes:

        v: Dialogue; wrap to pixel width and paginate after three rows (default).

        p: Prose; wrap to pixel width within one window, without pagination.

        n: Name; compiler uses an FF-terminated record, while validation requires
            glyphs on one line within the name's encoded-byte allowance.

        f: Fixed record; preserve authored line breaks and enforce per-row byte limits.

        m: Menu; position labels at @column: markers, with width in 12-pixel cells.

        c: Banner; preserve authored line breaks and center within 240 pixels
            during compilation; validate width and the explicit row allowance.

    Args:
        e (TranslationEntry): Source entry with hexadecimal id/end bounds, jp
            source text, and optional mode, w, and max_lines overrides. Pass
            metadata loaded by read_translation_files.load(); this function
            checks selected constraints rather than the complete metadata schema.

    Returns:
        EntryLayout: Resolved mode, width, row allowance, slot capacity and
            text-window category.

    Raises:
        ValueError: The mode is unsupported, hexadecimal slot bounds are invalid
            or reversed/outside LoROM, or an explicit w, max_lines, or
            max_name_bytes is not a positive integer.
    """
    # Missing mode metadata selects the ordinary paginated dialogue layout.
    mode = e.get("mode", "v")
    if mode not in ("v", "p", "n", "f", "m", "c"):
        raise ValueError(f"Unsupported text mode: {mode!r}")

    # id and end are hexadecimal file offsets in the unheadered original ROM.
    # end identifies the terminator; the writable slot excludes that byte.
    slot = int(e["end"], 16) - int(e["id"], 16)
    if not 0 <= int(e["id"], 16) <= int(e["end"], 16) < 0x400000:
        raise ValueError("Text slot is outside the ROM or has reversed bounds")
    for field in ("w", "max_lines", "max_name_bytes"):
        value = e.get(field)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{field} must be a positive integer")

    # w overrides the layout width: pixels except for menus, which use cells.
    width = e.get("w")

    # Explicit widths override the source-derived defaults. Wrapped prose and
    # names estimate pixels from 12-pixel Japanese characters, bounded to the
    # text window; menus instead retain the source row's character-column count.
    # 96 is the minimum inferred width; 236 is the ordinary dialogue width.
    if width is None:
        longest = max(len(line) for line in e["jp"].split("\n"))
        if mode in "pn":
            width = min(236, max(96, longest * 12))
        elif mode == "m":
            width = longest
        elif mode == "v":
            width = 236

    # Dialogue can continue over three-line pages. Prose and banners have a
    # finite visible area, so their line allowance comes from the window table
    # or max_lines metadata rather than automatic pagination.
    # The original text address identifies known window types and their row limits.
    category = text_limits.classify_text_window(int(e["id"], 16))
    lines = 3 if mode == "v" else None
    if mode == "p" and category:
        lines = text_limits.LINE_LIMIT.get(category, e["jp"].count("\n") + 1)
    if mode in ("p", "c"):
        lines = e.get("max_lines", lines)

    return {"mode": mode, "width": width, "lines": lines, "slot": slot, "category": category}


def encode_entry_layout(
    text: str,
    rom: bytes | bytearray,
    c: EntryLayout,
    out: list[str],
    *,
    name_width: int | None = None,
) -> bytes | bytearray:
    """
    Encode text for validation using the compiler's wrapping and menu layout helpers.

    Menu entries receive column positioning. Dialogue receives word wrapping
    and three-line pages. Other wrapped text has no automatic page breaks;
    fixed text and centered banners are encoded directly. Banner centering
    padding and text relocation are added later by the compiler and are absent
    from this result. Font measurements come from the supplied ROM. The
    returned bytes are inspected by the validator rather than written into it.

    Args:
        text (str): English translation with embedded commands and any explicit
            newline, page-break or menu-column markup.
        rom (bytes | bytearray): ROM containing the font tables used to
            measure glyphs and runtime name/number insertions.
        c (EntryLayout): Entry layout constraints. mode selects the encoding strategy;
            width sets the wrapping or menu-column allowance. Obtain it with
            derive_entry_layout_constraints(); this encoder does not validate the
            constraint dictionary as a separate schema.
        out (list[str]): Existing validation diagnostics. A centered banner
            exceeding 240 pixels appends a message to this list.
        name_width (int | None, optional): Pixel reservation for F8/12 and F8/18
            item/opponent-name commands. Use the same nonnegative value supplied
            to insert_translations() so validation and compilation produce the
            same line breaks and menu positions. None is allowed when this
            layout does not need to measure an inserted item/opponent name.

    Returns:
        bytes | bytearray: Encoded text with the selected line breaks,
            page breaks and column padding.

    Raises:
        KeyError: A source character has no game encoding.
        ValueError: Markup, tokens, widths, menu columns, or measured font
            records are invalid, or an inserted item/opponent name has no width.
        AssertionError: A word or menu label does not fit its assigned width.
    """
    mode = c["mode"]

    # Use the same name reservation as ROM insertion. Otherwise validation could
    # accept a shorter line layout than the compiler will actually produce.
    # Menus encode explicit column positions instead of wrapping ordinary prose.
    if mode == "m":
        return layout_text.encode_menu_layout(rom, text, c["width"], name_width=name_width)

    # Banner centering adds padding during insertion; this width check concerns
    # the unpadded text. Banners and fixed records retain their authored breaks.
    if (
        mode == "c"
        and measure_text.measure_text_width(
            rom, text_encoding.encode_text(text), name_width=name_width
        )
        > 240
    ):
        out.append("Banner exceeds 240px")
    if mode in "fc":
        return text_encoding.encode_text(text)

    # Use the compiler's wrapper so byte and line checks see its actual breaks.
    # Only dialogue mode may introduce new pages when three lines are filled.
    return layout_text.wrap_text(
        rom, text, c["width"] or 236, paginate=mode == "v", name_width=name_width
    )


def validate_translation_entry(
    e: TranslationEntry, rom: bytes | bytearray, *, name_width: int | None = None
) -> list[str]:
    """
    Report source, command, byte-capacity and layout violations for one translation.

    Use entries loaded by read_translation_files.load(), which validates their
    metadata shapes. This function reports the encoding and display violations
    it handles as strings; it is not a general schema validator for arbitrary
    dictionaries. A truthy skip permits blank English but does not bypass the
    other checks, so callers normally omit skipped entries. Entry and ROM bytes
    are not changed, and an empty result does not establish translation accuracy.

    Args:
        e (TranslationEntry): Source text, original slot bounds and optional
            layout/control metadata.
        rom (bytes | bytearray): Candidate ROM used to check encoded text
            and window fit.
        name_width (int | None, optional): Pixel width reserved during compilation
            for an inserted item/opponent name. Supply the nonnegative result of
            measure_text.measure_name_width() using the candidate ROM font.
            Layout checks use this value when testing the entry's fit. None is
            allowed when the layout needs no item/opponent-name measurement;
            otherwise the returned diagnostics report the missing width.

    Returns:
        list[str]: Validation diagnostics; empty when all entry constraints
            pass.
    """
    text = e["en"]
    try:
        c = derive_entry_layout_constraints(e)
    except ValueError as ex:
        return [str(ex)]
    # IDs identify numeric ROM offsets. Letter case and omitted leading zeros
    # must not bypass rules attached to a particular source entry.
    entry_id = f"{int(e['id'], 16):06X}"
    mode = c["mode"]
    out: list[str] = []

    # A form feed separates source pages only outside complete brace tokens.
    # Within raw hexadecimal markup it is a byte separator, just like a space.
    source_pages = [""]
    for index, fragment in enumerate(re.split(r"(\{[^{}]*\})", text)):
        if index % 2:
            source_pages[-1] += fragment
        else:
            pieces = fragment.split("\f")
            source_pages[-1] += pieces[0]
            source_pages.extend(pieces[1:])
    literal_text = re.sub(r"\{[^{}]*\}", "", text)

    # Prose and banners cannot spill onto a new page. Require a known line
    # allowance before measuring fit, and allow blank English only when the
    # entry is explicitly marked skip in its metadata.
    if mode in ("p", "c") and not c["lines"]:
        out.append("Text requires a reviewed max_lines allowance in technical metadata")
    if mode in ("p", "c") and len(source_pages) > 1:
        out.append("Non-paginated text does not support page breaks")
    if not text.strip() and not e.get("skip"):
        out.append("English translation must not be blank")

    # Menu markers place choices at specific columns. Compare row by row so
    # editing the words cannot silently move, remove or reorder those positions.
    # For example, @0:Yes @6:No places labels at cells 0 and 6; menu_columns
    # stores the required list of column numbers for each row.
    if mode == "m":
        columns = [[int(x) for x in re.findall(r"@(\d+):", line)] for line in text.split("\n")]
        expected_columns = e.get("menu_columns")
        if not expected_columns or columns != expected_columns:
            out.append("Menu column positions must match menu_columns in technical metadata")

    # Compare command multiplicities, not their order: English may reorder name
    # insertions while retaining them. Intentional differences and entries with
    # no Japanese source require an explicit expected English command list.
    if "reviewed_controls" in e or entry_id in KNOWN_CONTROL_DIFFERENCES or not e["jp"]:
        if "reviewed_controls" not in e:
            out.append("Translation requires reviewed_controls in technical metadata")
        want = collections.Counter(e.get("reviewed_controls", []))
        if collections.Counter(extract_text_commands(text)) != want:
            out.append(
                "control tokens differ from the reviewed translation "
                + " ".join(sorted(want.elements()))
            )
    else:
        want = collections.Counter(NAME_TOKENS.get(t, t) for t in extract_text_commands(e["jp"]))
        if collections.Counter(extract_text_commands(text)) != want:
            out.append(
                "control tokens differ from the Japanese " + " ".join(sorted(want.elements()))
            )

    # Command extraction only returns complete brace pairs. Counting all braces
    # separately also catches stray or nested braces that extraction would omit.
    if text.count("{") != len(extract_text_commands(text)) or text.count("}") != len(
        extract_text_commands(text)
    ):
        out.append("Unbalanced braces")
    if mode != "m" and re.search(r"@\d+:", text):
        out.append("Menu markers only work in menu entries")

    try:
        # Validate raw command structure before wrapping can interpret operands.
        # Explicit page breaks are consumed by the wrapper, not encode_text.
        # Menus contain @column markers, so their payload is parsed after the
        # menu encoder removes those markers instead.
        if mode != "m":
            for page in source_pages:
                parse_encoded_text(text_encoding.encode_text(page))

        # Name records have a default eight-byte allowance. Longer names and
        # spaces require explicit metadata; wrapping or pagination cannot make
        # an oversized name valid.
        if mode == "n":
            missing = missing_name_identifiers(e["jp"], text)
            if missing:
                out.append("Name omits source model identifier(s): " + ", ".join(missing))
            if entry_id in KNOWN_LONG_NAMES and "max_name_bytes" not in e:
                out.append("Reviewed name requires max_name_bytes in technical metadata")
            limit = e.get("max_name_bytes", 8)
            if (
                "\n" in literal_text
                or len(source_pages) > 1
                or len(text_encoding.encode_text(text)) > limit
                or (" " in literal_text and not e.get("name_spaces"))
            ):
                out.append(
                    f"Names: at most {limit} encoded bytes, one line"
                    + ("" if e.get("name_spaces") else ", no spaces")
                )
        if mode == "f" and len(source_pages) > 1:
            out.append("Fixed text does not support page breaks")

        # Measure the encoded layout, including automatic wrapping. Inspect
        # tokens rather than raw zero bytes so command payloads cannot be
        # mistaken for newline tokens.
        laid = encode_entry_layout(text, rom, c, out, name_width=name_width)
        tokens = parse_encoded_text(laid)

        # A standalone 00 byte advances to the next displayed row.
        line_breaks = tokens.count(b"\0")

        # Raw markup can encode a segment boundary without a source form feed.
        # Inspect whole tokens so FD/FE/FF inside command operands or extended
        # glyphs cannot be mistaken for a boundary.
        if mode in ("p", "c") and any(token[0] >= 0xFD for token in tokens):
            out.append("Non-paginated text does not support page breaks or embedded terminators")

        if mode == "v":
            pages = split_dialogue_pages(laid)
            if len(pages) > 1:
                # Every page needs a five-byte jump in the original event
                # script, with FD/18 between pages. Standalone reference
                # payloads and four-byte slots cannot carry those segments.
                if c["slot"] < 5 or e.get("imm") or (e.get("ptrs") and len(laid) > c["slot"]):
                    out.append("Paginated dialogue requires native segments in the original slot")
                elif 7 * len(pages) - 2 > c["slot"]:
                    out.append(f"{len(pages)} dialogue pages do not fit the {c['slot']}-byte slot")

        # The name copy loops accept only glyphs, not text commands or breaks.
        # Inspect bytes as well as source characters: raw {00}/{01} markup can
        # encode a newline or space without spelling either character in text.
        # FD/FE/FF end text segments; the name reader also stops if FF appears
        # as the second byte of a glyph, so those byte pairs cannot form names.
        if mode == "n":
            if any(
                token[0] in (0x00, 0xF8, 0xFD, 0xFE, 0xFF) or (len(token) == 2 and token[1] == 0xFF)
                for token in tokens
            ):
                out.append("Names must contain only glyphs on one line")
            if b"\x01" in tokens and not e.get("name_spaces"):
                out.append("Names must not contain encoded spaces")

        # Fixed records reserve bytes per displayed row, regardless of whether
        # its newline was written literally or supplied through raw markup.
        if mode == "f":
            rows = [0]
            for token in tokens:
                if token == b"\0":
                    rows.append(0)
                else:
                    rows[-1] += len(token)
            limits = e.get("fixed_line_bytes")
            if not limits or len(rows) != len(limits):
                out.append(
                    "Fixed text line count must match fixed_line_bytes in technical metadata"
                )
            else:
                for number, (length, limit) in enumerate(zip(rows, limits), 1):
                    if length > limit:
                        out.append(f"Fixed text line {number} exceeds its {limit}-byte allowance")

        if mode in ("p", "c") and c["lines"] and line_breaks + 1 > c["lines"]:
            out.append(f"Prose exceeds {c['lines']} visible lines")

        # A normal relocation jump needs five bytes in the original slot.
        # Short dialogue/prose/banner records have dedicated insertion paths;
        # other short records must fit in place unless pointers or immediate
        # address operands let the compiler redirect them to allocated storage.
        # ptrs lists 16-bit text-pointer locations; imm lists the instruction
        # operands holding the address's low, high and bank bytes. These are
        # reference locations to patch, rather than extra text-storage capacity.
        if (
            c["slot"] < 5
            and mode not in "vpc"
            and not e.get("ptrs")
            and not e.get("imm")
            and len(laid) > c["slot"]
        ):
            out.append(f"In-place text exceeds {c['slot']} bytes")
    except (AssertionError, ValueError, KeyError, IndexError) as ex:
        # Treat encoding/layout failures as entry diagnostics so validation can
        # continue through the remaining translations instead of stopping here.
        out.append(str(ex) or type(ex).__name__)
    return out
