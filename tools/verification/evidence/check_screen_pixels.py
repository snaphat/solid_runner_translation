"""
This module checks interface text pixels at positions recorded from the game's display registers.

Glyph traces reconstruct native text-buffer ink. Captured background settings select tilemaps
and exact scroll offsets for each visible scanline band; matching never searches the screenshot
for a convenient alignment. Tile flips, eight/sixteen-pixel cells and 32/64 cell tilemap
dimensions are applied before checking each visible copy against VRAM and RGB pixels.
"""

from __future__ import annotations

import collections
from typing import Any, Iterable, Sequence

from verification.evidence.instrument_rom import GlyphRecord
from verification.evidence.read_text_layers import TextLayer
from verification.evidence.reconstruct_text_ink import reconstruct_glyph_ink


def reconstruct_screen_ink(
    rom: bytes | bytearray, glyphs: Iterable[GlyphRecord]
) -> set[tuple[int, int]]:
    """
    Reconstruct glyph ink in the native text buffer before tilemap placement.

    Interface menus reuse parts of that buffer at different screen positions. The returned
    coordinates therefore retain values beyond x=240, where multi-column text continues into
    another buffer strip, for the tilemap search to resolve.

    Args:
        rom (bytes | bytearray): Font data used to reconstruct text-buffer ink.
        glyphs (Iterable[GlyphRecord]): Instrumented glyph records containing font offsets and
            text-buffer drawing coordinates.

    Returns:
        set[tuple[int, int]]: Expected (x, y) ink coordinates in the game text buffer, before
            tilemap placement or clipping.

    Raises:
        ValueError: If a glyph selects a negative or incomplete ROM font record.
    """
    return reconstruct_screen_mask(rom, glyphs)[0]


def reconstruct_screen_mask(
    rom: bytes | bytearray, glyphs: Iterable[GlyphRecord]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """
    This function reconstructs text ink, cleared spans, and right-hand shadow pixels in the
    game's text buffer.

    Draws are processed in order. Each glyph clears its advance span before adding ink. This
    models overwrite mode; the glyph trace does not record the renderer's overwrite flag. Only
    those cleared spans are checked for unwanted text. Adjacent labels and the scene visible
    through transparent tiles are outside this comparison.

    Args:
        rom (bytes | bytearray): Source font tables used to reconstruct the glyphs.
        glyphs (Iterable[GlyphRecord]): Ordered glyph draws for a single text record.

    Returns:
        tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]: Final ink,
            cleared spans clipped to the native buffer, and shadow-only pixels.

    Raises:
        ValueError: If a glyph selects a negative or incomplete ROM font record.
    """
    ink: set[tuple[int, int]] = set()
    cleared: set[tuple[int, int]] = set()
    shadow: set[tuple[int, int]] = set()
    for glyph in glyphs:
        pixels, (left, top, right, bottom) = reconstruct_glyph_ink(rom, glyph)
        span = {
            (x, y)
            for y in range(max(0, top), min(128, bottom))
            for x in range(max(0, left), min(720, right))
        }
        ink.difference_update(span)
        shadow.difference_update(span)
        ink.update(pixels)
        shadow.update((x + 1, y) for x, y in pixels if (x + 1, y) in span)
        cleared.update(span)
    shadow.difference_update(ink)
    return ink, cleared, shadow


def collect_word_ink(
    rom: bytes | bytearray, glyphs: Iterable[GlyphRecord], label_starts: Sequence[int] = ()
) -> list[set[tuple[int, int]]]:
    """
    This function groups adjacent glyphs so each word or menu label can be checked at a single screen position.

    Spaces, line changes and explicit cursor moves separate words or menu labels. Later glyphs
    remove overwritten pixels from earlier groups, matching the final buffer rather than
    requiring an intermediate draw to stay visible.

    Args:
        rom (bytes | bytearray): Source font tables used to identify ink and spaces.
        glyphs (Iterable[GlyphRecord]): Ordered draws for the text record.
        label_starts (Sequence[int], optional): Native buffer X positions where
            declared menu labels begin. These separate adjacent labels even when
            the compiler needs no space or cursor command between them.
            Defaults to ().

    Returns:
        list[set[tuple[int, int]]]: Nonempty final word-ink sets in buffer coordinates.

    Raises:
        ValueError: If a menu-label start is not an integer from zero through 719, or a glyph
            selects an incomplete or negative ROM font record.
    """
    if any(type(start) is not int or not 0 <= start < 720 for start in label_starts):
        raise ValueError("Menu label starts must be integer buffer positions from zero through 719")
    words: list[set[tuple[int, int]]] = []
    active: int | None = None
    previous: tuple[int, int] | None = None
    for glyph in glyphs:
        pixels, (left, top, right, bottom) = reconstruct_glyph_ink(rom, glyph)
        for word in words:
            word.difference_update(
                {p for p in word if left <= p[0] < right and top <= p[1] < bottom}
            )
        if not pixels:
            active = None
            previous = None
            continue
        positioned = bool(glyph["after_col"] & 0x80) and not glyph["col"] & 0x80
        if active is None or positioned or left in label_starts or previous != (left, top):
            words.append(set())
            active = len(words) - 1
        words[active].update(pixels)
        previous = (right, top)
    return [word for word in words if word]


def find_strip_character(line: int, strip: int, tilerow: int) -> int:
    """
    Find the first tile character of a text-buffer strip.

    Args:
        line (int): Logical text-buffer line. The eight base lines are 0-7;
            horizontal strip carry in map_buffer_pixel() can select lines 8 or 9.
        strip (int): Left 120-pixel strip (0) or right strip (1).
        tilerow (int): Upper 8-pixel tile row (0) or lower row (1) within
            the text line.

    Returns:
        int: Character number at tile zero of the selected buffer strip.
    """
    return line * 64 + strip * 32 + tilerow * 16


def map_buffer_pixel(x: int, y: int) -> tuple[int, int, int]:
    """
    Convert a native text-buffer pixel to its character tile and local pixel.

    Args:
        x (int): Horizontal buffer coordinate, zero through 719 across six strips.
        y (int): Vertical buffer coordinate, zero through 127 across eight lines.

    Returns:
        tuple[int, int, int]: Character number, pixel column and pixel row in its 8x8 tile.

    Raises:
        ValueError: If a coordinate is not an integer, is a boolean or is outside the buffer.
    """
    if type(x) is not int or type(y) is not int or not (0 <= x < 720 and 0 <= y < 128):
        raise ValueError("Pixel must be within the native text buffer")
    line, ly = divmod(y, 16)
    strip, sx = divmod(x, 120)
    line, strip = line + strip // 2, strip % 2
    tile_row, py = divmod(ly, 8)
    tx, px = divmod(sx, 8)
    return find_strip_character(line, strip, tile_row) + tx, px, py


# Native text-buffer tile numbers: eight lines, two 120-pixel strips per line,
# two tile rows per strip, and fifteen 8-pixel tiles across each strip.
CHARS = {
    find_strip_character(line, strip, tile_row) + tile_x
    for line in range(8)
    for strip in range(2)
    for tile_row in range(2)
    for tile_x in range(15)
}


def index_tile_characters(
    vram: bytes | bytearray, base: int, tile_size: int = 8
) -> dict[int, list[tuple[int, int, bool, bool, int]]]:
    """
    Index a tilemap's text characters and their display attributes.

    Args:
        vram (bytes | bytearray): Snapshot of libretro video RAM, indexed by byte offset.
        base (int): Nonnegative, 2 KiB-aligned integer byte offset of the
            VRAM tilemap. Booleans are rejected.
        tile_size (int, optional): Native cell width/height: eight or sixteen pixels.
            Sixteen-pixel cells are expanded into their four eight-pixel font tiles.
            Defaults to 8.

    Returns:
        dict[int, list[tuple[int, int, bool, bool, int]]]: Character number mapped to
            (tile X, tile Y, horizontal flip, vertical flip, attribute nibble)
            placements. The nibble is bits 10-13 of the tilemap word: three palette
            bits followed by the priority bit.

    Raises:
        ValueError: If the base is not an integer (excluding booleans), is negative or
            misaligned, or lacks a complete tilemap; or tile_size is not an integer equal to
            eight or sixteen.
    """
    if type(base) is not int or base < 0 or base % 0x800 or base + 0x800 > len(vram):
        raise ValueError("Tilemap must be a complete 2KB-aligned VRAM region")
    if type(tile_size) is not int or tile_size not in (8, 16):
        raise ValueError("Background tile size must be eight or sixteen pixels")
    cells: dict[int, list[tuple[int, int, bool, bool, int]]] = collections.defaultdict(list)
    scale = tile_size // 8
    for ty in range(32):
        for tx in range(32):
            # Each 16-bit tilemap word stores character ID in bits 0-9, flip flags in
            # bits 14-15, and an attribute nibble in bits 10-13. Retain every text placement.
            e = vram[base + 2 * (ty * 32 + tx)] | vram[base + 2 * (ty * 32 + tx) + 1] << 8
            hf, vf = bool(e & 0x4000), bool(e & 0x8000)
            for qy in range(scale):
                for qx in range(scale):
                    char = ((e & 0x3FF) + qx + qy * 16) & 0x3FF
                    if char in CHARS:
                        cells[char].append(
                            (
                                tx * scale + (scale - 1 - qx if hf else qx),
                                ty * scale + (scale - 1 - qy if vf else qy),
                                hf,
                                vf,
                                (e >> 10) & 0xF,
                            )
                        )

    return cells


def _validate_rgb_frame(rows: Sequence[bytes | bytearray]) -> None:
    """
    Reject incomplete or unsupported native-resolution RGB screenshots.

    Args:
        rows (Sequence[bytes | bytearray]): A 256-pixel-wide RGB frame with
            224 or 239 visible scanlines; every pixel must have three channels.

    Raises:
        ValueError: If the frame has neither 224 nor 239 scanlines, or a scanline does not
            contain 768 RGB bytes.
    """
    if len(rows) not in (224, 239) or any(len(row) != 256 * 3 for row in rows):
        raise ValueError("RGB frame must contain 224 or 239 complete 256-pixel scanlines")


def collect_bright_pixels(rows: Sequence[bytes]) -> set[tuple[int, int]]:
    """
    Locate bright pixels for labels whose native animation finishes in white.

    This helper does not validate dimmed interface text. The mission reward animation uses it
    after counting and palette fading have finished.

    Args:
        rows (Sequence[bytes]): Complete 256-pixel RGB scanlines; frame height
            is 224 or 239. Only the first 224 rows are inspected.

    Returns:
        set[tuple[int, int]]: Warm-white pixel coordinates in the first 224 rows.

    Raises:
        ValueError: If the RGB frame is incomplete or has unsupported dimensions.
    """
    _validate_rgb_frame(rows)
    return {
        (x, y)
        for y, row in enumerate(rows[:224])
        for x in range(256)
        if row[x * 3] > 220 and row[x * 3 + 1] > 220 and row[x * 3 + 2] > 180
    }


def check_screen_pixels(
    rom: bytes | bytearray,
    glyphs: Iterable[GlyphRecord],
    rows: Sequence[bytes | bytearray],
    vram: bytes | bytearray,
    *,
    layers: Sequence[TextLayer],
    box: tuple[int, int, int, int] | None = None,
    label_starts: Sequence[int] = (),
) -> dict[str, Any]:
    """
    Compare traced interface ink at the frame's recorded background positions.

    No tilemap or scroll is inferred from bright screenshot pixels. Each placement must belong
    to an enabled text background in the captured scanline band. Every completely mapped word
    copy must match; copies cannot supply missing pixels for one another. Cropped tile fragments
    provide no coverage. Cleared glyph spans must contain no extra ink in VRAM, and opaque
    shadow pixels must retain contrast within each complete word copy. Each word uses its own
    uniform foreground color; exact palette reproduction is not asserted. Opaque shadows must
    remain darker, while transparent cleared pixels may show any scene color. This is not a
    complete PPU compositor comparison. Literal translation matching remains the caller's
    separate responsibility.

    Args:
        rom (bytes | bytearray): Release containing the traced glyphs' font data.
        glyphs (Iterable[GlyphRecord]): Glyph draws belonging to the text record.
        rows (Sequence[bytes | bytearray]): Complete 256-pixel RGB scanlines;
            frame height must be 224 or 239.
        vram (bytes | bytearray): Complete 65536-byte VRAM snapshot for the same frame.
        label_starts (Sequence[int], optional): Native buffer X positions of
            declared menu-label starts, in pixels. Adjacent labels remain separate
            placement groups even when their rendered glyph advances touch.
            Defaults to ().
        box (tuple[int, int, int, int] | None, optional): Required output rectangle
            (left, top, right, bottom), with exclusive right/bottom. None checks
            every visible hardware-mapped occurrence without requiring the text
            to appear in a specified output rectangle.
        layers (Sequence[TextLayer]): Recorded text backgrounds, scroll offsets,
            tilemap sizes and visible scanline bands. Required; no search fallback.

    Returns:
        dict[str, Any]: passed requires nonempty ink and zero missing, extra,
            clipped or out-of-buffer pixels. Separate counters report screenshot
            ink/shadow errors and VRAM ink errors. placement contains the decoded
            text-layer list, optional box and comparison counts, or None when
            layers is empty. A word without a complete mapped copy makes
            passed False.

    Raises:
        ValueError: If the RGB frame or VRAM snapshot is incomplete, a text box or menu-label
            position is invalid, or font/tilemap data is outside its buffer.
    """
    _validate_rgb_frame(rows)
    if len(vram) != 0x10000:
        raise ValueError("Text verification requires a complete 65536-byte VRAM snapshot")
    draws = list(glyphs)
    ink, cleared, shadow = reconstruct_screen_mask(rom, draws)
    words = collect_word_ink(rom, draws, label_starts)
    points = sorted(ink)
    if box is not None and (
        len(box) != 4
        or any(type(v) is not int for v in box)
        or not (0 <= box[0] < box[2] <= 256 and 0 <= box[1] < box[3] <= len(rows))
    ):
        raise ValueError("Text box must be a nonempty rectangle within the captured frame")
    outside = [(x, y) for x, y in points if x < 0 or x >= 720 or y < 0 or y >= 128]
    points = [p for p in points if not (p[0] < 0 or p[0] >= 720 or p[1] < 0 or p[1] >= 128)]

    # Index only hardware-selected maps. A 64-tile dimension uses an additional
    # 32x32 page, whose tile coordinates are relative to the whole background.
    indexed: list[tuple[TextLayer, dict[int, list[tuple[int, int, bool, bool, int]]]]] = []
    for layer in layers:
        cells: dict[int, list[tuple[int, int, bool, bool, int]]] = collections.defaultdict(list)
        scale = layer["tile_size"] // 8
        across = 2 if layer["size"] & 1 else 1
        down = 2 if layer["size"] & 2 else 1
        for page_y in range(down):
            for page_x in range(across):
                base = (layer["base"] + (page_y * across + page_x) * 0x800) & 0xFFFF
                for char, tile_spots in index_tile_characters(
                    vram, base, layer["tile_size"]
                ).items():
                    cells[char].extend(
                        (tx + page_x * 32 * scale, ty + page_y * 32 * scale, hf, vf, attr)
                        for tx, ty, hf, vf, attr in tile_spots
                    )
        indexed.append((layer, cells))

    # A map can reuse one tile as blank fill, or show a cropped fragment in
    # another window. Only a whole word mapped to one position can pass the check.
    mapped: dict[tuple[int, int], set[tuple[int, int, int, tuple[int, ...]]]] = {}
    outside_box: set[tuple[int, int]] = set()
    word_copies: dict[tuple[int, int], set[tuple[int, ...]]] = collections.defaultdict(set)
    for x, y in sorted(ink | cleared):
        if not (0 <= x < 720 and 0 <= y < 128):
            continue
        char, px, py = map_buffer_pixel(x, y)
        spots: set[tuple[int, int, int, tuple[int, ...]]] = set()
        for layer, cells in indexed:
            width = layer["tile_size"] * (64 if layer["size"] & 1 else 32)
            height = layer["tile_size"] * (64 if layer["size"] & 2 else 32)
            for cx, cy, hf, vf, _ in cells.get(char, []):
                screen_x = (cx * 8 + (7 - px if hf else px) - layer["dx"]) % width
                screen_y = (cy * 8 + (7 - py if vf else py) - layer["dy"]) % height
                if not (screen_x < 256 and layer["top"] <= screen_y < layer["bottom"]):
                    continue
                if box is not None and not (
                    box[0] <= screen_x < box[2] and box[1] <= screen_y < box[3]
                ):
                    outside_box.add((x, y))
                    continue
                sign_x, sign_y = (-1 if hf else 1), (-1 if vf else 1)
                copy = (
                    layer["layer"],
                    layer["font_base"],
                    sign_x,
                    sign_y,
                    screen_x - sign_x * x,
                    screen_y - sign_y * y,
                )
                spots.add((screen_x, screen_y, layer["font_base"], copy))
                word_copies[(x, y)].add(copy)
        mapped[(x, y)] = spots

    accepted: dict[tuple[int, int], set[tuple[int, ...]]] = collections.defaultdict(set)
    complete_copies: set[tuple[int, ...]] = set()
    incomplete = ignored_fragments = 0
    for word in words:
        copies = [word_copies[p] for p in word]
        complete = copies[0].intersection(*copies[1:])
        observed = copies[0].union(*copies[1:])
        ignored_fragments += len(observed - complete)
        if not complete:
            incomplete += max(1, len(observed))
        for point in word:
            accepted[point].update(complete)
        complete_copies.update(complete)
    missing = unmapped = unexpected = vram_missing = vram_extra = clipped = comparisons = 0
    for (x, y), mapped_spots in mapped.items():
        expected = (x, y) in ink
        permitted = accepted[(x, y)] if expected else complete_copies
        selected = [spot for spot in mapped_spots if spot[3] in permitted]
        if not selected:
            if expected:
                missing += 1
                unmapped += 1
                clipped += (x, y) in outside_box and not mapped_spots
            continue
        char, px, py = map_buffer_pixel(x, y)
        for screen_x, screen_y, font_base, copy in selected:
            comparisons += 1
            address = (font_base + char * 16 + py * 2) & 0xFFFF
            actual = bool(vram[address] & (0x80 >> px))
            if expected:
                vram_missing += not actual
            else:
                vram_extra += actual

    # Foreground color is measured separately for each complete word copy.
    # Color math can dim one save slot while leaving another white. A shared
    # brightness threshold would reject valid copies or accept their backgrounds.
    # Exact VRAM masks remain independent of this RGB shape/contrast comparison.
    def rgb(point: tuple[int, int]) -> bytes:
        """
        Read one RGB pixel from the frame being checked.

        Args:
            point (tuple[int, int]): Visible output X/Y coordinates.

        Returns:
            bytes: The pixel's three channel values.
        """
        x, y = point
        return bytes(rows[y][x * 3 : x * 3 + 3])

    def has_shadow(point: tuple[int, int], font_base: int) -> bool:
        """
        Test whether a cleared buffer pixel has opaque shadow in VRAM plane one.

        Args:
            point (tuple[int, int]): Native text-buffer X/Y coordinates.
            font_base (int): Byte address of the mapped two-bit font graphics.

        Returns:
            bool: True when the shadow plane contains this pixel.
        """
        character, px, py = map_buffer_pixel(*point)
        address = (font_base + character * 16 + py * 2 + 1) & 0xFFFF
        return bool(vram[address] & (0x80 >> px))

    for word in words:
        first, *rest = word
        copies = accepted[first].intersection(*(accepted[point] for point in rest))
        guards = {(x + 1, y) for x, y in word} & shadow
        for copy in copies:
            foreground = [
                (sx, sy)
                for point in word
                for sx, sy, _, placement in mapped[point]
                if placement == copy
            ]
            background = {
                (sx, sy)
                for point in guards
                for sx, sy, font_base, placement in mapped.get(point, set())
                if placement == copy and has_shadow(point, font_base)
            }
            colors = collections.Counter(rgb(point) for point in foreground)
            color = colors.most_common(1)[0][0]
            # Allow each RGB channel to vary by one intensity level within a word;
            # shadows must differ by more than this same tolerance.
            missing += sum(
                any(abs(actual - expected) > 1 for actual, expected in zip(rgb(point), color))
                for point in foreground
            )
            # Plane-one shadow pixels are opaque; transparent cleared pixels
            # can expose any scene color and must not be treated as shadows.
            # A single dark spot cannot make an otherwise solid block a word.
            unexpected += sum(
                max(expected - actual for expected, actual in zip(color, rgb(point))) <= 1
                for point in background
            )
            if not background:
                unexpected += 1

    passed = bool(points) and not (
        outside or missing or unexpected or vram_missing or vram_extra or clipped or incomplete
    )
    return dict(
        ink=len(points),
        outside_ink=len(outside),
        missing_screen_ink=missing,
        unexpected_screen_ink=unexpected,
        missing_vram_ink=vram_missing,
        unexpected_vram_ink=vram_extra,
        clipped_ink=clipped,
        incomplete_copies=incomplete,
        passed=passed,
        placement=dict(
            layers=list(layers),
            missing=missing,
            unmapped=unmapped,
            unexpected=unexpected,
            missing_vram=vram_missing,
            unexpected_vram=vram_extra,
            clipped=clipped,
            comparisons=comparisons,
            incomplete_copies=incomplete,
            ignored_fragments=ignored_fragments,
            box=box,
        )
        if layers
        else None,
    )
