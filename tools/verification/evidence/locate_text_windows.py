"""
This module supplies text origins and ink thresholds for gameplay screenshot checks.

get_capture_window() maps an original-ROM text-entry ID to the expected text
origin and per-channel RGB brightness thresholds. It selects settings for
battle captions, centered captions, map labels, concert lyrics, or the default
dialogue window. Unlisted IDs use the dialogue settings. The lookup returns
coordinates and thresholds; it does not read images or detect text itself.
"""

from __future__ import annotations

from typing import cast

from core.text.get_text_limits import ROM_CAPACITY, is_combat_caption


def get_capture_window(entry: str) -> tuple[int, int, tuple[int, int, int]]:
    """
    Select the screen origin and ink threshold for a captured text entry.

    Dialogue, battle captions, centered narration, map labels and concert lyrics
    occupy different screen regions. Pixel checks use this lookup to place expected
    font pixels into the correct frame coordinates.

    Args:
        entry (str): Exactly six ASCII hexadecimal digits identifying the
            original text entry. Uppercase and lowercase are accepted; prefixes,
            signs, whitespace, and other lengths are rejected.

    Returns:
        tuple[int, int, tuple[int, int, int]]: Text-plane origin (x, y) in captured-frame
            pixels, followed by the RGB brightness thresholds used to recognize ink.

    Raises:
        TypeError: entry is not a string.
        ValueError: entry is not exactly six ASCII hexadecimal digits.
    """
    if not isinstance(cast(object, entry), str):
        raise TypeError("Text-entry ID must be a string")
    if len(entry) != 6 or any(char not in "0123456789abcdefABCDEF" for char in entry):
        raise ValueError("Text-entry ID must contain exactly six ASCII hexadecimal digits")

    # These are framebuffer pixel origins and per-channel brightness cutoffs,
    # not text-storage sizes. Different overlays use different colors and positions.
    n = int(entry, 16)
    if n < ROM_CAPACITY and is_combat_caption(n):
        return 8, 200, (150, 150, 120)  # battle caption strip

    if n in (0x169A7E, 0x16E1C5, 0x0B97F6):
        return 8, 105, (100, 100, 80)  # centred captions

    if 0x0F3978 <= n <= 0x0F3C21:
        return 8, 201, (100, 100, 80)  # map labels

    if 0x0F4EB7 <= n <= 0x0F4FA5:
        return 8, 192, (220, 220, 180)  # concert lyric overlay

    return 8, 168, (220, 220, 180)  # dialogue window
