"""
This module defines joypad button IDs and validates held-button IDs and frame counts before emulator playback.

BTN maps supported button names to libretro IDs. collect_button_ids() validates and deduplicates
supplied IDs; validate_frame_count() rejects negative durations and noninteger values. These
helpers require no loaded emulator, so task preflight and runtime playback can share the same
rules without loading the native core.
"""

from typing import Iterable

from core.validation.validate_integers import is_integer

# Libretro joypad IDs consumed by the input-state callback, not SNES register masks.
BTN = {
    "b": 0,
    "y": 1,
    "select": 2,
    "start": 3,
    "up": 4,
    "down": 5,
    "left": 6,
    "right": 7,
    "a": 8,
    "x": 9,
    "l": 10,
    "r": 11,
}

# Reused for per-frame input validation without rebuilding the ID set.
BUTTON_IDS = frozenset(BTN.values())


def validate_frame_count(frames: int) -> None:
    """
    Require a nonnegative integer duration before changing input or core state.

    Args:
        frames (int): Nonnegative duration in frames; zero and integer subclasses
            are accepted. Booleans and nonintegers are rejected.

    Raises:
        ValueError: If the duration is not a nonnegative integer.
    """
    if (not is_integer(frames)) or frames < 0:
        raise ValueError("Frame count must be a nonnegative integer")


def collect_button_ids(held: Iterable[int]) -> set[int]:
    """
    Materialize held input once and reject IDs outside the supported joypad.

    The iterable is consumed once, and every item is validated before deduplication. Exceptions
    raised by the input iterator propagate to the caller.

    Args:
        held (Iterable[int]): Integer libretro button IDs from BTN, possibly a generator.
            Integer subclasses are accepted; booleans and nonintegers are rejected.

    Returns:
        set[int]: Independent set that can be reused for every requested frame.

    Raises:
        ValueError: If an ID does not identify a supported SNES button.
        TypeError: If held is not iterable.
    """
    # Check before deduplication: 1, True and 1.0 compare equal in a set,
    # which could hide an invalid ID when a valid one appears first.
    buttons = list(held)
    if any((not is_integer(button)) or button not in BUTTON_IDS for button in buttons):
        raise ValueError("Unsupported joypad button ID")
    return set(buttons)
