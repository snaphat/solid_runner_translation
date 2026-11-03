"""
This module changes player or enemy health at battle boundaries and applies declared defeat fixtures during verification.

The helpers inspect the game's battle state before refilling living player HP, setting
configured opponent HP, or forcing a declared defeat. Per-opponent and turn-boundary tracking
prevents repeated application. These fixtures accelerate verification while leaving the game's
text and transition routines in control.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Iterable, Protocol, Sequence

from core.validation.validate_integers import is_integer

if TYPE_CHECKING:
    from verification.game.control_gameplay import GameplayController


class BattleMemory(Protocol):
    """
    Native memory and bounded writes required by the player-health fixture.
    """

    @property
    def W(self) -> ctypes.Array[ctypes.c_ubyte] | Sequence[int]:
        """
        Return the current WRAM byte view used to inspect battle state.

        Returns:
            ctypes.Array[ctypes.c_ubyte] | Sequence[int]: Readable region-relative
                WRAM bytes; a native view is invalid after unloading the ROM.
        """
        ...

    def poke(self, address: int, data: bytes | Iterable[int], region: int = 2) -> None:
        """
        Write a bounded fixture payload into a native memory region.

        Args:
            address (int): Region-relative destination byte offset.
            data (bytes | Iterable[int]): Replacement byte values.
            region (int, optional): Libretro memory ID; 2 selects WRAM.
                Defaults to 2.
        """
        ...


def refill_player_health(game: BattleMemory) -> None:
    """
    Refill a living player at an idle native battle boundary.

    Args:
        game (BattleMemory): Battle controller whose player HP in WRAM
            may be restored to its native maximum.
    """
    w = game.W

    # Only the idle battle handler may receive a refill. Bit 5 of $16BC marks
    # player defeat, which this fixture must never undo.
    if w[0x1835] != 0xCA or w[0x1836] != 0xD1 or w[0x16BC] & 0x20:
        return

    hp = w[0x189A] | w[0x189B] << 8
    maximum = w[0x1896] | w[0x1897] << 8
    if 0 < hp < maximum:
        game.poke(0x189A, maximum.to_bytes(2, "little"))


def lower_enemy_health(game: GameplayController) -> None:
    """
    Set the current opponent's configured health after initialization.

    The health override applies only while the opponent is alive and its pair
    of WRAM $182C and native maximum HP is absent from health_fixed. The pair
    is added after the write. It remains marked until the caller clears it,
    including if a later battle uses the same pair. The configured value is
    bounded by native maximum HP, not by the current remaining HP.

    Args:
        game (GameplayController): Battle controller supplying
            get_opponent_health_fixture(); opponent HP and the health_fixed tracking set
            are updated. The configured health must be an integer; integer
            subclasses are accepted and booleans are rejected.

    Raises:
        ValueError: If the requested health is a boolean, is not an integer,
            or is outside 1 through the native opponent maximum.
        AssertionError: If no opponent health value is configured when an eligible
            initialized opponent is encountered.
    """
    w = game.W
    if w[0x183B] != 0x40:
        return

    current = w[0x189C] | w[0x189D] << 8
    maximum = w[0x1898] | w[0x1899] << 8

    # The set records WRAM $182C and maximum-HP pairs already overridden.
    # A matching pair is skipped until the caller removes it.
    token = (w[0x182C], maximum)
    if current and token not in game.health_fixed:
        value = game.get_opponent_health_fixture()
        if value is None:
            raise AssertionError("No opponent health fixture was configured")

        if not is_integer(value) or not 1 <= value <= maximum:
            raise ValueError("Enemy health fixture outside the original maximum")

        game.poke(0x189C, value.to_bytes(2, "little"))
        game.health_fixed.add(token)


def finish_battle(game: GameplayController, after_turns: int = 0) -> None:
    """
    Apply the one-time defeat fixture at an eligible turn boundary.

    Turn counts and the defeated set use the pair of WRAM $182C and native
    opponent maximum HP. This function does not reset that tracking when
    another battle uses the same pair; the caller manages its lifetime.

    Args:
        game (GameplayController): Battle controller whose turn counters,
            defeated set, HP and defeat flags are updated; task.lose_fixture
            selects player defeat.
        after_turns (int, optional): Number of native turn-end boundaries to
            leave untouched. The fixture applies at the next eligible boundary;
            zero allows the first one. Nonnegative integers and their subclasses
            are accepted; booleans are rejected.
            Defaults to 0.

    Raises:
        ValueError: If after_turns is a boolean, is not an integer, or is negative.
    """
    # Validate before changing turn counters, flags or health.
    if not is_integer(after_turns) or after_turns < 0:
        raise ValueError("Turn budget must be a nonnegative integer, not a boolean")
    w = game.W
    at = (
        w[0x179C] == 120
        and w[0x1835] == 0x6E
        and w[0x1836] == 0xD3
        and not (w[0x16BC] & 0x20 or w[0x16BD] & 0x20)
    )
    token = (w[0x182C], w[0x1898] | w[0x1899] << 8)

    # Count entry into the turn-end handler, not each frame spent waiting in it.
    if at and not game.at_turn_end:
        game.turn_ends[token] = game.turn_ends.get(token, 0) + 1

    game.at_turn_end = at
    if (
        at
        and game.turn_ends.get(token, 0) > after_turns
        and w[0x189C] | w[0x189D] << 8
        and token not in game.defeated
    ):
        if (game.task or {}).get("lose_fixture"):
            # Player-defeat fixtures clear player HP and set $16BC bit 5.
            game.poke(0x189A, [0, 0])
            game.poke(0x16BC, [w[0x16BC] | 0x20])
        else:
            game.poke(0x189C, [0, 0])
            game.poke(0x16BD, [w[0x16BD] | 0x20])

        game.defeated.add(token)
