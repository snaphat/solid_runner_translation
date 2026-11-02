"""
This module finds the next maze-menu choice toward a task's configured goals.

It reads the controller's native maze position and heading, uses the ROM connectivity table to
find a route, and returns a direction relative to the current heading. Progress is stored in
maze_goal and maze_steps on the controller; the configured goal list is not modified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verification.game.control_gameplay import GameplayController
import collections


def choose_option(game: GameplayController) -> int | None:
    """
    Choose the native maze-menu option toward the next goal.

    Goals already at the current position advance maze_goal before the search
    budget and heading checks. maze_steps counts searches attempted, including
    a search that fails to find a route. These progress updates are not undone
    when a later check or search raises. The configured goals are not changed.

    Args:
        game (GameplayController): Controller supplying native maze position
            and the goals in entry_setup["maze_goals"]. Progress is recorded
            by updating maze_goal and maze_steps on this controller.

    Returns:
        int | None: Zero-based menu option leading toward the next maze
            goal, or None when all goals are satisfied or maze_steps is at least
            100. A disconnected remaining goal raises instead of returning None.

    Raises:
        ValueError: If maze progress is negative or the current maze position or heading
            is invalid.
        RuntimeError: If the next goal is disconnected in the original connectivity
            table.
        AssertionError: If the controller has no active replay instructions.
    """
    if game.entry_setup is None:
        raise AssertionError("Operation requires an active replay task")

    goals = game.entry_setup["maze_goals"]
    if game.maze_goal < 0 or game.maze_steps < 0:
        raise ValueError("Maze progress must be nonnegative")

    address, heading = game.W[0x1814], game.W[0xBD]
    # Completed (and empty) routes have no next goal to index. Repeated goals
    # at the current position are already satisfied too.
    while game.maze_goal < len(goals) and address == goals[game.maze_goal]:
        game.maze_goal += 1

    if game.maze_goal >= len(goals) or game.maze_steps >= 100:
        return None

    if not (0xA0 <= address <= 0xF5 and address % 16 < 6) or heading not in (0, 64, 128, 192):
        raise ValueError("Invalid native maze position or heading")

    game.maze_steps += 1

    # Connectivity bytes describe exits on the native six-column maze grid.
    # Search absolute headings first; translate the first step to a menu choice below.
    table = game.release[0x9EFC4:0x9EFF4]
    queue: collections.deque[tuple[int, list[int]]] = collections.deque([(address, [])])
    seen: set[int] = set()
    route: list[int] | None = None

    while queue:
        at, path = queue.popleft()
        if at == goals[game.maze_goal]:
            route = path
            break

        if at in seen:
            continue

        seen.add(at)

        for direction, bit, delta in ((0, 1, -16), (64, 8, 1), (128, 2, 16), (192, 4, -1)):
            to = at + delta
            if 0xA0 <= to <= 0xF5 and to % 16 < 6 and table[((at >> 4) - 10) * 8 + (at & 15)] & bit:
                queue.append((to, path + [direction]))

    if not route:
        raise RuntimeError("Maze goal is not connected in the original table")

    # The menu choices are relative to the current heading: forward, right/back,
    # or left. Heading values advance by 64 per quarter turn.
    return 0 if route[0] == heading else 1 if (route[0] - heading) % 256 in (64, 128) else 2
