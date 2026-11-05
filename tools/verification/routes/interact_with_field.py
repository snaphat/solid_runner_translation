"""
This module sets up rooms and performs actor, tile, and map interactions specified by field-route tasks.

Task definitions supply progression flags, room IDs and an actor, tile or movement target.
Room-setup instructions handle room-specific initialization. Actor interaction waits for native
handlers; city-map traversal uses controller inputs. The resulting outcome distinguishes blocked
initialization and missing actors from completed or idle playback.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Any, cast

from core.validation.validate_integers import is_integer

from verification.game.control_gameplay import DialogueOutcome
from verification.tasks import validate_requirements as requirements

if TYPE_CHECKING:
    from verification.game.control_gameplay import GameplayController

from core.rom.convert_addresses import convert_snes_to_file

from verification.game.control_gameplay import BTN

# Target (stage, variant) -> fallback room to load if initialization leaves no
# actor matching the requested talk script. Both keys and values are native room IDs.
REENTRIES = {(4, 9): (4, 6), (4, 10): (4, 6), (3, 2): (3, 0)}


def _validate_task(task: dict[str, Any]) -> None:
    """
    Validate shared task fields and field-interaction options before playback.

    Args:
        task (dict[str, Any]): Definition read without mutation. Numeric setup fields
            accept integer subclasses and reject booleans. Optional field_approach
            is up, right, down or left; field_trigger.activation is a or walk.
            Talk-handler and actor-root addresses contain six hexadecimal digits.

    Raises:
        ValueError: If the task is not a dictionary or a supported field is invalid.
    """
    if not isinstance(cast(object, task), dict):
        raise ValueError("Field task must be an object")
    requirements.validate_definition(task)
    if "field_approach" in task and task["field_approach"] not in (
        "up",
        "right",
        "down",
        "left",
    ):
        raise ValueError("field_approach must be up, right, down or left")
    if "field_trigger" in task and task["field_trigger"].get("activation", "a") not in (
        "a",
        "walk",
    ):
        raise ValueError("field_trigger activation must be a or walk")
    if task.get("field_position") and "field_trigger" not in task:
        raise ValueError("field_position requires a field_trigger object")
    for name in ("talk", "root", "expected_actor_root"):
        value = task.get(name)
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 6
            or any(character not in "0123456789ABCDEFabcdef" for character in value)
        ):
            raise ValueError(f"{name} must contain six hexadecimal digits")
    if "preparatory_talks" in task:
        values = task["preparatory_talks"]
        if not isinstance(values, list):
            raise ValueError("preparatory_talks must be a list of handler addresses")
        for value in cast(list[object], values):
            if (
                not isinstance(value, str)
                or len(value) != 6
                or any(character not in "0123456789ABCDEFabcdef" for character in value)
            ):
                raise ValueError(
                    "Preparatory talk addresses require six hexadecimal digits"
                )


def _filter_actor_candidates(
    game: GameplayController, task: dict[str, Any], candidates: list[int]
) -> list[int]:
    """
    Retain approached actors that still match the requested actor-root script.

    Approach helpers use the task's root field when locating a remembered actor.
    expected_actor_root can impose a different constraint on the final conversation.
    This check applies that constraint after approach without admitting other slots.

    Args:
        game (GameplayController): Controller whose live actor pointers are read;
            emulator memory and captures are unchanged.
        task (dict[str, Any]): Validated definition containing talk and an optional
            expected_actor_root, a six-digit hexadecimal SNES script address.
            None or an absent value leaves candidate selection unchanged.
        candidates (list[int]): Slots returned by the approach helper, in selection
            order. This list is not modified.

    Returns:
        list[int]: New filtered list when a root is required; otherwise the original
            candidates list. An empty list means no eligible actor was approached.

    Raises:
        ValueError: If the controller cannot read complete actor-pointer tables.
    """
    root = task.get("expected_actor_root")
    if root is None:
        return candidates
    eligible = set(game.find_talk_slots(task["talk"], root))
    return [slot for slot in candidates if slot in eligible]


def derive_entry_setup(task: dict[str, Any]) -> dict[str, Any] | None:
    """
    Derive the room-specific preparation needed before attempting a field interaction.

    Some targets require a native maze path, entry room, return packet or prior conversation to
    initialize their actors. The returned setup dictionary describes those requirements; this
    function itself does not move the player or write memory.

    Args:
        task (dict[str, Any]): Route key, stage and variant with optional
            native_field_prelude, native_maze_goals and quest_flags; inspected
            without mutation.

    Returns:
        dict[str, Any] | None: Room-specific preparation instructions, or None when no
            special setup is required. Explicit prelude and maze-goal lists are
            referenced from task rather than copied; callers must not mutate them.

    Raises:
        ValueError: If a supported task field is invalid, the key is not a string,
            a maze goal is outside the native grid, or a cafe chapter is outside 0-3.
        KeyError: If stage or variant is missing.
    """
    _validate_task(task)
    if not isinstance(task.get("key"), str):
        raise ValueError("Room setup requires a string task key")
    room = (task["stage"], task["variant"])

    # A declared native prelude takes precedence over inferred room setup;
    # it represents an actual arrival sequence needed by the interaction.
    if task.get("native_field_prelude"):
        return dict(prelude=task["native_field_prelude"])

    # The cafe reply depends on chapter and alternate-route flags. Enter via
    # the adjacent room and require the branch-specific reply from the native interaction.
    if task["key"].startswith("context-coffee-"):
        chapter = next(
            (f["value"] for f in task.get("quest_flags", []) if f["address"] == 0x5B), 0
        )
        alternate = any(
            f["address"] == 0x15 and f["value"] & 0x10
            for f in task.get("quest_flags", [])
        )
        if not 0 <= chapter <= 3:
            raise ValueError("Cafe setup requires a chapter flag from 0 through 3")
        required = (
            "0AA100"
            if chapter == 0 and alternate
            else ("0AA0CC", "0AA100", "0AA123", "0AA166")[chapter]
        )
        return dict(
            entry_room=[3, 0],
            movement_route=["up"] * 8,
            required_interaction_ids=[required],
        )

    if room == (5, 9):
        return dict(save_return_packet=True, reentry=[5, 9])

    if task["key"] == "npc-branches-0127":
        return dict(
            entry_room=[12, 0],
            movement_route=["down"] * 5 + ["left"] * 2 + ["down"] * 5,
            required_interaction_ids=["0B8FB0"],
        )

    # An unlocked report changes the arrival sequence for these rooms.
    # Replay its entrance prelude so automatic text runs before the requested actor.
    report = (
        room[0] == 14
        and room[1] in (1, 2, 5, 6, 7, 11)
        and any(
            f["address"] == 0x1D7 and f["value"] & 0x80
            for f in task.get("quest_flags", [])
        )
    )
    if report:
        return dict(prelude=[14, 0], automatic_event_ids=["0BCC75", "0BCCBD"])

    if task["key"] == "auto-0807":
        return dict(
            save_return_packet=True,
            entry_room=[21, 8],
            required_event_ids=["09E92A", "09E944", "09E996"],
            required_leave_room=[22, 2],
            choice_plans={"09E4A5": [0, 1, 2], "09E7A5": [2], "09E8A9": [2]},
        )

    if task.get("native_maze_goals"):
        goals = task["native_maze_goals"]
        if not all(0xA0 <= g <= 0xF5 and g % 16 < 6 for g in goals):
            raise ValueError("Maze goal outside the original address grid")

        return dict(save_return_packet=True, maze_goals=goals, frames=48000)

    if task["key"] == "auto-0806":
        # Collect all four signals and return to E1 through the original maze menu.
        return dict(
            save_return_packet=True,
            maze_goals=[0xD5, 0xA1, 0xC0, 0xF3, 0xE1],
            frames=48000,
        )

    if room in ((1, 10), (2, 12), *REENTRIES):
        return dict(reentry=list(REENTRIES[room]) if room in REENTRIES else None)

    return None


def apply_input_sequence(game: GameplayController, task: dict[str, Any]) -> str:
    """
    Submit a task's controller inputs and declared fixture writes.

    Args:
        game (GameplayController): Controller receiving each input/write;
            assigned task and resulting captures remain on the controller.
        task (dict[str, Any]): Route containing ordered inputs; each step specifies a
            controller action or declared memory write. Button steps require frames
            and buttons (names from BTN). Write steps require poke (region-relative
            offset), data (hexadecimal bytes), and optional region (default WRAM 2).

    Returns:
        str: idle after every step was submitted. This does not test native idle
            state, capture text, or establish that requested targets were reached.

    Raises:
        ValueError: If a task field, button, frame count, hexadecimal payload or
            memory-write span is invalid. The whole sequence is checked before
            game.task is assigned; runtime writes also check the actual region size.
        KeyError: If inputs is missing.
        PlaybackStopped: If frame advancement reaches a configured stop condition.
    """
    _validate_task(task)
    steps = task["inputs"]
    game.task = task

    # Fixture writes and held-button intervals share one ordered sequence.
    # A write takes effect before the following frame interval is emulated.
    for step in steps:
        if "poke" in step:
            game.poke(step["poke"], bytes.fromhex(step["data"]), step.get("region", 2))
        else:
            game.run(step["frames"], {BTN[b] for b in step["buttons"]})

    return "idle"


def visit_locations(game: GameplayController, task: dict[str, Any]) -> str:
    """
    Visit city-map areas and check every available location caption.

    The helper uses the native city-map connectivity table at $95:BBCE. It opens
    area lists without confirming a location, so the player never enters one.
    Only captures made during this call satisfy the requested targets. Existing
    captures are retained. Emulator state is not restored after traversal or failure.

    Args:
        game (GameplayController): Controller on the city map; buttons traverse
            areas and location captions accumulate in pages.
        task (dict[str, Any]): Map route whose targets list specifies the location
            entries that must be observed.

    Returns:
        str: idle if every target was captured during this call, targets_incomplete
            if only some were captured, or target_not_reached if none were captured.
            Capturing an entry does not by itself establish that its pixels passed.

    Raises:
        ValueError: If a task field or the 13-area connectivity table is invalid.
            Page verification can also reject invalid text chains, font offsets,
            glyph positions, frame/VRAM dimensions or source metadata.
        KeyError: If targets is missing or verified text contains an unsupported command.
        RuntimeError: If an area is disconnected or a required frame snapshot is
            missing or failed.
        PlaybackStopped: If frame advancement reaches a configured stop condition.
        IndexError: If text verification reads beyond the ROM.
        AssertionError: If page verification requires a frame before one is captured.
        OSError: If verification sources cannot be read or screenshots cannot be saved.
    """
    _validate_task(task)
    targets = task["targets"]
    # The native city-map table stores four neighboring area indices per
    # area, in Up/Right/Down/Left order; route finding uses those edges.
    table = game.release[
        convert_snes_to_file(0x95BBCE) : convert_snes_to_file(0x95BBCE) + 52
    ]
    if len(table) != 52 or any(area >= 13 for area in table):
        raise ValueError(
            "City-map table must contain four area indices 0-12 for each of 13 areas"
        )
    before = len(game.pages)

    def find_map_path(start: int, goal: int) -> list[str]:
        """
        Find a shortest sequence of directional map inputs.

        Args:
            start (int): Starting area index in the original four-neighbor map
                table, 0 through 12, supplied by the traversal loop.
            goal (int): Destination area index, 0 through 12, in the same table.

        Returns:
            list[str]: Button names leading from start to goal; empty when they
                match.

        Raises:
            RuntimeError: If no path connects the two areas.
        """
        queue: collections.deque[tuple[int, list[str]]] = collections.deque(
            [(start, [])]
        )
        seen: set[int] = set()

        # Breadth-first traversal finds the fewest map inputs. Mark expanded areas
        # to avoid following the native adjacency table around a cycle indefinitely.
        while queue:
            area, steps = queue.popleft()
            if area == goal:
                return steps

            if area not in seen:
                seen.add(area)
                queue.extend(
                    (table[area * 4 + i], steps + [b])
                    for i, b in enumerate(("up", "right", "down", "left"))
                )

        raise RuntimeError("Area not connected on the original city map")

    def press(button: str) -> None:
        """
        Press one map button and capture a resulting location label.

        Args:
            button (str): Named button accepted by game.press(); held for four
                frames before waiting up to 240 frames for a completed page,
                followed by 120 settling frames. Captures receive the city map phase.

        Raises:
            PlaybackStopped: If frame advancement reaches a configured stop condition.
            ValueError: If page verification encounters invalid text or display data.
            KeyError: If the button or a verified text command is unsupported.
            IndexError: If verified text extends beyond the ROM.
            RuntimeError: If a required frame snapshot is unavailable or failed.
            AssertionError: If page verification requires a frame before one is captured.
            OSError: If sources cannot be read or a screenshot cannot be saved.
        """
        game.clear_log()
        ends = game.T[0x30]
        game.press(button, 4, 0)

        for _ in range(240):
            game.run(1)

            # Wait for a new completed text draw before capturing its label; a stale
            # terminator from the previous selection must not satisfy this map input.
            if game.T[0x30] != ends and game.is_page_finished():
                game.check_page()["phase"] = "city map"
                break

        game.run(120)

    game.run(300)  # the map screen settles; nothing is pressed
    press("b")
    area = 0
    for goal in range(13):
        for button in find_map_path(area, goal):
            press(button)

        area = goal
        press("a")

        for _ in range(3):  # at most three locations per area
            press("up")

        press("b")

    # Distinguish partial coverage from no target at all, so a route that
    # visits some areas cannot pass by merely returning to the idle map.
    seen = {p.get("entry") for p in game.pages[before:]}
    missing = set(targets) - seen
    return (
        "idle"
        if not missing
        else "targets_incomplete"
        if set(targets) & seen
        else "target_not_reached"
    )


def interact_with_actor(game: GameplayController, task: dict[str, Any]) -> str:
    """
    Reach a configured native actor or automatic event.

    Args:
        game (GameplayController): Controller supplying the entry setup and
            actor slots; progression flags, player position and captures may
            change.
        task (dict[str, Any]): Actor route with talk handler, optional actor index,
            choice_plan and frame_budget; automatic routes only complete
            initialization. talk is a six-digit hexadecimal SNES handler address.
            Optional expected_actor_root additionally requires the actor's root
            script pointer to equal that six-digit hexadecimal address. This applies
            after movement, room reentry and approach as well as initial selection.
            actor is preferred only among matching slots; otherwise the first
            matching slot is selected. The definition is not modified.

    Returns:
        str: Initialization or conversation playback outcome, or
            blocked_movement_event, blocked_active_initialization, or
            actor_unavailable when setup prevents the requested interaction.

    Raises:
        ValueError: If a supported task field, actor address, menu plan or memory
            write is invalid, or page verification rejects text or display data.
        KeyError: If route is missing or a text command is unsupported.
        AssertionError: If required controller setup or a frame is unavailable.
        PlaybackStopped: If playback reaches a configured stop condition.
        RuntimeError: If a required frame snapshot is unavailable or failed, or a
            requested maze destination is disconnected.
        IndexError: If text verification reads beyond the ROM.
        OSError: If verification sources cannot be read or screenshots cannot be saved.
    """
    _validate_task(task)
    route = task["route"]
    automatic = route == "automatic" or bool(
        (game.entry_setup or {}).get("automatic_event_ids")
    )
    if not automatic and not isinstance(task.get("talk"), str):
        raise ValueError("Actor interaction requires a talk-handler address")
    prep = game.play_dialogue(
        "initialization",
        (game.entry_setup or {}).get("frames", task.get("frame_budget", 12000)),
    )
    if automatic:
        return prep["status"]

    # Locate the actor by its registered talk handler after initialization,
    # rather than assuming that a particular actor slot still owns it.
    candidates = game.find_talk_slots(task["talk"], task.get("expected_actor_root"))
    if (game.entry_setup or {}).get("movement_route") and prep["status"] == "idle":
        if game.entry_setup is None:
            raise AssertionError("Operation requires an active replay task")

        for button in game.entry_setup["movement_route"]:
            if candidates:
                break

            game.press(button, 40, 10)

            if game.play_dialogue("movement", 12000, 32)["status"] != "idle":
                return "blocked_movement_event"

            candidates = game.find_talk_slots(
                task["talk"], task.get("expected_actor_root")
            )

    if (
        game.entry_setup
        and not candidates
        and game.entry_setup.get("reentry")
        and prep["status"] == "idle"
    ):
        game.set_flags(task.get("quest_flags", []))
        game.load_room(game.entry_setup["reentry"])
        prep = game.play_dialogue("reentry", 12000)
        candidates = game.find_talk_slots(task["talk"], task.get("expected_actor_root"))

    if prep["status"] != "idle" or game.is_busy():
        return "blocked_active_initialization"

    if not candidates:
        candidates = _filter_actor_candidates(
            game, task, game.approach_sighted(task["talk"])
        )

    # A transient actor may disappear during initialization. Retry from the
    # room start and catch its registered handler before automatic dialogue begins.
    if not candidates and game.sighted:
        game.pages = []
        candidates = game.catch_early(task)

    actor = (
        task["actor"]
        if task.get("actor") in candidates
        else candidates[0]
        if candidates
        else None
    )
    if actor is None:
        return "actor_unavailable"

    game.clear_log()
    game.talk(actor)
    return game.play_dialogue("interaction", 6000, 120, task.get("choice_plan"))[
        "status"
    ]


def play_scene(
    game: GameplayController, task: dict[str, Any], early: bool = False
) -> str:
    """
    Execute a prepared scene's movement, tile interaction, and/or conversation.

    Call after game.begin(task). Initialization and position/tile playback use frame_budget
    (default 24000) per phase; the final conversation uses its own 24000-frame budget. These are
    not one cumulative task limit.

    position is a pixel pair adjusted by position_offset (default [0,16]); position_feet
    additionally subtracts the current actor height. field_position always specifies feet
    coordinates and approaches from field_approach (default up), beginning field_start_distance
    pixels away (default 8). Its field_trigger selects A or walk activation; field_repeat bounds
    additional interactions.

    A talk handler can follow the tile event. preparatory_talks may install that handler through
    earlier conversations, bounded by preparatory_talk_limit (default 3). deferred_quest_flags
    are applied only before the final talk. talk_repeat bounds repeated conversations after the
    first. No supported handler or an already-busy field produces handler_unavailable.

    Boolean options require bool values; numeric zero and one are rejected.
    Completed memory writes and captures remain after a failure; there is no rollback.

    Args:
        game (GameplayController): Controller updated in place, including
            captures and task state.
        task (dict[str, Any]): Scene route with actor/field positioning, interaction
            choices and frame_budget; optional wallet and stock fixtures are
            applied after initialization. talk selects a six-digit hexadecimal SNES
            handler address. Optional expected_actor_root restricts the final actor
            to a six-digit hexadecimal root-script address, including after approach
            and before every repeat. None or absence disables that restriction.
            root instead filters preparatory actors and remembered-actor approach;
            it does not replace expected_actor_root. The definition is not modified.
        early (bool, optional): Permit interaction while the room is still
            settling, once the requested handler is live. Defaults to False.

    Returns:
        str: Playback outcome from initialization, placement or conversation.
            A failed placement or prerequisite conversation stops later interactions.
            handler_unavailable indicates that the field is busy or no requested
            handler is available after successful setup. Pixel checks and required
            entry evaluation remain separate from this outcome.

    Raises:
        TypeError: If early is not a boolean.
        ValueError: If supported task fields, handler addresses, menu plans or
            fixture values are invalid, a memory write is out of bounds, or page
            verification rejects text or display data. Numeric setup fields accept
            integer subclasses and reject booleans.
        KeyError: If required tile-trigger data is missing or verified text contains
            an unsupported command.
        PlaybackStopped: If playback reaches a configured stop condition.
        RuntimeError: If the native relocation snapshot fails, a requested maze
            destination is disconnected, or a required frame snapshot is unavailable
            or failed.
        AssertionError: If relocation requires an active task or page verification
            requires a frame before one is captured.
        IndexError: If text verification reads beyond the ROM.
        OSError: If verification sources cannot be read or screenshots cannot be saved.
    """
    if type(early) is not bool:
        raise TypeError("Boolean options require bool values: early")
    _validate_task(task)
    budget = task.get("frame_budget", 24000)
    plan = task.get("choice_plan", []) if not task.get("talk") else None
    placed = task.get("position") or task.get("field_position")

    # An early actor can be caught before initialization dialogue begins.
    # Explicit placement tasks must still perform their normal room setup.
    if early and not placed and game.await_talk(task, budget):
        prep: DialogueOutcome = {"status": "idle", "choices": []}
    else:
        prep = game.play_dialogue("initialization", budget, plan=plan)

    interaction = None

    if (
        prep["status"] == "idle"
        and task.get("controller_phase") == "native_controller_interaction"
    ):
        # The task's own button presses (walking, menus) once the scene has initialized.
        interaction = game.play_dialogue("controller", budget, plan=plan)
        if interaction["status"] != "idle":
            return interaction["status"]

    if (
        prep["status"] == "idle"
        and task.get("reapply_quest_flags")
        and (task.get("talk") or task.get("field_position"))
    ):
        game.set_flags(task.get("quest_flags", []))

    if prep["status"] == "idle" and "wallet_fixture" in task:
        if (
            not is_integer(task["wallet_fixture"])
            or not 0 <= task["wallet_fixture"] <= 999999
        ):
            raise ValueError("Wallet fixture outside the legal money range")

        # WRAM $1AD6-$1AD8 holds the player's money as a little-endian integer.
        game.poke(0x1AD6, task["wallet_fixture"].to_bytes(3, "little"))

    if prep["status"] == "idle" and task.get("position"):
        x, y = task["position"]
        dx, dy = task.get("position_offset", [0, 16])

        if task.get(
            "position_feet"
        ):  # the coordinates are the player's feet, as on field tiles
            # $173A selects the player actor; $131F + slot stores its height in pixels.
            dy -= game.W[0x131F + game.W[0x173A]]

        game.relocate_player(x + dx, y + dy)

        if task.get("walk_until_text"):
            # Walk from there (e.g. into a laser beam) until the game starts its own text.
            prep = game.play_dialogue("position", budget)
            if prep["status"] != "idle":
                return prep["status"]
            walk = task["walk_until_text"]
            for _ in range(walk["frames"]):
                if game.is_busy():
                    break

                game.run(1, {BTN[b] for b in walk["buttons"]})

            prep = game.play_dialogue("walk", budget, plan=task.get("choice_plan", []))
        elif not (early and task.get("talk") and game.await_talk(task, 2400)):
            prep = game.play_dialogue("position", budget)
            if prep["status"] != "idle":
                return prep["status"]
            game.press("up", 2, 12)  # movement runs the game's spatial actor loader
            prep = game.play_dialogue("spawn", budget)

    # Tile coordinates describe the player's feet. Start beside the target,
    # subtract the actor height, then approach from the declared direction.
    if prep["status"] == "idle" and task.get("field_position"):
        x, y = task["field_position"]
        direction = task.get("field_approach", "up")
        distance = task.get("field_start_distance", 8)
        dx, dy = {
            "up": (0, distance),
            "down": (0, -distance),
            "left": (distance, 0),
            "right": (-distance, 0),
        }[direction]
        game.relocate_player(x + dx, y + dy - game.W[0x131F + game.W[0x173A]])
        prep = game.play_dialogue("field_position", budget)
        if prep["status"] != "idle":
            return prep["status"]

        if task.get("reapply_quest_flags"):
            game.set_flags(task.get("quest_flags", []))

        trigger = task["field_trigger"]
        game.press(
            direction,
            task.get(
                "field_move_frames", 12 if trigger.get("native_record_type") == 8 else 2
            ),
            12,
        )

        if trigger.get("activation", "a") == "a":
            game.press("a", 4, 0)

        before = len(game.pages)
        result = game.play_dialogue("tile", budget, plan=task.get("choice_plan", []))

        # Some walk-trigger records still need an A input after approach. Try
        # that fallback only when the declared handler does not check A and drew no page.
        if (
            trigger.get("activation") == "walk"
            and trigger.get("handler_checks_a_button") is False
            and result["status"] == "idle"
            and len(game.pages) == before
        ):
            game.press("a", 4, 0)
            result = game.play_dialogue(
                "tile", budget, plan=task.get("choice_plan", [])
            )

        # Repeat only while each interaction produces another page. Stop when
        # playback blocks or no new text appears rather than hammering an inactive trigger.
        for _ in range(task.get("field_repeat", 0)):
            if result["status"] != "idle":
                break

            before = len(game.pages)
            game.press("a", 4, 0)
            result = game.play_dialogue(
                "tile", budget, plan=task.get("choice_plan", [])
            )

            if len(game.pages) == before:
                break

        if not task.get("talk") or result["status"] != "idle":
            return result["status"]

        # A registered conversation can follow the tile interaction, e.g. Minerva after the seat.
    if prep["status"] == "idle" and task.get("talk"):
        candidates = game.find_talk_slots(task["talk"], task.get("expected_actor_root"))

        def has_preparatory_actor() -> bool:
            """
            Return whether a declared prerequisite talk handler is currently registered.

            Returns:
                bool: True if any preparatory handler matches a live actor, using
                    the optional root script address from the enclosing task.
            """
            return any(
                game.find_talk_slots(t, task.get("root"))
                for t in task.get("preparatory_talks", [])
            )

        if (
            not candidates and not has_preparatory_actor()
        ):  # (an earlier conversation first registers the handler)
            for _ in range(3):
                game.press("up", 16, 8)
                prep = game.play_dialogue("spawn_walk", 12000)
                candidates = game.find_talk_slots(
                    task["talk"], task.get("expected_actor_root")
                )
                if candidates or has_preparatory_actor() or prep["status"] != "idle":
                    break

        # A prior conversation may install the requested talk handler. Complete
        # only available prerequisite talks and rescan after each native conversation.
        for _ in range(task.get("preparatory_talk_limit", 3)):
            if candidates or prep["status"] != "idle":
                break

            available = [
                (slot, t)
                for t in task.get("preparatory_talks", [])
                for slot in game.find_talk_slots(t, task.get("root"))
            ]
            if not available or game.is_busy():
                break

            game.talk(available[0][0])
            prep = game.play_dialogue(
                "preparatory", 24000, plan=task.get("preparatory_choice_plan", [])
            )
            candidates = game.find_talk_slots(
                task["talk"], task.get("expected_actor_root")
            )

        if not candidates and prep["status"] == "idle":
            candidates = _filter_actor_candidates(
                game, task, game.approach_sighted(task["talk"])
            )

        if prep["status"] != "idle":
            return prep["status"]
        if not candidates or game.is_busy():
            return "handler_unavailable"

        if task.get("reapply_quest_flags"):
            game.set_flags(task.get("quest_flags", []))

        # Deferred flags belong to the final conversation state, after prerequisite
        # talks have finished; applying them earlier could change which actor is available.
        game.set_flags(task.get("deferred_quest_flags", []))
        actor = candidates[0]
        if "pickup_stock_fixture" in task:
            # Pickup handler $83:B086 reads the actor's item ID at $0E5A + slot.
            # Inventory quantities occupy WRAM $1AE9 + item and are capped at 80.
            quantity, item = task["pickup_stock_fixture"], game.W[0x0E5A + actor]
            if (
                int(task["talk"], 16) != 0x83B086
                or not 0 <= quantity <= 80
                or not 0 <= item < 98
            ):
                raise ValueError("Stock fixture outside the original item limits")

            game.poke(0x1AE9 + item, [quantity])

        game.talk(actor)
        result = game.play_dialogue(
            "conversation", 24000, plan=task.get("choice_plan", [])
        )
        for _ in range(task.get("talk_repeat", 0)):
            if (
                result["status"] != "idle"
                or game.is_busy()
                or actor
                not in game.find_talk_slots(
                    task["talk"], task.get("expected_actor_root")
                )
            ):
                break

            game.talk(actor)
            result = game.play_dialogue(
                "conversation", 24000, plan=task.get("choice_plan", [])
            )

        return result["status"]

    return (interaction or prep)["status"]
