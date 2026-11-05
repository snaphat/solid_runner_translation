"""
RouteRunner prepares field-route tasks, records their text captures, and returns
execution outcomes.

RouteRunner extends gameplay control with task fixtures, room setup and ordered prerequisite
scenes. It dispatches actor, tile, map, arena and calibrated-display tasks to their feature
modules and returns explicit outcomes for later evaluation. Prerequisite captures remain
attached to the task that needs them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from verification.game.control_gameplay import GameplayController, PlaybackStopped
from verification.routes import advance_arena as arena
from verification.routes import display_text_directly as direct_display
from verification.routes import interact_with_field as field
from verification.tasks import validate_requirements as requirements


# These names select the handlers below. BattleRunner handles ordinary battle
# and caption routes itself; this runner handles battle routes only when a field
# tile interaction starts the encounter.
SUPPORTED_ROUTES = frozenset(
    {
        "inputs",
        "progression",
        "arena",
        "forced",
        "direct",
        "scene",
        "map",
        "city",
        "dialogue-context",
        "npc",
        "npc-branches",
        "automatic",
        "choices",
        "battle",
    }
)


def _validate_route(task: dict[str, Any]) -> str:
    """
    Validate a task and identify the route handler supported by this runner.

    Args:
        task (dict[str, Any]): Definition read without mutation. route must name
            a supported handler; battle requires field_position because other
            battle starts are handled by BattleRunner.

    Returns:
        str: Validated route name. No emulator state or files are changed.

    Raises:
        ValueError: If the task is not a dictionary, a supported task field is
            invalid, the route is unknown, or battle lacks field_position.
        KeyError: If route is missing.
    """
    if not isinstance(cast(object, task), dict):
        raise ValueError("Route task must be an object")
    requirements.validate_definition(task)
    route = task["route"]
    if not isinstance(route, str) or route not in SUPPORTED_ROUTES:
        raise ValueError("Route must name a supported playback handler")
    if route == "battle" and not task.get("field_position"):
        raise ValueError("Battle routes without field_position require BattleRunner")
    return route


class RouteRunner(GameplayController):
    """
    This controller prepares route tasks and runs their prerequisite scenes.

    The inherited constructor establishes the emulator and capture state. start()
    must establish a free-roaming snapshot before a fresh room can be prepared.
    Playback changes emulator memory, controller settings and capture lists; failures
    do not roll back completed changes. Each worker supplies a fresh capture batch
    before running an independent task.
    """

    def observe_page(self, page: dict[str, Any]) -> None:
        """
        Apply arena introduction state before a target stop.

        Args:
            page (dict[str, Any]): Newly captured page, read without mutation.
                entry=0A09A8 identifies the arena introduction that reapplies the
                active arena task's encounter-selection flags in SRAM. Other
                entries leave emulator memory unchanged through this hook.

        Raises:
            ValueError: If an arena fixture contains an invalid flag or memory write.
            KeyError: If an active arena task lacks its quest_flags field.
        """
        arena.advance_intro(self, page)

    def begin(self, task: dict[str, Any]) -> None:
        """
        Prepare a task's room, progression and declared fixtures.

        Args:
            task (dict[str, Any]): Native stage and variant bytes, each 0-255, with
                optional quest_flags (SRAM address/mask/value records),
                chained/continue_state reuse, RNG seeds and inventory fixtures;
                stored by reference as the active task. Requires stage, variant, and
                a supported route; derive_entry_setup() additionally uses the task key. This can
                restore a cached state, write flags/inventory, and request a room load;
                later playback must complete that load.

        Raises:
            AssertionError: If a fresh start is needed but start() has not established
                base, or page verification requires a frame before one is captured.
            RuntimeError: If the emulator rejects a state, the native field snapshot
                fails, the entry prelude does not finish, or a required frame snapshot
                is unavailable or failed.
            ValueError: If a task numeric input has an invalid type or value, or an
                inventory fixture exceeds the original item limits. Also raised for
                invalid room-specific setup, flag writes, text or display data.
            KeyError: If a required stage, variant or route field is missing,
                or verified text contains an unsupported command.
            PlaybackStopped: If setup reaches a configured target or stable-battle stop.
            IndexError: If text verification reads beyond the ROM.
            OSError: If verification sources cannot be read or screenshots cannot be saved.
        """
        # Derive setup before loading a state or advancing frames so malformed room
        # fields cannot partially apply setup before they are rejected.
        route = _validate_route(task)
        setup = field.derive_entry_setup(task)
        # chained retains the prerequisite scene's state. continue_state also
        # skips the room load, allowing playback to resume inside a battle or scene.
        in_place = task.get("continue_state")
        if not (in_place or task.get("chained")):
            if self.base is None:
                raise AssertionError("Free-roaming state has not been initialized")

            self.emu.load_state(self.base)

        self.controller, self.player_control_seen = False, False
        self.task, self.entry_setup, self.advance, self.counted = (
            task,
            setup,
            0,
            None,
        )
        self.health_fixed, self.defeated, self.turn_ends, self.at_turn_end = (
            set(),
            set(),
            {},
            False,
        )
        self.maze_goal = self.maze_steps = 0
        self.watch, self.sighted = None, None
        # run() reads the active task's fixtures and observation state. Install
        # the new task before advancing so the preceding task cannot affect it.
        if task.get("rng_seed") is not None:
            # The RNG routine at $81:824C reads table[$81:825C + (++$0000)].
            # Set the initial table index; later native calls still advance it.
            self.poke(0x0000, [task["rng_seed"]])
        self.run(1)
        self.watch = task.get("talk")
        if route in ("dialogue-context", "battle"):
            # Context routes handle battle and result screens. Snapshot the field
            # position so later coordinate loads can restore it.
            self.entry_setup = dict(self.entry_setup or {}, context=True)
            if not self.is_busy() and not in_place:
                self.save_field_snapshot()

        self.clear_log()

        if self.entry_setup and self.entry_setup.get("save_return_packet"):
            # The game's own field-position snapshot ($80:CF37), run on the idle carrier actor.
            self.start_script(bytes.fromhex("05 37 cf 80 00"))
            self.run(8)

        if self.entry_setup and self.entry_setup.get("prelude"):
            self.load_room(self.entry_setup["prelude"])

            if self.play_dialogue("prelude", 24000)["status"] != "idle":
                raise RuntimeError("Native entry prelude did not finish")

        self.set_flags(task.get("quest_flags", []))

        if task.get("room_rng_seed") is not None:
            # Seed the encounter draw after prerequisites, whose own random draws differ
            # between the Japanese and English dialogue timings.
            self.poke(0x0000, [task["room_rng_seed"]])

        if in_place:
            if self.is_busy():
                self.advance = (
                    4  # a page from before this scene may be waiting for input
                )
        else:
            self.load_room(
                (self.entry_setup or {}).get(
                    "entry_room", [task["stage"], task["variant"]]
                )
            )
            # SRAM $013E selects the report on the mission-results screen; start
            # room playback at report zero rather than retaining an earlier index.
            self.poke(0x13E, [0], region=0)

        for stock in task.get("inventory_fixture", []):
            if not 0 <= stock["item"] < 98 or not 0 <= stock["quantity"] <= 80:
                raise ValueError("Inventory fixture outside the original item limits")

            # WRAM $1AE9 + item stores the carried quantity for each of 98 items.
            self.poke(0x1AE9 + stock["item"], [stock["quantity"]])

    def play_before(self, before: dict[str, Any]) -> bool:
        """
        Run a task's ordered prerequisite scene and retain its evidence.

        Full playback can record persistent state for later calibrated runs. A failed or
        incomplete prerequisite outside the permitted endpoint statuses is not cached as a
        usable start. Its own target-not-reached statuses can still be accepted: a prerequisite
        needs the resulting game state, not all of its own standalone evidence requirements.

        Args:
            before (dict[str, Any]): Prerequisite route definition, optionally ending at
                a ready battle with stop='battle'. The dictionary is not modified.
                Playback changes emulator state and appends captures. The nesting
                counter and any temporary battle-stop settings are restored on exit.

        Returns:
            bool: Whether prerequisite playback reached the point required by the
                later route. A normal prerequisite accepts idle, targets_incomplete,
                target_not_reached, or repeated_script. A stop=battle prerequisite
                succeeds only when PlaybackStopped signals the ready-battle boundary.

        Raises:
            ValueError: If prerequisite setup, fixtures, text or display data are invalid.
            TypeError: If a nested prerequisite contains values JSON cannot serialize.
            KeyError: If the prerequisite lacks a required field or verified text
                contains an unsupported command.
            PlaybackStopped: If a normal prerequisite reaches a configured stop.
                Battle prerequisites instead return False for a stop that occurs
                before the battle has remained ready for 30 frames.
            RuntimeError: If emulator state save/restore, native setup or a required
                frame snapshot fails, or a requested maze destination is disconnected.
            AssertionError: If required controller setup or a frame is unavailable.
            IndexError: If text verification reads beyond the ROM.
            StopIteration: If prerequisite map-label playback cannot allocate an actor slot.
            OSError: If verification sources cannot be read or screenshots cannot be saved.
        """
        if not isinstance(cast(object, before), dict):
            raise ValueError("Prerequisite task must be an object")
        requirements.validate_definition(before)
        if before.get("stop") == "battle":
            # The earlier scene ends where its battle is ready; the later task's fixtures then fight
            # it.
            previous_halt, previous_ready = self.halt, self.halt_ready
            self.halt, self.halt_ready = "battle", 0
            self.in_before += 1
            try:
                self.run_route({k: v for k, v in before.items() if k != "stop"})
                return False
            except PlaybackStopped:
                # Target-page stops use the same exception. Only the controller's
                # 30-frame stable-battle counter proves the requested endpoint.
                return self.halt_ready >= 30
            finally:
                self.halt, self.halt_ready = previous_halt, previous_ready
                self.in_before -= 1

        # Reaching a repeated script is a valid prerequisite endpoint.
        self.in_before += 1

        try:
            return self.run_route(before) in (
                "idle",
                "targets_incomplete",
                "target_not_reached",
                "repeated_script",
            )
        finally:
            self.in_before -= 1

    def run_route(self, task: dict[str, Any]) -> str:
        """
        Replay a route and prerequisites with explicit outcome reporting.

        Args:
            task (dict[str, Any]): Route to replay, including optional before prerequisites
                and expect/target requirements used for stopping and outcome
                evaluation. The definition is not modified; after a prerequisite,
                a shallow copy with chained=True becomes the active task. The caller
                resets pages and prerequisite_pages before an independent task;
                nested prerequisite calls retain the enclosing capture batch.
                Route names select actor, scene, map, arena, progression, input or
                direct-display playback. A battle route with field_position uses
                scene playback to trigger its encounter; ordinary battle and caption
                tasks require BattleRunner. Prerequisite definitions must contain
                JSON-serializable values because their serialized form keys the cache.
                A cached native prerequisite is reused only when its captures are
                also available. An explicitly supplied persistent-state record can
                restore setup without supplying native text evidence.

        Returns:
            str: Feature playback status, possibly adjusted to targets_incomplete
                or target_not_reached for missing declared targets. Captures remain
                on pages/prerequisite_pages rather than being returned. Prerequisite
                failure returns before_scene_incomplete. Exceptions from invalid task
                setup, emulation, or deliberate stop boundaries can still propagate.
                Retained prerequisite captures count toward target presence regardless
                of whether their scene was played now or restored from a cache;
                result evaluation checks their pixels and source hashes separately.

        Raises:
            ValueError: If route is unsupported, a supported setup or
                requirement field is invalid, or playback rejects text or display data.
            KeyError: If route or another field required by the selected helper is
                missing, or verified text contains an unsupported command.
            TypeError: If prerequisite definitions contain values JSON cannot serialize.
            PlaybackStopped: If playback reaches a configured stop not consumed by
                prerequisite or arena handling.
            RuntimeError: If emulator state save/restore, native setup or a required
                frame snapshot fails, or a requested maze destination is disconnected.
            AssertionError: If required controller setup or a frame is unavailable.
            IndexError: If text verification reads beyond the ROM.
            StopIteration: If map-label playback cannot allocate an actor slot.
            OSError: If verification sources cannot be read or screenshots cannot be saved.
        """
        route = _validate_route(task)
        # Spawn counters are local to an emulated state. Clear the recorded script starts
        # before restoring or beginning another route.
        self.spawns = {}  # stamps repeat after a state is loaded
        if task.get("before"):  # an earlier scene whose outcome this one needs
            # Only the prerequisite's endpoint matters, not all of its own targets.
            # Reuse a cached emulator state when available. Quick replay may instead
            # restore a recorded persistent outcome; otherwise play the scene natively.
            before = task["before"]
            first_page, first_prerequisite = (
                len(self.pages),
                len(self.prerequisite_pages),
            )
            key = json.dumps(before, sort_keys=True)
            name = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            # start() can share emulator states across runner instances, while
            # chain_pages belongs to this instance. Without those captures, replay
            # the native prerequisite unless a persistent-state replay was configured.
            if key in self.chains and (
                key in self.chain_pages or name in self.chain_states
            ):
                self.emu.load_state(self.chains[key])
                self.prerequisite_pages.extend(self.chain_pages.get(key, []))
            elif name in self.chain_states:
                # The quick run: the earlier scene's outcome as the game's persistent state,
                # recorded in a full run (direct.json 'chains'), set on the free-roaming point
                # instead of playing it.
                self.set_persistent(self.chain_states[name])
                self.chains[key] = self.emu.save_state()
            elif not self.play_before(before):
                return "before_scene_incomplete"
            else:
                self.chains[key] = self.emu.save_state()
                self.chain_pages[key] = (
                    list(self.pages[first_page:])
                    + self.prerequisite_pages[first_prerequisite:]
                )
                if self.record_chains:  # full runs: keep the outcome for the quick run
                    self.chain_records[name] = self.serialize_persistent_state()

            task = dict(task, chained=True)

        self.targets_left = None
        if self.early_stop and not self.in_before and task.get("expect"):
            # 'expect': every (entry, page) this task verified on a run without early stops
            # (verify_gameplay.py --record-expect); the Japanese ROM only identifies entries, so it
            # waits for each expected entry.
            wanted = requirements.collect_required_entries(task)
            self.targets_left = (
                {
                    (entry, page)
                    for entry, page in self.starts.values()
                    if entry in wanted
                }
                if self.english
                else set[tuple[str, int] | str](wanted)
            )

        if task["route"] == "inputs":
            return field.apply_input_sequence(self, task)

        if task["route"] == "progression":
            return arena.advance_levels(self, task)

        if task["route"] == "arena":
            return arena.enter_round(self, task)

        self.begin(task)

        if task["route"] in ("forced", "direct"):
            return direct_display.show_text(self, task)

        if task["route"] == "scene":
            return direct_display.show_scene(self, task)

        if task["route"] == "map":
            return direct_display.show_map_labels(self, task)

        if task["route"] == "city":
            return field.visit_locations(self, task)

        if route in ("dialogue-context", "battle"):
            # 'talk_early': this NPC walks off-screen and is unloaded before the room falls quiet,
            # so the player talks as soon as its handler is live (as a player would).
            outcome = field.play_scene(self, task, early=bool(task.get("talk_early")))
        else:
            outcome = field.interact_with_actor(self, task)

        # Cached prerequisite captures satisfy the same requirements as captures
        # from a prerequisite played during this call; evaluation checks their pixels.
        seen = {p.get("entry") for p in self.pages + self.prerequisite_pages}
        targets = set(task.get("targets") or [])
        if outcome == "idle" and targets - seen:
            outcome = "targets_incomplete" if targets & seen else "target_not_reached"

        return outcome
