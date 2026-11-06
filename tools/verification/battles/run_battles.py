"""
BattleRunner sets up encounters, runs controller sequences, and captures battle
text for verification.

BattleRunner reaches named or prerequisite battle starts, submits timed controller sequences,
applies declared health/outcome fixtures and captures announcements. Calibrated caption tasks
are tried in their recorded native battle contexts. Task results retain both setup evidence and
the final execution outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, cast

from core.validation.validate_integers import is_integer
from emulator.validate_controller_input import collect_button_ids, validate_frame_count

from verification.evidence import evaluate_results as evaluation
from verification.game import apply_battle_fixtures as battle_fixtures
from verification.game.control_gameplay import BTN, PlaybackStopped
from verification.routes import advance_arena as arena
from verification.routes.run_routes import RouteRunner
from verification.tasks import validate_requirements as requirements
from verification.tasks.load_tasks import load_battles as tasks
from verification.tasks.load_tasks import load_routes


class _BattleStartUnavailable(RuntimeError):
    """
    This exception means native setup returned without reaching its required endpoint.

    Caption retries may remember this failure for the affected start. Other runtime
    errors, such as a missing video frame during caption verification, do not imply
    that the start itself is unavailable.
    """


def _validate_task(task: dict[str, Any]) -> None:
    """
    Validate shared task fields and battle-response fixtures before playback.

    Args:
        task (dict[str, Any]): Definition read without mutation. Numeric settings
            accept integer subclasses and reject booleans. response_fixture is
            None, hit, evasion or escape-failure.

    Raises:
        ValueError: If task is not a dictionary or a supported field is invalid.
    """
    if not isinstance(cast(object, task), dict):
        raise ValueError("Battle task must be an object")
    requirements.validate_definition(task)
    if task.get("response_fixture") not in (None, "hit", "evasion", "escape-failure"):
        raise ValueError(
            "response_fixture must be hit, evasion, escape-failure or None"
        )


def _decode_caption_state(state: str | None) -> bytes | None:
    """
    Decode the memory ranges recorded for one caption invocation.

    Args:
        state (str | None): Hexadecimal bytes for WRAM $1820-$18AF and
            $16BC-$16BD, optionally followed by SRAM $0000-$0212. None means
            no override; an empty string is invalid.

    Returns:
        bytes | None: 146 or 677 decoded bytes, or None. No emulator memory or
            files are changed.

    Raises:
        ValueError: If state is not hexadecimal text of either supported size.
    """
    if state is None:
        return None
    if not isinstance(cast(object, state), str):
        raise ValueError("Battle state must be hexadecimal text or None")
    block = bytes.fromhex(state)
    if len(block) not in (146, 677):
        raise ValueError(
            "Battle state must contain 146 WRAM bytes and optionally 531 SRAM bytes"
        )
    return block


def _parse_caption_definition(task: dict[str, Any]) -> tuple[int, str, int]:
    """
    Validate the script, text identifier and parent slot before a caption attempt.

    Args:
        task (dict[str, Any]): Caption definition, read without mutation. script
            and text are six-digit hexadecimal addresses; text uses uppercase
            digits to match capture IDs. parent is an integer actor slot 0-36,
            excluding booleans; integer subclasses are accepted.

    Returns:
        tuple[int, str, int]: Parsed SNES script address, unchanged text entry ID,
            and parent actor slot. No emulator state or files are changed.

    Raises:
        ValueError: If supported task fields or either address are invalid.
        KeyError: If script, text or parent is missing.
    """
    _validate_task(task)
    script, text, parent = task["script"], task["text"], task["parent"]
    for label, address in (("script", script), ("text", text)):
        alphabet = "0123456789ABCDEF" if label == "text" else "0123456789ABCDEFabcdef"
        if (
            not isinstance(address, str)
            or len(address) != 6
            or any(character not in alphabet for character in address)
        ):
            raise ValueError(f"Caption {label} must contain six hexadecimal digits")
    return int(script, 16), text, parent


class BattleRunner(RouteRunner):
    """
    This controller reaches battle starts and records battle text for verification.

    It extends RouteRunner with battle input timing, encounter fixtures, cached
    native start states and direct caption attempts. Playback changes emulator
    memory and captured pages. Callers reset captures before independent tasks and
    evaluate the resulting pages separately from the execution status.
    """

    def __init__(
        self,
        rom: bytes | bytearray,
        scratch: Path,
        english: bool = True,
        shots: Path | None = None,
        accumulate: bool = False,
    ) -> None:
        """
        Initialize battle-start, capture and spawned-script caches.

        Boolean options require bool values; numeric zero and one are rejected.

        Args:
            rom (bytes | bytearray): Uninstrumented source ROM retained by reference.
                The buffer must remain unchanged while the session is open.
            scratch (Path): Output path for the ROM file with tracing code added;
                its parent directory must exist. The input ROM is not modified.
            english (bool, optional): True for English, False for the Japanese original.
                Defaults to True.
            shots (Path | None, optional): Screenshot directory; None disables saving.
                Defaults to None.
            accumulate (bool, optional): Retain glyph traces across interface draws.
                Defaults to False.

        Raises:
            TypeError: If rom is not bytes or bytearray, or english or accumulate
                is not a boolean.
            OSError: If instrumentation, core or translation files cannot be read,
                or the instrumented ROM file cannot be written.
            ValueError: If instrumentation, source metadata, text chains or required
                emulator memory are invalid.
            RuntimeError: If the core cannot load the instrumented ROM.
            IndexError: If text indexing reads beyond the ROM.
            KeyError: If indexed text contains an unsupported command.
        """
        if type(english) is not bool or type(accumulate) is not bool:
            raise TypeError("Boolean options require bool values: english, accumulate")
        super().__init__(
            rom, scratch, english=english, shots=shots, accumulate=accumulate
        )
        self.cache = {}
        # State snapshots need both their captures and spawned-script metadata;
        # another runner can share the state cache without sharing these records.
        self.cache_spawns: dict[str, dict[int, tuple[int, int, int, str]]] = {}
        # Menu answers belong to the same script invocation as its spawn record.
        self.cache_spawn_choices: dict[str, dict[int, list[int | str]]] = {}
        self.cache_pages: dict[str, list[dict[str, Any]]] = {}
        # Only explicit native endpoint failures exclude a start from later attempts.
        self.unreached: set[str] = set()
        self.stop: str | None = None
        self.steps: list[dict[str, Any]] | None = None
        self.origin = self.ready = self.submitted = 0
        self.story = self.phased = False
        self.condition_seeded = self.response_seeded = False

    def begin(self, task: dict[str, Any]) -> None:
        """
        Prepare a field or arena route without retaining previous battle overrides.

        The parent prepares the room and native fixtures. Battle input steps,
        story/phase modes and condition/response application flags are cleared before
        it advances frames, so a previous attempt cannot affect the new route.

        Args:
            task (dict[str, Any]): Route definition accepted by RouteRunner.begin(),
                including key, route, stage and variant. The parent retains the
                dictionary by reference but does not modify it. Emulator memory,
                task settings and captures can change during preparation.

        Raises:
            ValueError: If route, room setup, fixtures, text or display data are invalid.
            KeyError: If a required task field is missing or a text command is unsupported.
            RuntimeError: If state restoration, native setup or a required frame fails.
            PlaybackStopped: If preparation reaches a configured stop boundary.
            AssertionError: If a required base state or video frame is unavailable.
            IndexError: If text verification reads beyond the ROM.
            OSError: If sources cannot be read or screenshots cannot be saved.
        """
        _validate_task(task)
        self.steps = None
        self.story = self.phased = False
        self.condition_seeded = self.response_seeded = False
        self.submitted = 0
        super().begin(task)

    # Controller timing and battle-state updates.
    def reset_battle_observation(self, task: dict[str, Any]) -> None:
        """
        Clear battle-local tracking before observing a route or direct caption.

        Args:
            task (dict[str, Any]): Active battle or caption definition retained
                for fixture handling and capture attribution. An optional rng_seed
                sets the native random-table index for this attempt.
                Input steps, story/phase overrides and condition/response application
                flags are reset. task is retained by reference but is not modified.

        Raises:
            ValueError: If a supported task field or random-table index is invalid.
        """
        _validate_task(task)
        self.task, self.entry_setup, self.advance, self.counted = task, None, 0, None
        self.health_fixed, self.watch, self.sighted = (
            set(),
            None,
            None,
        )
        self.condition_seeded = self.response_seeded = False
        self.defeated, self.turn_ends, self.at_turn_end = set(), {}, False
        self.maze_goal = self.maze_steps = 0
        self.story, self.phased, self.submitted = False, False, 0
        self.steps = None
        self.controller, self.menu_open = False, False

        # $81:824C advances WRAM $0000 before reading its random-number table.
        # Restore the declared index after loading the battle state; the native
        # combat routines still decide hits, misses, damage and status effects.
        if task.get("rng_seed") is not None:
            self.poke(0x0000, [task["rng_seed"]])

    def count_sequence_frames(self) -> int:
        """
        Return the active controller sequence's total duration in emulated frames.

        Returns:
            int: Sum of configured controller-step frame counts, or zero without a
                controller sequence.

        Raises:
            ValueError: If a step duration is a boolean, noninteger or negative.
            KeyError: If an active step lacks frames.
        """
        steps = self.steps or []
        for step in steps:
            validate_frame_count(step["frames"])
        return sum(step["frames"] for step in steps)

    def count_elapsed_frames(self) -> int:
        """
        Return submitted controller frames for story battles, elapsed frames otherwise.

        Returns:
            int: Submitted sequence frames in story mode, otherwise emulated frames
                since this battle attempt began.
        """
        return self.submitted if self.story else self.frame - self.origin

    def get_native_inputs(self) -> set[int]:
        """
        Return held libretro button IDs for this frame, empty during dialogue.

        Returns:
            set[int]: Libretro button IDs for the current sequence frame, or an empty
                set during dialogue or after the sequence ends. Eligible calls advance
                the submitted-frame counter.

        Raises:
            ValueError: If a step duration is a boolean, noninteger or negative.
            AssertionError: If no input sequence is configured when input is requested.
            KeyError: If a sequence names an unsupported controller button or a step
                lacks frames or buttons.
        """
        # Story battles count submitted inputs, so dialogue pauses their sequence.
        # Ordinary battles use elapsed emulator frames even while input is blocked.
        if RouteRunner.is_busy(self):
            return set()

        if self.steps is None:
            raise AssertionError("Battle controller has no input sequence")
        self.count_sequence_frames()
        age = self.count_elapsed_frames()
        self.submitted += 1

        # Durations are relative to successive steps, so subtract completed steps
        # until the elapsed sequence frame lands in the current held-button interval.
        for step in self.steps:
            if age < step["frames"]:
                return {BTN[b] for b in step["buttons"]}

            age -= step["frames"]

        return set()

    def run(self, n: int = 1, held: Iterable[int] = ()) -> None:
        """
        Advance battle input and declared fixtures.

        Args:
            n (int, optional): Nonnegative emulated-frame count, excluding booleans;
                integer subclasses are accepted. Defaults to 1.
            held (Iterable[int], optional): Explicit libretro button IDs. Empty
                allows eligible battle-sequence or shared task-controller input;
                nonempty overrides both automatic sequences. Defaults to ().

        Raises:
            ValueError: If a frame count or step duration is not a nonnegative
                integer excluding booleans, a button ID is unsupported, a health
                setting is not an integer in 0-65535, phase_sram is not an integer
                offset in 0-131071, or phase_mask is not an integer byte in 0-255.
                Booleans are rejected for health and phase settings.
                A phase-flag offset must also fit the exposed SRAM when enabled.
            PlaybackStopped: If a stable native battle boundary requested by the task
                has been reached.
            TypeError: If held is not iterable.
        """
        validate_frame_count(n)
        held_buttons = collect_button_ids(held)
        self.get_opponent_health_fixture()
        for name, maximum in (
            ("phase_enemy_hp", 0xFFFF),
            ("phase_sram", 0x1FFFF),
            ("phase_mask", 255),
        ):
            definition = self.task or {}
            value = definition.get(name)
            required = name in definition and name != "phase_enemy_hp"
            if (required or value is not None) and (
                not is_integer(value) or not 0 <= value <= maximum
            ):
                raise ValueError(
                    f"{name} must be an integer from zero through {maximum}"
                )
        if (self.task or {}).get("phase_enemy_hp"):
            address = (self.task or {}).get("phase_sram", 0xC0)
            if not is_integer(address) or not 0 <= address < len(self.S):
                raise ValueError("phase_sram lies outside the exposed SRAM")
        for _ in range(n):
            buttons = held_buttons.copy()
            if (
                not buttons
                and self.steps
                and self.count_elapsed_frames() < self.count_sequence_frames()
                and not self.menu_open
            ):
                buttons = self.get_native_inputs()

            w = self.W

            if self.story:
                battle_fixtures.refill_player_health(self)

            phase = self.task.get("phase_enemy_hp") if self.task else None
            if (
                phase
                and not self.phased
                and self.S[(self.task or {}).get("phase_sram", 0xC0)]
                & (self.task or {}).get("phase_mask", 2)
                and not w[0x18B1] & 4
            ):
                # Once the original phase script has set its save flag and released the battle,
                # lower the next phase's health once (within the maximum); the native damage routine
                # still destroys the enemy.
                current, maximum = (
                    w[0x189C] | w[0x189D] << 8,
                    w[0x1898] | w[0x1899] << 8,
                )
                if current > phase and 1 <= phase <= maximum:
                    self.poke(0x189C, phase.to_bytes(2, "little"))

                self.phased = True

            super().run(1, buttons)

            if self.task and self.task.get("defeat_fixture"):
                # Opt-in deterministic outcome fixture: defeat each
                # opponent at the post-turn boundary through battle_fixtures.finish_battle.
                battle_fixtures.finish_battle(self)

            if (
                self.task
                and self.task.get("condition_fixture")
                and not self.condition_seeded
            ):
                # $90:D36E waits for the turn animation before condition processing and the post-turn
                # monitor ($90:DA60); seed the listed conditions there, once.
                if (
                    w[0x1835] == 0x6E
                    and w[0x1836] == 0xD3
                    and not (w[0x16BC] & 0x20 or w[0x16BD] & 0x20)
                ):
                    for flag in self.task["condition_fixture"]:
                        self.poke(
                            flag["address"],
                            [(w[flag["address"]] & ~flag["mask"]) | flag["value"]],
                        )

                    self.condition_seeded = True

            seed = self.task.get("response_fixture") if self.task else None
            turn = w[0x183B] & 63
            if (
                seed
                and not self.response_seeded
                and w[0x1835] == 0x6E
                and w[0x1836] == 0xD3
                and turn == {"evasion": 2, "escape-failure": 5}.get(seed, turn)
                and not (w[0x16BC] & 0x20 or w[0x16BD] & 0x20)
            ):
                # Seed one response display at the action boundary: a 1-point hit ($1894), the
                # player's evasion flag ($16BC bit 2) or a failed escape ($1821 bits 1-2); the game
                # prints its own reaction. It does not show that an action caused the outcome.
                if seed == "hit":
                    if w[0x1898] | w[0x1899] << 8:
                        self.poke(0x1894, [1, 0])
                        self.response_seeded = True
                elif seed == "evasion":
                    self.poke(0x16BC, [w[0x16BC] | 4])
                    self.response_seeded = True
                elif seed == "escape-failure":
                    self.poke(0x1821, [w[0x1821] & ~6])
                    self.response_seeded = True

            if self.stop == "battle":
                ready = (
                    w[0x179C] == 120
                    and w[0x183B] == 0x40
                    and not RouteRunner.is_busy(self)
                    and not w[0x1719] & 0x80
                )
                self.ready = self.ready + 1 if ready else 0
                if self.ready >= 30:
                    raise PlaybackStopped

    def is_busy(self) -> bool:
        """
        Check dialogue waits and unfinished battle controller steps.

        Returns:
            bool: Whether dialogue or unfinished battle-controller steps prevent
                the runner from treating the current frame as idle.

        Raises:
            ValueError: If an active controller-step duration is invalid.
            KeyError: If an active step lacks frames.
        """
        return super().is_busy() or bool(
            self.steps and self.count_elapsed_frames() < self.count_sequence_frames()
        )

    # Battle-start preparation and playback.
    def load_start_state(self, name: str) -> bytes:
        """
        Return a cached battle state, or reach and cache its native start.

        It appends prerequisite pages even when loading a cached start. A cache hit
        returns bytes without restoring the emulator. Uncached starts advance the emulator
        through their prerequisite scenes without early target stops; the caller must load the
        returned state before beginning its attempt. A shared state without this
        runner's capture and spawn records is rebuilt through native playback.
        Rebuilding disables targets_left stopping; battle playback configures its
        own targets after loading the returned state.

        Args:
            name (str): Native battle-start key: route:<key>, arena:<key> or a
                supported named start.

        Returns:
            bytes: Serialized emulator state at the requested native battle
                boundary.

        Raises:
            ValueError: If name is invalid or native setup rejects a task, fixture,
                text chain, font, frame or VRAM buffer.
            RuntimeError: If native setup cannot reach its endpoint or an emulator
                state operation or required frame snapshot fails.
            PlaybackStopped: If native setup reaches a different stop boundary.
            StopIteration: If a route or arena key has no maintained task.
            KeyError: If setup lacks a required field or text uses an unsupported command.
            IndexError: If text verification reads beyond the ROM.
            AssertionError: If required controller setup or a frame is unavailable.
            OSError: If sources cannot be read or screenshots cannot be saved.
        """
        if not isinstance(cast(object, name), str) or not name:
            raise ValueError("Battle start must be a nonempty string")
        if (
            name in self.cache
            and name in self.cache_pages
            and name in self.cache_spawns
            and name in self.cache_spawn_choices
        ):
            self.prerequisite_pages.extend(self.cache_pages.get(name, []))
            return self.cache[name]

        # A start is an earlier scene played for its outcome: in full, never ended early
        # (in_before).
        self.targets_left = None
        self.in_before += 1

        try:
            state = self.build_start(name)
            self.prerequisite_pages.extend(self.cache_pages.get(name, []))
            return state
        finally:
            self.in_before -= 1

    def build_start(self, name: str) -> bytes:
        """
        Reach and cache a named route/arena battle start.

        The cache retains captures from this setup and its nested prerequisites,
        script-spawn records and the menu answers recorded for those scripts.
        Captures inherited before this call are excluded from the cached record.

        Args:
            name (str): Native battle-start key: route:<key>, arena:<key> or a
                supported named start.

        Returns:
            bytes: Serialized emulator state at the requested native battle
                boundary.

        Raises:
            ValueError: If the named battle-start kind is unknown, or setup rejects
                task, fixture, text or display data.
            RuntimeError: If native playback cannot reach the requested battle start.
            StopIteration: If a route: or arena: key has no matching maintained task.
            PlaybackStopped: If playback stops before this runner's stable-battle
                counter reaches 30. Existing stop settings are restored on exit.
            KeyError: If setup lacks a required task field or a text command is unsupported.
            IndexError: If text verification reads beyond the ROM.
            AssertionError: If required controller setup or a video frame is unavailable.
            OSError: If task or verification sources cannot be read or screenshots cannot be saved.
        """
        if not isinstance(cast(object, name), str) or not name:
            raise ValueError("Battle start must be a nonempty string")
        first_prerequisite = len(self.prerequisite_pages)
        self.pages = []

        # Battle starts stop playback at a native boundary via PlaybackStopped.
        # A normal return from a route is a failure to reach that requested boundary.
        if name.startswith("route:"):
            task = next(t for t in load_routes() if t["key"] == name[6:])
            previous_stop, previous_ready = self.stop, self.ready
            self.stop, self.ready = "battle", 0
            try:
                self.run_route(task)
                raise _BattleStartUnavailable(
                    f"start {name}: the battle was not reached"
                )
            except PlaybackStopped:
                if self.ready < 30:
                    raise
            finally:
                self.stop, self.ready = previous_stop, previous_ready
        elif name.startswith("arena:"):
            task = next(t for t in tasks() if t["key"] == name[6:])
            if arena.enter_round(self, task) != "idle":
                raise _BattleStartUnavailable(
                    f"start {name}: the announcement was not completed"
                )
        elif name == "arena-combat":
            announcement = "arena:arena-default-c0-s0-r1"
            self.emu.load_state(self.load_start_state(announcement))
            self.spawns = dict(self.cache_spawns[announcement])
            self.spawn_choices = {
                slot: list(answers)
                for slot, answers in self.cache_spawn_choices[announcement].items()
            }
            previous_stop, previous_ready = self.stop, self.ready
            self.stop, self.ready = "battle", 0
            try:
                for _ in range(60):  # through the remaining pages and the arena screens
                    self.play_dialogue("combat entry", 2000, 200)
                    self.press("a", 4, 60)

                raise _BattleStartUnavailable(
                    "start arena-combat: the battle was not reached"
                )
            except PlaybackStopped:
                if self.ready < 30:
                    raise
            finally:
                self.stop, self.ready = previous_stop, previous_ready
        else:
            raise ValueError(f"Unknown battle start: {name}")

        # Cache the native state together with spawned-script contexts and captures;
        # restoring only RAM would lose the evidence collected while reaching the start.
        self.cache[name] = self.emu.save_state()
        self.cache_spawns[name] = dict(
            self.spawns
        )  # scripts the game spawned while it was reached
        self.cache_pages[name] = (
            list(self.pages) + self.prerequisite_pages[first_prerequisite:]
        )
        self.cache_spawn_choices[name] = {
            slot: list(answers) for slot, answers in self.spawn_choices.items()
        }
        return self.cache[name]

    def load_origin_state(self, origin: dict[str, Any]) -> bytes:
        """
        Reach the native endpoint of a battle's prerequisite chain.

        An origin task may run to completion or stop at a ready battle. Its before field
        identifies any earlier scene needed to establish that state. A ref uses a named start
        instead. Cached calls return state bytes and append prerequisite evidence without
        restoring the emulator. Uncached playback collects its captures separately,
        then appends them to the preceding captures even if playback raises.

        Args:
            origin (dict[str, Any]): Task chain describing native battle progression and its
                optional stop boundary, None or battle. Alternatively ref names a
                route:<key>, arena:<key> or arena-combat start. The definition is
                read without mutation. Captures and native state are cached together
                with spawned-script records; incomplete cache entries are rebuilt.

        Returns:
            bytes: Serialized emulator state at the requested native battle
                boundary.

        Raises:
            RuntimeError: If an origin-chain task does not complete its native setup.
            PlaybackStopped: If playback stops somewhere other than the requested
                ready battle. Prior stop settings and prerequisite depth are restored.
            ValueError: If origin is not an object, its stop or referenced name is
                invalid, or setup rejects task, text or display data.
            TypeError: If origin includes values JSON cannot serialize.
            StopIteration: If a referenced route or arena key has no maintained task.
            KeyError: If setup lacks a required field or text uses an unsupported command.
            IndexError: If text verification reads beyond the ROM.
            AssertionError: If required controller setup or a frame is unavailable.
            OSError: If sources cannot be read or screenshots cannot be saved.
        """
        if not isinstance(cast(object, origin), dict):
            raise ValueError("Battle origin must be an object")
        _validate_task(origin)
        if origin.get("stop") not in (None, "battle"):
            raise ValueError("Origin stop must be battle or None")
        if "ref" in origin:
            return self.load_start_state(origin["ref"])

        # Canonicalize the whole prerequisite description so equivalent origin
        # objects share a cached setup regardless of dictionary key order.
        key = json.dumps(origin, sort_keys=True)
        if (
            key not in self.cache
            or key not in self.cache_pages
            or key not in self.cache_spawns
            or key not in self.cache_spawn_choices
        ):
            first_prerequisite = len(self.prerequisite_pages)
            previous_pages, self.pages = self.pages, []
            task = {k: v for k, v in origin.items() if k != "stop"}
            previous_stop, previous_ready = self.stop, self.ready
            self.stop, self.ready = origin.get("stop"), 0
            self.in_before += 1  # played for its outcome, in full (as start_state)

            try:
                status = self.run_route(task)
                if self.stop or evaluation.is_failure_status(status, self.english):
                    raise _BattleStartUnavailable(
                        f"origin {task['key']}: setup did not complete ({status})"
                    )
            except PlaybackStopped:
                if self.stop != "battle" or self.ready < 30:
                    raise
            finally:
                self.stop, self.ready = previous_stop, previous_ready
                self.in_before -= 1
                origin_pages = list(self.pages)
                self.pages = previous_pages + origin_pages

            self.cache[key] = self.emu.save_state()
            self.cache_spawns[key] = dict(self.spawns)
            self.cache_spawn_choices[key] = {
                slot: list(answers) for slot, answers in self.spawn_choices.items()
            }

            # Keep only captures produced while preparing this origin, including its
            # nested prerequisites. Later callers inherit that evidence with the cached state.
            self.cache_pages[key] = (
                origin_pages + self.prerequisite_pages[first_prerequisite:]
            )

        self.prerequisite_pages.extend(self.cache_pages.get(key, []))
        return self.cache[key]

    def get_opponent_health_fixture(self) -> int | None:
        """
        Return the enabled native enemy-health fixture, or None.

        story_enemy_hp is applied once at battle entry. It is not returned here,
        because the shared controller uses a nonzero result for continuous battle
        fixtures as well as health validation.

        Returns:
            int | None: Enabled enemy-HP fixture for this task, or None when no
                opponent-health override is active.

        Raises:
            ValueError: If a configured health value is a boolean, noninteger or outside
                the unsigned 16-bit range.
        """
        super().get_opponent_health_fixture()
        return (self.task or {}).get("native_enemy_health_fixture") or None

    def run_route(self, task: dict[str, Any]) -> str:
        """
        Dispatch battle/caption tasks or shared field routes.

        Args:
            task (dict[str, Any]): Route to dispatch; battle tasks with field_position use
                scene interaction. Their optional origin is resolved and loaded first,
                including ref-based origins. Other battle tasks use start or origin;
                caption tasks use their recorded start/state lists. The definition
                is not modified. Emulator state, task settings and captures change.

        Returns:
            str: Status from the chosen playback path. Battle and caption paths
                return execution status without a complete target/page verdict; callers
                must evaluate the retained captures against task requirements.

        Raises:
            ValueError: If a supported task, fixture, state, text or display field is invalid.
            KeyError: If a required field is missing or a text command is unsupported.
            TypeError: If an origin contains values JSON cannot serialize.
            RuntimeError: If native setup, emulator state save/restore or a required
                frame snapshot fails, or a requested maze destination is disconnected.
            PlaybackStopped: If playback reaches a configured stop not handled by
                the selected route helper.
            StopIteration: If a named start is absent or no script actor can be allocated.
            IndexError: If text verification reads beyond the ROM.
            AssertionError: If required controller setup or a frame is unavailable.
            OSError: If sources cannot be read or screenshots cannot be saved.
        """
        _validate_task(task)
        if task["route"] == "battle" and not task.get("field_position"):
            return self.play_battle_route(
                task
            )  # e.g. a battle that a later scene needs ('before')

        if task["route"] == "caption":
            return self.play_caption_route(task)

        if (
            task["route"] == "battle"
            and task.get("field_position")
            and task.get("origin")
        ):
            # Origins may be complete prerequisite routes or references to named
            # starts. Resolve either form before preparing the field interaction.
            self.emu.load_state(self.load_origin_state(task["origin"]))
            task = dict(task, route="dialogue-context", chained=True)
        return super().run_route(task)

    def play_battle_route(self, task: dict[str, Any]) -> str:
        """
        Replay a native battle with its declared display fixtures.

        Args:
            task (dict[str, Any]): Battle definition with start or origin and
                controller_steps; optional inventory, health and quest fixtures
                establish the encounter. origin takes precedence over start.
                Each controller step contains nonnegative integer frames and named
                buttons. Story battles pause the submitted-frame sequence during
                dialogue; other sequences follow elapsed emulator frames. Optional
                consumable_fixture requires maximum player HP of at least two.
                The definition is unchanged; emulator memory and captures change.

        Returns:
            str: Combat dialogue execution status. An idle return does not establish
                that every required target/page was captured or verified; the caller
                evaluates pages and prerequisite_pages separately.

        Raises:
            ValueError: If an enemy-health fixture is a boolean, noninteger or
                outside 1 through the native opponent maximum. Integer subclasses
                are accepted. Also raised for invalid task, state, text or display
                data, or a consumable fixture whose maximum player HP is below two.
            KeyError: If controller_steps or the chosen start/origin field is
                missing, or verified text contains an unsupported command.
            TypeError: If an origin contains values JSON cannot serialize.
            RuntimeError: If native setup, state save/restore or a required frame fails.
            PlaybackStopped: If playback reaches a separately configured stop;
                battle input steps and story override are disabled on exit.
            StopIteration: If a named start is absent or a prerequisite cannot
                allocate a script actor.
            IndexError: If text verification reads beyond the ROM.
            AssertionError: If required controller setup or a frame is unavailable.
            OSError: If sources cannot be read or screenshots cannot be saved.
        """
        _validate_task(task)
        steps = task["controller_steps"]
        state = (
            self.load_origin_state(task["origin"])
            if task.get("origin")
            else self.load_start_state(task["start"])
        )
        self.pages = []
        self.emu.load_state(state)
        self.targets_left = None  # building a start plays a route, which sets its own

        # Scripts the game spawned while the start was reached are still running: their pages are
        # theirs.
        origin = task.get("origin")
        key = (
            task["start"]
            if not origin
            else origin["ref"]
            if "ref" in origin
            else json.dumps(origin, sort_keys=True)
        )
        self.spawns = dict(self.cache_spawns.get(key, {}))
        self.spawn_choices = {
            slot: list(answers)
            for slot, answers in self.cache_spawn_choices.get(key, {}).items()
        }
        if self.early_stop and not self.in_before and task.get("expect"):
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

        self.reset_battle_observation(task)
        self.clear_log()
        self.set_flags(task.get("quest_flags", []))
        w = self.W

        if task.get("full_inventory"):
            self.poke(0x1AE9, [80] * 98)  # every item at the original limit

        if task.get("consumable_fixture"):
            # Items 85-97 are the consumables; one each, legal damage and the four afflictions the
            # item-eligibility routines ($83:9A7F-9AA0) inspect.
            maximum = w[0x1896] | w[0x1897] << 8
            if maximum < 2:
                raise ValueError(
                    "Consumable setup requires maximum player HP of at least two"
                )
            self.poke(0x189A, (maximum - 1).to_bytes(2, "little"))
            self.poke(0x1AE9 + 85, [1] * 13)
            self.poke(0x186B, [w[0x186B] | 0xF0])

        health = task.get("enemy_health_fixture", task.get("story_enemy_hp"))
        if health is not None:
            maximum = w[0x1898] | w[0x1899] << 8
            if not is_integer(health) or not 1 <= health <= maximum:
                raise ValueError("Enemy health fixture outside the original maximum")

            self.poke(0x189C, health.to_bytes(2, "little"))

        if RouteRunner.is_busy(self):
            self.advance = 4

        self.story, self.phased, self.submitted = (
            task.get("story_enemy_hp") is not None,
            False,
            0,
        )
        self.steps, self.origin = steps, self.frame
        try:
            status = self.play_dialogue(
                "combat",
                self.count_sequence_frames() + 12000,
                plan=task.get("choice_plan", []),
            )["status"]
        finally:
            self.steps = None
            self.story = False

        return status

    def play_caption_route(self, task: dict[str, Any]) -> str:
        """
        Try calibrated native battle starts for a direct caption check.

        The helper replays the recorded script with its original parent slot so the
        script opens its own window. It tries starts and states in declared order.
        A native endpoint failure marks that start unavailable for this runner;
        other runtime errors fail the attempt without excluding the start from later
        caption tasks on this runner.

        Args:
            task (dict[str, Any]): Caption definition with ordered starts, text entry ID and
                script/parent fields. starts contains names or origin dictionaries.
                states is None or a list of 146/677-byte hexadecimal memory contexts;
                None or an empty list tries the unmodified start once. A None list
                element also selects that unmodified start. All state payloads are
                validated before playback. The definition is not modified.

        Returns:
            str: First nonfailed attempt status whose captured target pages all
                verify; otherwise the last attempted status, or target_not_reached if
                no attempt ran. A last idle status can still have no target capture,
                so callers must evaluate required evidence separately.

        Raises:
            ValueError: If caption fields or recorded states are invalid, or
                verification rejects text or display data.
            KeyError: If a required caption field is missing or a text command is unsupported.
            TypeError: If a start dictionary contains values JSON cannot serialize.
            StopIteration: If a named start is absent or no script actor can be allocated.
            IndexError: If text verification reads beyond the ROM.
            AssertionError: If a required frame or controller setup is unavailable.
            OSError: If sources cannot be read or screenshots cannot be saved.
        """
        _parse_caption_definition(task)
        starts = task["starts"]
        states = task.get("states")
        if not isinstance(starts, list) or any(
            not isinstance(start, (str, dict)) for start in cast(list[object], starts)
        ):
            raise ValueError("Caption starts must be a list of names or origin objects")
        if states is not None:
            if not isinstance(states, list):
                raise ValueError(
                    "Caption states must be a list of recorded memory contexts"
                )
            for state in cast(list[Any], states):
                _decode_caption_state(state)
        start_options = cast(list[str | dict[str, Any]], starts)
        state_options = cast(list[str | None], states or [None])
        # Try calibrated contexts in order; skip starts already known to fail
        # without reusing captures produced by those failed starts.
        status = "target_not_reached"
        for start in start_options:
            name = json.dumps(start, sort_keys=True)
            if name in self.unreached:
                continue

            for state in state_options:
                try:
                    status = self.check_caption_in_state(task, start, state)
                except _BattleStartUnavailable as error:
                    status, self.pages = f"error: {error}", []
                    self.unreached.add(name)
                    break
                except (
                    RuntimeError,
                    PlaybackStopped,
                ) as error:
                    status, self.pages = f"error: {error}", []
                    continue

                # Only a verified capture of the requested caption completes this attempt;
                # other text shown by a valid battle context cannot satisfy the caption target.
                mine = [p for p in self.pages if p.get("entry") == task["text"]]
                if (
                    mine
                    and all(p.get("verified") for p in mine)
                    and not evaluation.is_failure_status(status, self.english, task)
                ):
                    return status

        return status

    def check_caption_in_state(
        self, task: dict[str, Any], start: str | dict[str, Any], state: str | None
    ) -> str:
        """
        Check one directly displayed caption in a native battle state.

        Args:
            task (dict[str, Any]): Caption definition containing hexadecimal script address,
                parent actor slot 0-36, uppercase hexadecimal text entry ID and
                optional menu-answer plan. Addresses contain six hexadecimal
                digits. The definition is not modified.
            start (str | dict[str, Any]): Native battle-start descriptor, or an origin
                chain to replay. Its cached script-spawn records and menu answers
                are restored together with emulator memory before invocation.
            state (str | None): Hexadecimal battle-state blob: WRAM
                0x1820-0x18AF and 0x16BC-0x16BD, optionally followed by SRAM
                story/actor flags.
                None leaves the loaded state unchanged; empty text is invalid.

        Returns:
            str: Dialogue execution status, with pages filtered to the requested
                entry and marked as direct battle displays, including partial
                output when playback raises. Emulator memory remains changed;
                battle input is disabled and the prior capture filter is restored. The direct label
                does not establish native calibration or complete page coverage.

        Raises:
            ValueError: If a caption field or named start is invalid, state does not
                encode 146 WRAM bytes optionally followed by 531 SRAM bytes, or
                verification rejects text, font, frame or VRAM data.
            RuntimeError: If the requested native start cannot be reached.
                Also raised if emulator state save/restore or a required frame fails.
            PlaybackStopped: If playback reaches a separately configured stop.
            KeyError: If a required caption field is missing or a text command is unsupported.
            TypeError: If a start dictionary contains values JSON cannot serialize.
            StopIteration: If a named start is absent or no script actor can be allocated.
            IndexError: If text verification reads beyond the ROM.
            AssertionError: If required controller setup or a frame is unavailable.
            OSError: If sources cannot be read or screenshots cannot be saved.
        """
        script, text, parent = _parse_caption_definition(task)
        block = _decode_caption_state(state)
        self.emu.load_state(
            self.load_origin_state(start)
            if isinstance(start, dict)
            else self.load_start_state(start)
        )
        key = (
            start.get("ref", json.dumps(start, sort_keys=True))
            if isinstance(start, dict)
            else start
        )
        self.spawns = dict(self.cache_spawns.get(key, {}))
        self.spawn_choices = {
            slot: list(answers)
            for slot, answers in self.cache_spawn_choices.get(key, {}).items()
        }
        self.targets_left = None  # building the start plays a route, which sets its own
        self.pages = []
        self.reset_battle_observation(task)
        self.story, self.phased, self.submitted = False, False, 0
        self.clear_log()

        if block is not None:
            # Recorded battle context from script spawn or text display: branch inputs at $1820-$18AF
            # and $16BC-$16BD, and when shown also the story flags and actor sync values its
            # branches and waits read (ops 30-35, 89).
            self.poke(0x1820, block[:0x90])
            self.poke(0x16BC, block[0x90:0x92])

            if len(block) > 0x92:  # story flags and actor sync values, $70:0000-$0212
                self.poke(0x000, block[0x92:], region=0)

        self.spawn_script(script, parent)
        previous_steps, previous_filter = self.steps, self.photograph_only
        self.steps, self.origin, self.photograph_only = [], self.frame, text
        try:
            # Replay the recorded menu answers to select the same script branch.
            status = self.play_dialogue("combat", 3000, plan=task.get("plan"))["status"]
        finally:
            self.steps, self.photograph_only = previous_steps, previous_filter
            # Background battle scripts are not part of the requested caption.
            # Filter and classify partial captures even when playback raises.
            self.pages = [p for p in self.pages if p.get("entry") == text]
            for page in self.pages:
                page["direct"] = (
                    "caption invoked by its recorded script in a battle context"
                )
                page["battle_start"], page["battle_state"] = start, state

        return status

    def play(self, task: dict[str, Any]) -> str:
        """
        Replay a battle/arena task and enforce its declared targets.

        Args:
            task (dict[str, Any]): Battle or arena route with target/targets; origin supplies
                battle prerequisites and forced labels diagnostic captures, even
                when playback raises. The caller clears the preceding task's pages
                and prerequisite_pages first. The definition is not modified.

        Returns:
            str: Execution status, changing idle to targets_incomplete or
                target_not_reached when declared target entries were not captured.
                Pixel verification and complete page requirements are graded separately.
                Captures from retained prerequisites count toward entry presence.

        Raises:
            ValueError: If task, fixture, state, text or display data are invalid.
            KeyError: If a required task field is missing or a text command is unsupported.
            TypeError: If an origin contains values JSON cannot serialize.
            RuntimeError: If native setup, state save/restore or a required frame fails.
            PlaybackStopped: If playback reaches an unhandled configured stop;
                diagnostic labels are retained on partial captures.
            StopIteration: If a named start is absent or no script actor can be allocated.
            IndexError: If text verification reads beyond the ROM.
            AssertionError: If required controller setup or a frame is unavailable.
            OSError: If sources cannot be read or screenshots cannot be saved.
        """
        _validate_task(task)
        try:
            status = self.run_route(task)
        finally:
            if task.get("forced"):
                for page in self.pages:
                    page["forced"] = task["forced"]

        targets = set(
            task.get("targets") or ([task["target"]] if task.get("target") else [])
        )
        seen = {p.get("entry") for p in self.pages + self.prerequisite_pages}
        if status == "idle" and targets - seen:
            status = "targets_incomplete" if targets & seen else "target_not_reached"

        return status
