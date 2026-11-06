"""
ScreenRunner loads a named game state, runs a registered controller program, and checks captured interface text against glyph traces and screen pixels.

ScreenRunner reaches a named route, power-on or saved-reset start, then gives its ScreenActions
interface to a callback registered in screens/programs. The callback submits ordinary controller
input and declared fixtures; the runner owns emulator loading, cached starts and capture
evaluation.

Each capture matches traced terminators to translated entries, compares literal glyphs and
checks every expected pixel through the current VRAM tilemaps. Repeated menu draws retain the
best capture for each page. Peripheral-driver fixtures stay labelled separately from ordinary
native playback.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence, cast

from core.validation.validate_integers import is_integer

import ctypes
import time
from pathlib import Path

from core.rom.convert_addresses import convert_snes_to_file
from core.text.read_compiled_text import decode_literal_glyphs
from emulator.validate_controller_input import collect_button_ids, validate_frame_count

from verification.evidence import check_screen_pixels as screen_ink
from verification.evidence import evaluate_results as evaluation
from verification.evidence import instrument_rom as instrument
from verification.evidence.index_gameplay_text import index_page_terminators
from verification.evidence.read_text_layers import capture_text_layers
from verification.game.control_gameplay import BTN
from verification.routes.run_routes import RouteRunner
from verification.screens import emulate_turbo_file as turbo_stub
from verification.screens.register_screen_programs import SCRIPTS
from verification.tasks.load_tasks import load_screen_starts


class ScreenStartReached(Exception):
    """
    The emulator reached the stopping condition specified by the starting-state definition.
    """


class ScreenRunner(RouteRunner):
    """
    This controller reaches registered starts and captures screen-program output.

    Call start() before programs that depend on the roaming snapshot. Start states and timings
    are cached separately from each program's done-page set. A verified entry/page is not
    rechecked again during that program. The runner owns its emulator and live memory views;
    close it after use.
    """

    early_stop = False  # starting-state definitions must reach their stop point
    fast_text = False

    def __init__(
        self,
        rom: bytes | bytearray,
        scratch: Path,
        english: bool = True,
        shots: Path | None = None,
    ) -> None:
        """
        Initialize accumulating traces and native screen-start indexes.

        Keeps the power-on state and accumulates glyph traces for menu captures.

        Boolean options require bool values; numeric zero and one are rejected.

        Args:
            rom (bytes | bytearray): English release or Japanese original to
                instrument for screen checks. Retained by reference and not modified;
                the caller must keep this buffer unchanged while the session is open.
            scratch (Path): Output path for the ROM file with tracing code
                added. The parent directory must already exist.
            english (bool, optional): True for the English ROM; False for the
                Japanese original. Defaults to True.
            shots (Path | None, optional): Screenshot output directory;
                None disables saving images. Defaults to None.

        Raises:
            TypeError: If rom is not bytes or bytearray, or english is not a boolean.
            OSError: If source files cannot be read or the instrumented ROM file
                cannot be written or read back.
            ValueError: If instrumentation, translation data, screen-start JSON or
                required emulator memory is invalid.
            RuntimeError: If the core cannot load the ROM or serialize its state.
            KeyError: If indexed text contains an unsupported command.
            IndexError: If text indexing reads beyond the ROM.
        """
        if type(english) is not bool:
            raise TypeError("Boolean options require bool values: english")
        super().__init__(rom, scratch, english=english, shots=shots, accumulate=True)
        try:
            # Retain the instrumented file's contents for restoration after fixtures.
            self.image = Path(scratch).read_bytes()
            self.power_on = self.emu.save_state()
            self.terms = index_page_terminators(rom)
            self.screen_starts = load_screen_starts()
        except BaseException:
            self.close()
            raise
        self.done: set[tuple[str, int]] = (
            set()
        )  # (entry, page) verified by the current script
        self.cache = {}
        # Detect recursive after-start dependencies before they exhaust Python's stack.
        self.building_starts: set[str] = set()
        self.built: dict[
            str, float
        ] = {}  # start name: seconds its build took (load_tasks.py's scheduling)
        self.quick_starts = (
            False  # verify_gameplay.py's quick run: starts built without script waits
        )
        self.menu_entries: set[str] = set()
        self.stop_rule: str | None = None
        self.relocated = False

    def run(self, n: int = 1, held: Iterable[int] = ()) -> None:
        """
        Advance shared gameplay until a requested screen boundary.

        Args:
            n (int, optional): Emulated frames to advance.
                Defaults to 1.
            held (Iterable[int], optional): Explicit libretro button IDs passed to
                shared gameplay control. Empty permits eligible automatic input.
                Defaults to ().

        Raises:
            ValueError: If the frame count is not a nonnegative integer or held input is
                unsupported.
            ScreenStartReached: If playback reaches the native screen boundary requested
                for program control.
        """
        validate_frame_count(n)
        buttons = collect_button_ids(held)
        for _ in range(n):
            super().run(1, buttons)

            if self.stop_rule == "mission_results" and self.get_current_room() == (
                23,
                7,
            ):
                raise ScreenStartReached

            if self.stop_rule == "runner_naming" and self.get_current_room() == (22, 4):
                raise ScreenStartReached  # the naming screen draws under the script's trace

    def relocate_player(
        self, x: int, y: int, room: Sequence[int] | None = None
    ) -> None:
        """
        Relocate through the native loader and mark this screen start as positioned.

        Args:
            x (int): Destination X coordinate in native field pixels.
            y (int): Destination Y coordinate in native field pixels.
            room (Sequence[int] | None, optional): Stage/variant pair; None uses
                the active task's room.
                Defaults to None.
        """
        super().relocate_player(x, y, room)
        self.relocated = True

    def press(self, button: str, frames: int = 4, after: int = 10) -> None:
        """
        Submit a named button, or stop before the requested tile interaction's A press.

        Args:
            button (str): One button name from BTN.
            frames (int, optional): Emulated frames to hold the button.
                Defaults to 4.
            after (int, optional): Frames after the explicit press; shared
                gameplay control may supply automatic input during this interval.
                Defaults to 10.

        Raises:
            ValueError: If a hold or release duration is not a nonnegative integer.
                Booleans are rejected; integer subclasses are accepted. Validation
                occurs before a screen-stop exception can replace the button press.
            KeyError: If button is not a supported joypad name.
            ScreenStartReached: If playback reaches the native screen boundary requested
                for program control.
        """
        validate_frame_count(frames)
        validate_frame_count(after)
        if button not in BTN:
            raise KeyError(button)
        if self.stop_rule == "before_tile" and self.relocated and button == "a":
            raise ScreenStartReached  # after the approach step, before the tile's A

        super().press(button, frames, after)

    def begin(self, task: dict[str, Any]) -> None:
        """
        Clear screen-start approach tracking before shared route setup.

        Args:
            task (dict[str, Any]): Route setup passed to RouteRunner.begin(); screen
                tile-approach tracking is cleared first.
        """
        self.relocated = False
        super().begin(task)

    # Reusable game states reached before screen-program input begins.
    def load_start_state(self, name: str) -> bytes:
        """
        Return a cached screen start, or build and time a new one.

        Args:
            name (str): Built-in roam or power-on state, or a screen-start key
                from screens/data/screen_starts.json.

        Returns:
            bytes: Serialized state at the configured native screen boundary. A cached
                hit only returns bytes; the caller must load them into the emulator.
                A new build runs setup and records its elapsed time.

        Raises:
            ValueError: If name is not a nonempty string or its after-start
                dependencies contain a cycle.
            KeyError: If name is not a built-in or registered start.
            RuntimeError: If the native setup cannot reach its configured boundary.
        """
        if not isinstance(cast(object, name), str) or not name:
            raise ValueError("Screen start must be a nonempty string")
        if name in self.cache:
            return self.cache[name]
        if name in self.building_starts:
            raise ValueError(f"Screen starts contain a dependency cycle at {name}")

        clock = time.monotonic()
        skip, self.skip_waits = (
            self.skip_waits,
            self.quick_starts,
        )  # quick runs: no script waits (outside text)

        # Wait skipping applies only while preparing the reusable start. Restore
        # the caller's policy before the actual screen program is graded.
        self.building_starts.add(name)
        try:
            state = self.build_start(name)
        finally:
            self.skip_waits = skip
            self.building_starts.remove(name)

        self.built[name] = round(time.monotonic() - clock, 1)
        return state

    def check_page(self) -> dict[str, Any]:
        """
        Capture dialogue or stop at a settled manual-menu boundary.

        Returns:
            dict[str, Any]: Captured dialogue page, including its entry and page number when
                identified.

        Raises:
            ScreenStartReached: If playback reaches the native screen boundary requested
                for program control.
        """
        page = super().check_page()
        if self.stop_rule == "menu" and page.get("entry") in self.menu_entries:
            self.run(120)  # the menu settles, waiting for the script's buttons
            raise ScreenStartReached

        return page

    def build_start(self, name: str) -> bytes:
        """
        Build and cache a serialized native screen-start state.

        Args:
            name (str): Built-in roam or power-on state, or a screen-start key
                from screens/data/screen_starts.json.

        Returns:
            bytes: Serialized state at the configured native screen boundary.

        Raises:
            ValueError: If name is not a nonempty string or nested starts form a cycle.
            KeyError: If a named start or required task field is missing.
            RuntimeError: If the configured native screen start cannot be reached.
            AssertionError: If start() has not established the roaming snapshot.
        """
        if not isinstance(cast(object, name), str) or not name:
            raise ValueError("Screen start must be a nonempty string")
        if name == "roam":
            state = self.base
        elif name == "power-on":
            state = self.power_on
        elif self.screen_starts[name].get("after"):
            # Save to slot 1 at the device, then reset the console; SRAM keeps the save.
            self.emu.load_state(
                self.load_start_state(self.screen_starts[name]["after"])
            )
            self.press("a", 4, 90)
            self.press("a", 4, 6)
            self.run(240)
            self.emu.lib.retro_reset()
            state = self.emu.save_state()
        else:
            spec = self.screen_starts[name]
            manual = spec["task"].get("manual_menu_entries", [])
            saved = {
                i: self.entries.pop(i) for i in manual if i in self.entries
            }  # menus left to the steps

            # A start that leaves a menu to the scripts ends once that menu is up (check_page), not
            # when its frame budget runs out.
            self.stop_rule, self.menu_entries = (
                spec.get("stop") or ("menu" if manual else None),
                set(manual),
            )
            try:
                self.pages = []
                task = (
                    dict(spec["task"], before=spec["before"])
                    if spec.get("before")
                    else spec["task"]
                )
                status = self.run_route(task)
                if self.stop_rule or evaluation.is_failure_status(status, self.english):
                    raise RuntimeError(
                        f"start {name}: stop point not reached ({status})"
                    )
            except ScreenStartReached:
                pass
            finally:
                self.stop_rule = None
                self.entries.update(saved)

            state = self.emu.save_state()

        if state is None:
            raise AssertionError("Screen start has no serialized emulator state")

        self.cache[name] = state
        return state

    # Text identification and pixel checks for captured frames.
    def read_records(self, memory: bytes | None = None) -> list[dict[str, Any]]:
        """
        Return terminator-read records since clear, with translated IDs when known.

        Args:
            memory (bytes | None, optional): Frozen frame SRAM for capture evidence;
                None reads the live trace used to control gameplay.
                Defaults to None.

        Returns:
            list[dict[str, Any]]: Decoded terminator-read records annotated with translated
                entry IDs where the current terminator mapping identifies them. Each
                record adds ids (zero or one entry ID) and hit (entry/page/start tuple
                or None). WRAM terminators are treated as dynamic inserts and receive
                no translated hit.
        """
        out: list[dict[str, Any]] = []
        for rec in instrument.read_terminator_log(self.S if memory is None else memory):
            # WRAM addresses identify dynamic inserted text, not a literal ROM page.
            # Only a mapped ROM terminator establishes the owning entry and page.
            hit = (
                None
                if rec["term"] >> 16 in (0x7E, 0x7F)
                else self.terms.get(convert_snes_to_file(rec["term"]))
            )
            out.append(dict(rec, ids=[hit[0]] if hit else [], hit=hit))

        return out

    def check_capture(
        self, script: str, stem: str, context: str
    ) -> list[dict[str, Any]]:
        """
        Check newly relevant traced text records against the current frame.

        Skip pages already verified in this program. English also skips records whose literal
        glyphs began before the last trace clear. Successful English records update done; the
        method returns records but does not save photographs or append them to
        ScreenActions.results. Japanese records identify pages without claiming English
        literal/pixel verification.

        Terminator offsets must identify complete records present in the captured
        glyph log. A full log cannot distinguish its final recorded draw from later
        unrecorded draws, so records at that limit receive no verification capture.

        Args:
            script (str): Screen-program key stored as the capture's screen:
                phase, linking it to the registered program definition.
            stem (str): Capture label stored in the result's capture field.
                Screenshot filenames are assigned separately by photograph().
            context (str): Human-readable description attached to captures from
                this screen state.

        Returns:
            list[dict[str, Any]]: Captured entry-page records with screen-program
                phases and, for English captures, literal and pixel checks.

        Raises:
            ValueError: If a trace, video frame, VRAM buffer, display-register
                snapshot or text layout is invalid.
            RuntimeError: If the emulator has no usable snapshot for this frame.
            KeyError: If script or a captured entry is not registered, or compiled
                text contains an unsupported command.
            IndexError: If compiled text reads beyond the supplied ROM.
            AssertionError: If pixel checking has no captured RGB frame or VRAM.
        """
        frame_memory = self.emu.read_frame_memory(0) if self.english else None
        recs = self.read_records(frame_memory)

        # Skip pages this program has already verified. If no unchecked records
        # remain, no screenshot or memory snapshot is copied for this attempt.
        recs = [
            r
            for r in recs
            if not (r["hit"] and (r["hit"][0], r["hit"][1]) in self.done)
        ]
        if not any(r["hit"] for r in recs):
            return []

        rows = self.emu.read_rgb_rows() if self.english else None
        vram = self.emu.read_frame_memory(3) if self.english else None
        layers = (
            capture_text_layers(
                self.release,
                frame_memory,
                self.emu.read_frame_memory(2),
                self.emu.read_frame_registers(),
                len(rows),
            )
            if frame_memory is not None and rows is not None
            else []
        )
        glyphs = (
            instrument.read_glyph_log(frame_memory) if frame_memory is not None else []
        )
        out: list[dict[str, Any]] = []
        for rec in recs:
            if self.english:
                # Never substitute an earlier log end for missing draw evidence.
                # At capacity, subsequent native draws no longer advance this counter.
                draw_end = rec["draw_end"]
                if draw_end % 12 or not 0 <= draw_end <= len(glyphs) * 12:
                    raise ValueError(
                        "Terminator glyph offset is outside complete captured records"
                    )
                if draw_end == instrument.LOG_SIZE:
                    continue
            end = rec["draw_end"] // 12
            if rec["hit"]:
                entry, page, start = rec["hit"]
                result = dict(
                    entry=entry,
                    page=page,
                    phase=f"screen:{script}",
                    context=context,
                    capture=stem,
                    frame=self.frame,
                    room=list(self.get_current_room()),
                )
                if self.english:
                    wanted, _ = decode_literal_glyphs(self.release, start)

                    # Count backward from this record's end to select its ROM-string
                    # glyphs. WRAM name/number insertions remain in the trace but do
                    # not count toward the literal-glyph total.
                    k, need = end, len(wanted)
                    while k > 0 and need:
                        k -= 1
                        need -= glyphs[k]["bank"] not in (0x7E, 0x7F)

                    if (
                        need
                    ):  # the page began before the last clear: not all of it is logged
                        continue

                    mine = glyphs[k:end]

                    # Compare literal ROM glyphs independently of pixels. Runtime insertions
                    # from WRAM are retained for ink checking but excluded from this literal list.
                    drawn = [
                        g["font_offset"] // 22
                        for g in mine
                        if g["bank"] not in (0x7E, 0x7F)
                    ]

                    if rows is None or vram is None:
                        raise AssertionError(
                            "Screen verification requires a captured frame and VRAM"
                        )

                    pixels = screen_ink.check_screen_pixels(
                        self.release,
                        mine,
                        rows,
                        vram,
                        layers=layers,
                        box=SCRIPTS[script].get("text_boxes", {}).get(entry),
                        label_starts=[
                            column * 12
                            for columns in self.entries[entry].get("menu_columns", [])
                            for column in columns
                        ],
                    )
                    result.update(
                        literal_match=wanted == drawn,
                        ink=pixels["ink"],
                        missing_ink=pixels["missing_screen_ink"],
                        outside_ink=pixels["outside_ink"],
                        pixel_placement=pixels["placement"],
                        pixel_errors={
                            key: pixels[key]
                            for key in (
                                "unexpected_screen_ink",
                                "missing_vram_ink",
                                "unexpected_vram_ink",
                                "clipped_ink",
                                "incomplete_copies",
                            )
                        },
                        pixel_method="recorded-display-registers",
                    )
                    result["verified"] = (
                        bool(mine) and result["literal_match"] and pixels["passed"]
                    )
                    if result["verified"]:
                        self.done.add((entry, page))

                initialization = (
                    SCRIPTS[script].get("initialization_only", {}).get(entry)
                )
                if initialization and not result.get("verified"):
                    result["credit"] = False
                    result["initialization_only"] = initialization
                out.append(result)

        return out

    def load_rom(self, image: bytes) -> None:
        """
        Replace the loaded ROM and reacquire the session's live memory views.

        Reload without advancing emulation. Save states restore RAM for the loaded cartridge;
        they do not replace its ROM image.

        Args:
            image (bytes): Complete ROM image to load; native memory views are
                reacquired after reload.

        Raises:
            RuntimeError: If the core rejects the replacement ROM image.
            ValueError: If required WRAM or SRAM is unavailable after reload.
        """
        self.emu.reload_rom(image)

        # Save states preserve RAM contents, not the allocation addresses of
        # a reloaded cartridge; do not reuse the previous live ctypes views.
        self.W, self.S = self.get_memory_view(2), self.get_memory_view(0)
        self.T = (ctypes.c_uint8 * 0x80).from_address(
            ctypes.addressof(self.S) + instrument.TRACE
        )

    def play_script(self, name: str) -> list[dict[str, Any]]:
        """
        Restore a registered start and execute one screen-program callback.

        Clear the trace and per-program done set, then collect callback captures and retain the
        best record per entry/page. No captures means an empty result, not a successful coverage
        verdict. A device or forced-display fixture temporarily replaces the instrumented ROM
        and restores the normal image in finally after callback execution; restoring ROM bytes
        invalidates and reacquires live memory views.

        Successful playback stores the selected records on pages. A failed callback
        leaves its partial captures there, with forced or peripheral labels retained.
        Failure while installing a fixture also attempts to restore the ordinary
        instrumented ROM before propagating the error.

        Args:
            name (str): Registered screen-program key in SCRIPTS.

        Returns:
            list[dict[str, Any]]: Best capture per entry/page, with English pixel
                checks, forced-display labels and peripheral labels for Turbo File stand-ins. Each
                record includes total and verified capture-attempt counts.

        Raises:
            KeyError: If a screen program or required start is not registered.
            ValueError: If a device fixture lacks statuses or contains invalid function
                IDs or reply bytes, a ROM patch lacks forced
                attribution, or a guarded patch does not match the source image.
            RuntimeError: If native start setup, ROM reload, or save-state handling
                fails.
            OSError: If setup sources cannot be read or a screenshot cannot be saved.
            IndexError: If captured text reads beyond the ROM.
            AssertionError: If a required start state, frame or VRAM is unavailable.
        """
        # Restore the registered start before program inputs run. Device
        # stand-ins receive explicit capture labels after playback below.
        script = SCRIPTS[name]
        stub = script.get("turbo_stub")
        patches = script.get("rom_patches", [])
        forced = script.get("forced")
        patched_image: bytes | None = None
        if patches:
            if not forced or stub:
                raise ValueError(
                    "ROM display patches require forced attribution without a device stub"
                )
            image = bytearray(self.image)
            for offset, expected, replacement in patches:
                if (
                    (not is_integer(offset))
                    or offset < 0
                    or not isinstance(cast(object, expected), bytes)
                    or not isinstance(cast(object, replacement), bytes)
                    or not expected
                    or len(expected) != len(replacement)
                    or offset + len(expected) > len(image)
                    or bytes(image[offset : offset + len(expected)]) != expected
                ):
                    raise ValueError(
                        "Forced-display ROM patch does not match its guarded instruction"
                    )
                image[offset : offset + len(expected)] = replacement
            patched_image = bytes(image)
        if stub:
            statuses = script.get("turbo_statuses")
            if statuses is None:
                raise ValueError("Turbo File stand-in requires device status responses")

            data = None

            if script.get(
                "turbo_file"
            ):  # the stub serves save slot 1 of the reset-with-save start
                self.emu.load_state(self.load_start_state("reset-with-save"))
                data = bytes(self.S[0x800:0x1000])

            patched_image = turbo_stub.patch_turbo_driver(self.image, statuses, data)

        start = self.load_start_state(script["state"])
        driver = ScreenActions(self, name)
        self.pages = driver.results
        try:
            if patched_image is not None:
                self.load_rom(patched_image)
            self.emu.load_state(start)
            self.task, self.entry_setup = {}, None
            self.done = set()  # (entry, page) verified by this script
            self.clear_trace()
            script["run"](driver)
        # The emulator loads the patched data; self.image retains the unmodified ROM data.
        # Reload the ordinary instrumented image even if playback fails.
        finally:
            # Failed callbacks still leave attributable partial captures for auditing.
            for page in driver.results:
                if forced:
                    page["forced"] = forced
                if stub:
                    page["peripheral"] = "Turbo File driver stub"
                    page["fixture"] = (
                        "Turbo File stand-in driver (emulate_turbo_file.py): the emulator has no Turbo File"
                    )
            if patched_image is not None:
                self.load_rom(self.image)

        # Several frames can capture one page. Retain its best pixel result
        # while counting all attempts, rather than counting repeated draws as more coverage.
        best: dict[tuple[str, int], dict[str, Any]] = {}
        for result in driver.results:
            key = (result["entry"], result["page"])
            old = best.get(key)
            result["captures"] = (old["captures"] if old else 0) + 1
            result["verified_captures"] = (
                old["verified_captures"] if old else 0
            ) + bool(result.get("verified"))
            if old is None or self.is_better_capture(result, old):
                best[key] = result
            else:
                old["captures"], old["verified_captures"] = (
                    result["captures"],
                    result["verified_captures"],
                )

        pages = list(best.values())
        self.pages = pages
        return pages

    @staticmethod
    def is_better_capture(new: dict[str, Any], old: dict[str, Any]) -> bool:
        """
        Prefer verified captures, then those with less missing ink.

        Args:
            new (dict[str, Any]): Candidate captured-page pixel-check record.
            old (dict[str, Any]): Previously retained captured-page record to compare
                against.

        Returns:
            bool: True if only the new record is verified, or if both records
                have the same verification verdict and the new one has fewer
                combined missing/outside ink pixels. Ties retain the old record.
        """
        if bool(new.get("verified")) != bool(old.get("verified")):
            return bool(new.get("verified"))

        return bool(
            (new.get("missing_ink", 0) + new.get("outside_ink", 0))
            < (old.get("missing_ink", 0) + old.get("outside_ink", 0))
        )


class ScreenActions:
    """
    Controller and capture interface passed to screen-program run callbacks.

    Screen programs submit input and append checked captures through this interface.
    ScreenRunner owns emulator loading, closing and fixture attribution.

    Attributes:
        r (ScreenRunner): Runner receiving all input, memory writes and
            captures.
        e (Emu): The runner's owned emulator, exposed for native memory snapshots.
        name (str): Program key prefixed to capture labels and used in phases.
        results (list[dict]): Capture records appended by capture(), consumed
            by the screen runner.
    """

    def __init__(self, runner: ScreenRunner, name: str) -> None:
        """
        Bind a screen runner and program name.

        Args:
            runner (ScreenRunner): Live runner to receive controller input,
                fixtures and capture requests.
            name (str): Registered screen-program key in SCRIPTS.
        """
        self.r, self.e, self.name = runner, runner.emu, name
        self.results: list[dict[str, Any]] = []

    def run(self, n: int, held: Iterable[str] = ()) -> None:
        """
        Advance the screen runner while holding named buttons.

        Args:
            n (int): Emulated frames to advance.
            held (Iterable[str], optional): Explicit button names held together
                for n frames. Empty permits the runner's eligible automatic input.
                Defaults to ().

        Raises:
            ValueError: If the frame count is not a nonnegative integer.
            KeyError: If held contains a button name not present in BTN.
            TypeError: If held is not iterable or contains an unhashable button name.
            ScreenStartReached: If emulation reaches an active setup stop boundary.
        """
        self.r.run(n, {BTN[b] for b in held})

    def press(self, b: str, hold: int = 4, after: int = 60) -> None:
        """
        Hold one named button, then resume normal runner input for after frames.

        Args:
            b (str): One button name from BTN, such as a, up or start.
            hold (int, optional): Emulated frames for which the button remains
                pressed.
                Defaults to 4.
            after (int, optional): Frames after the explicit press; the runner
                may supply automatic input during this interval.
                Defaults to 60.

        Raises:
            ValueError: If a hold or release duration is not a nonnegative integer.
            KeyError: If b is not a key in BTN.
        """
        validate_frame_count(hold)
        validate_frame_count(after)
        self.run(hold, (b,))
        self.run(after)

    def poke(
        self, address: int, data: bytes | Iterable[int], region: int = 2, why: str = ""
    ) -> None:
        """
        Write a declared fixture to a memory region.

        Args:
            address (int): Byte offset relative to the selected native memory
                region.
            data (bytes | Iterable[int]): Replacement bytes within the
                selected native memory region.
            region (int, optional): Libretro memory region ID: 0 is SRAM, 2 is WRAM
                and 3 is VRAM.
                Defaults to 2.
            why (str, optional): Fixture rationale accepted for callback readability;
                this argument is not stored. Result attribution comes from the
                program's registered fixture metadata.
                Defaults to ''.

        Raises:
            ValueError: If the write span or payload byte values are outside native
                bounds.
            TypeError: If data is not bytes or an iterable of byte values.
        """
        self.r.poke(address, data, region)

    def clear(self) -> None:
        """
        Clear traces while retaining the screen and accumulated results.
        """
        self.r.clear_trace()

    def read_records(self) -> list[dict[str, Any]]:
        """
        Return the runner's terminator-read records since the last clear.

        Returns:
            list[dict[str, Any]]: The runner's decoded terminator-read records since the last
                trace clear, including any identified translation IDs.
        """
        return self.r.read_records()

    def capture(self, stem: str, context: str, settle: int = 3) -> None:
        """
        Settle and append checked, labeled page captures.

        Advance the requested settling frames before checking the current frame, retaining the
        trace. Zero permits capture of transient text. Settling time is additional to waits in
        the screen program. All evaluated records are retained before screenshots
        are saved, so an image-writing failure does not discard those records.

        Args:
            stem (str): Capture label prefixed with the program name and stored
                in the result. It does not select the screenshot filename.
            context (str): Human-readable description attached to captures from
                this screen state.
            settle (int, optional): Nonnegative frames to advance before capture;
                defaults to three. Zero samples the current retained frame.

        Raises:
            ValueError: If settle is not a nonnegative integer.
                Also raised if capture verification rejects trace or display data.
            RuntimeError: If the emulator has no usable snapshot for this frame.
            KeyError: If a program, entry or compiled text command is unknown.
            IndexError: If captured text reads beyond the ROM.
            AssertionError: If a required frame or VRAM snapshot is unavailable.
            OSError: If a screenshot cannot be saved.
            ScreenStartReached: If settling reaches an active setup stop boundary.
        """
        validate_frame_count(settle)
        # Allow the last draw to settle before checking pixels, without clearing
        # the trace that identifies which text produced them.
        self.r.run(settle)
        stem = f"{self.name}-{stem}"
        captures = self.r.check_capture(self.name, stem, context)
        self.results.extend(captures)
        for result in captures:
            self.r.photograph(result)
