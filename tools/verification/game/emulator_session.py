"""
GameSession loads a ROM with tracing code into Snes9x and captures the text pages the game displays.

The session writes the instrumented ROM data to its scratch path, then loads that file into the
emulator. It provides live WRAM/SRAM views, identifies game text commands, captures their pages,
and optionally saves screenshots. Its close() method unloads the ROM and shuts down the
emulator. English checks compare traced glyphs and reconstructed font pixels with the actual
frame; Japanese playback records original text presence for comparison.

Closing the session or reloading its ROM invalidates live memory views. The uninstrumented
release bytes remain the reference for text and font decoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, cast

from core.validation.validate_integers import is_integer

if TYPE_CHECKING:
    import pathlib
import ctypes
import hashlib

from core.files.validate_binary_buffers import require_byte_buffer
from core.graphics.write_png import write_png
from core.rom.convert_addresses import convert_snes_to_file
from core.text.read_compiled_text import decode_literal_glyphs
from emulator.snes9x_core import Emu
from emulator.validate_controller_input import collect_button_ids, validate_frame_count
from verification.evidence import check_dialogue_pixels as ink
from verification.evidence import check_screen_pixels as screen_ink
from verification.evidence import instrument_rom as instrument
from verification.evidence.index_gameplay_text import (
    get_text_slot_ranges,
    index_item_names,
    index_page_starts,
)
from verification.evidence.locate_text_windows import get_capture_window
from verification.evidence.read_text_layers import capture_text_layers

# Libretro joypad IDs for advancing dialogue and starting the game.
A, START = 8, 3

# Process-local caches retain instrumented ROMs by image/options and the
# free-roaming state reached by each runner class. Pooled workers reuse them.
BUILT: dict[tuple[str, bool, bool], bytes] = {}
BOOTED: dict[tuple[Any, ...], Any] = {}


class GameSession:
    """
    This class manages Snes9x playback and text captures for either language's ROM.

    Attributes:
        release (bytes | bytearray): Caller-owned uninstrumented source ROM used
            to decode expected text/font data; retained without copying.
        emu (Emu): Owned native emulator; call close() when finished.
        W (ctypes.Array): Live WRAM byte view, invalid after unload or ROM
            reload.
        S (ctypes.Array): Live cartridge SRAM byte view containing game data
            and instrumentation.
        T (ctypes.Array): Live instrumentation counters within cartridge SRAM.
        pages (list[dict]): Capture records accumulated by the current
            playback.
        frame (int): Number of frames submitted through the session since
            construction.
    """

    def __init__(
        self,
        rom: bytes | bytearray,
        scratch: pathlib.Path,
        english: bool = True,
        shots: pathlib.Path | None = None,
        accumulate: bool = False,
    ) -> None:
        """
        The constructor writes a ROM file with tracing code to scratch and loads that file into
        Snes9x.

        The supplied rom buffer is retained for decoding expected text and fonts; it is not
        modified. Live emulator memory views become invalid when the session is closed or its
        ROM is replaced.

        Args:
            rom (bytes | bytearray): ROM data before tracing code is added. The source buffer is
                retained by reference for text and font decoding and must stay unchanged during
                the session.
            scratch (pathlib.Path): Output path for the ROM file with tracing code added. The
                parent directory must already exist.
            english (bool, optional): True for the English ROM; False for the Japanese original.
                Defaults to True.
            shots (pathlib.Path | None, optional): Screenshot output directory; None disables
                saving images.
            accumulate (bool, optional): Retain glyph-read history across text draws for
                interface-screen checks. Defaults to False.

        Raises:
            TypeError: If rom is not bytes or bytearray, or a language/log option is not a
                boolean.
            OSError: If instrumentation files or the scratch image cannot be written, Asar
                cannot be started, the native core cannot be loaded, or translation sources
                cannot be read.
            ValueError: If instrumentation, source metadata or text indexing rejects the image,
                or required native memory is unavailable.
            RuntimeError: If the emulator cannot load the instrumented image.
            IndexError: If text indexing encounters an entry, operand, token or jump target
                outside the supplied ROM.
            KeyError: If text indexing encounters an unsupported F8 command.
        """
        require_byte_buffer(rom, (bytes, bytearray))
        if type(english) is not bool or type(accumulate) is not bool:
            raise TypeError("english and accumulate must be booleans")
        self.release = rom
        self.english = english

        # Instrumentation depends on ROM bytes, language and glyph-log retention.
        # Do not reuse a scratch image prepared for another combination.
        key = (hashlib.sha1(rom).hexdigest(), english, accumulate)
        if key not in BUILT:
            BUILT[key] = instrument.build_instrumented_rom(
                rom, glyph_log=english, accumulate=accumulate
            )

        scratch.write_bytes(BUILT[key])
        self.emu = Emu(str(scratch))
        try:
            # Hardware-position pixel evidence is collected for the English renderer.
            self.emu.capture_registers = english
            self.W = self.get_memory_view(2)  # WRAM $7E0000-$7FFFFF, read in place
            self.S = self.get_memory_view(0)  # save RAM: story flags and instrumentation trace

            if len(self.S) < instrument.TRACE + 0x80:
                raise ValueError("Cartridge SRAM does not contain the instrumentation header")

            # T is a live view of the added 128-byte trace header inside SRAM,
            # not a copy. Its offsets are defined by the instrumentation hooks.
            self.T = (ctypes.c_uint8 * 0x80).from_address(ctypes.addressof(self.S) + instrument.TRACE)
            self.starts = index_page_starts(rom) if english else {}
            self.ranges = get_text_slot_ranges()
            self.shots = shots
            self.kept: dict[str, bool] = {}
            self.photograph_only: str | None = (
                None  # Entry ID limiting screenshots during caption playback.
            )
            self.items: dict[int, tuple[str, list[int]]] | None = None
            self.pages: list[dict[str, Any]] = []
            self.frame = 0
        except BaseException:
            # An indexing or memory-view failure must not leave a loaded core.
            self.close()
            raise

    def get_current_text_start(self) -> int:
        """
        Locate the release-ROM offset belonging to the native text command now running.

        Repair wrappers can redirect through a show-text command or scratch-WRAM stub rather
        than pointing straight into a translated slot. The lookup resolves those paths before
        capture identification uses the returned offset.

        Returns:
            int: Current text command start as an unheadered release-ROM file
                offset, or -1 when a WRAM command cannot be resolved to a ROM entry.
        """
        # Trace bytes $04-$06 hold the 24-bit start of the current text command.
        t = self.T

        # A text command can execute from WRAM. Resolve its initial text jump
        # before assigning that command to a ROM translation entry.
        if t[0x06] in (0x7E, 0x7F):  # a stub event in WRAM (forced displays)
            at = (t[0x06] - 0x7E) << 16 | t[0x05] << 8 | t[0x04]
            code, start = bytes(self.W[at : at + 5]), None
        else:
            start = convert_snes_to_file(t[0x04] | t[0x05] << 8 | t[0x06] << 16)
            code = self.release[start : start + 5] if self.find_entry_at(start)[0] is None else b""

        if len(code) == 5 and code[0] == 0xF8 and code[1] == 0x06:
            # RAM jump destinations cannot identify ROM text. Do not pass their
            # bank bytes through a converter that discards ROM mirror bits.
            if code[4] in (0x7E, 0x7F):
                return -1
            slot = convert_snes_to_file(code[2] | code[3] << 8 | code[4] << 16)
            if self.find_entry_at(slot)[0] is not None:
                return slot

        # A RAM address is not a mirrored ROM pointer. Unresolved commands
        # remain unidentified rather than acquiring an unrelated ROM entry.
        return start if start is not None else -1

    def find_entry_at(self, start: int) -> tuple[str, int] | tuple[None, None]:
        """
        Map a text file offset to (entry ID, page number), or (None, None).

        Japanese page numbers count observations of the original entry rather than English
        compiled pages.

        Integer inputs accept subclasses; booleans are rejected.

        Args:
            start (int): Text start as an unheadered file offset in the release
                ROM.

        Returns:
            tuple[str, int] | tuple[None, None]: Entry ID and one-based page number, or (None, None)
                when unidentified.

        Raises:
            ValueError: If start is a boolean or noninteger.
        """
        if not is_integer(start):
            raise ValueError("start must be an integer, not a boolean")
        # English has explicit relocated page starts. Japanese keeps original slot
        # ranges, so its page number follows captures already recorded for that entry.
        if self.english:
            return self.starts.get(start, (None, None))

        for lo, hi, entry in self.ranges:
            if lo <= start < hi:
                return entry, 1 + sum(p["entry"] == entry for p in self.pages)

        return None, None

    def photograph(
        self, page: dict[str, Any], *, rows: Sequence[bytes] | None = None
    ) -> None:
        """
        Save or reuse a page screenshot and attach its location to the capture.

        An existing image is retained unless an English capture verifies and this
        session has not yet kept a verified image for that filename. No image is saved
        when screenshots are disabled or the entry filter excludes the page.

        Args:
            page (dict[str, Any]): Capture record with entry, page and verification
                fields. When a screenshot is selected, shot and shot_dir are
                added in place, including when its existing image is reused.
            rows (Sequence[bytes] | None, optional): RGB scanlines of the evaluated
                frame. None saves the emulator's most recent frame.

        Raises:
            OSError: If the screenshot directory or image cannot be written.
            ValueError: If supplied scanlines have invalid PNG dimensions.
            AssertionError: If rows is None and no emulator frame exists.
        """
        if self.shots and page["entry"] and self.photograph_only in (None, page["entry"]):
            name = f"{page['entry']}-p{page['page']}.png"
            target = self.shots / name

            # Keep the first image unless a verified English capture can replace an
            # unverified one; repeated weaker draws must not overwrite a useful screenshot.
            if not target.exists() or (
                self.english and page.get("verified") and not self.kept.get(name)
            ):
                self.shots.mkdir(parents=True, exist_ok=True)
                if rows is None:
                    self.emu.save_screenshot(str(target))
                else:
                    write_png(target, rows, 3)
                self.kept[name] = bool(page.get("verified"))

            page["shot"], page["shot_dir"] = name, self.shots.name

    def close(self) -> None:
        """
        Unload the cartridge, deinitialize the core, and invalidate all live views.

        A session must be closed only once; repeated calls are not guarded. This method
        does not clear Python capture records or delete the instrumented scratch ROM and
        saved screenshots.
        """
        try:
            self.emu.lib.retro_unload_game()
        finally:
            self.emu.lib.retro_deinit()

    def get_memory_view(self, region: int) -> ctypes.Array[ctypes.c_ubyte]:
        """
        Return a writable ctypes view of a native memory region.

        Args:
            region (int): Libretro memory region ID: SRAM 0, cartridge RTC 1,
                WRAM 2 or VRAM 3. A region must be exposed by the loaded cartridge.

        Returns:
            ctypes.Array[ctypes.c_ubyte]: Writable uint8 view aliasing core memory; invalid after
                core unload or ROM reload.

        Raises:
            ValueError: If the region ID is unsupported or the core exposes no backing
                memory, or the ID is not an integer (excluding booleans).
        """
        # ROM reloads reacquire these views because the core may replace its
        # backing allocations; old views still reference memory the core may have released.
        lib = self.emu.lib
        if (not is_integer(region)) or region not in (0, 1, 2, 3):
            raise ValueError("Unsupported libretro memory region")
        size = lib.retro_get_memory_size(region)
        address = lib.retro_get_memory_data(region)
        if not size or not address:
            raise ValueError("The core does not expose the requested memory region")
        return cast(
            ctypes.Array[ctypes.c_ubyte],
            (ctypes.c_uint8 * size).from_address(address),
        )

    def clear_trace(self) -> None:
        """
        Clear added trace counters without changing game-owned saves.
        """
        self.emu.poke(instrument.TRACE, bytes(0x80), id_=0)

    def get_wram_view(self) -> ctypes.Array[ctypes.c_ubyte]:
        """
        Return the live, writable uint8 WRAM view.

        The view aliases core memory and is invalid after core unload or ROM reload.

        Returns:
            ctypes.Array[ctypes.c_ubyte]: The session's writable WRAM view, aliasing native memory.
                It becomes invalid after core unload or ROM reload.
        """
        return self.W

    def run(self, n: int = 1, held: Iterable[int] = ()) -> None:
        """
        Advance with integer button IDs and then release all input.

        Args:
            n (int, optional): Emulated frames to advance.
                Defaults to 1.
            held (Iterable[int], optional): Held libretro button IDs; empty
                releases all buttons.
                Defaults to ().

        Raises:
            ValueError: If the frame count is not a nonnegative integer or a button ID
                is unsupported.
            TypeError: If held is not iterable.
        """
        validate_frame_count(n)
        self.emu.held = collect_button_ids(held)
        try:
            for _ in range(n):
                self.emu.run(1)
                self.frame += 1
        finally:
            self.emu.held = set()

    def is_waiting_for_input(self) -> bool:
        """
        Check for an actor waiting on a finished page with its text window open.

        Returns:
            bool: Whether an actor is waiting on a completed page with an open text
                window, allowing the session to advance it with a button press.
        """
        w = self.get_wram_view()
        # The 37 actor slots expose a wait bit ($40) at $B1B9 and a text
        # state byte at $B712. Window state bits 0 and 1 must both be set.
        waiters = [x for x in range(37) if w[0xB1B9 + x] & 0x40 and w[0xB712 + x] == 1]
        return bool(waiters) and w[0x18B0] & 3 == 3

    def is_busy(self) -> bool:
        """
        Check whether the event text engine or dialogue-window flag is active.

        Returns:
            bool: Whether native text reading or the dialogue-window flag is active.
        """
        w = self.get_wram_view()
        # $18B0 bit 0 marks the dialogue window; $181A bit 1 marks text activity.
        return bool(w[0x18B0] & 1 or w[0x181A] & 2)

    def get_current_room(self) -> tuple[int, int]:
        """
        Read the native stage and room variant used to identify the current location.

        Returns:
            tuple[int, int]: Current native stage number and room variant from WRAM.
        """
        w = self.get_wram_view()
        # Adjacent WRAM bytes contain stage and room variant, not map coordinates.
        return w[0x1AA8], w[0x1AA9]

    def check_page(self) -> dict[str, Any]:
        """
        Identify and record the current text page, checking English glyphs and pixels.

        Japanese playback waits eight frames and records identity/room data without English
        verification. English compares traced literal glyphs with compiled source, then tests up
        to 30 settling frames (150 for centered captions), retaining the best
        missing/outside-ink counts together with that frame's number and RGB scanlines. These
        settling waits advance the session. New screenshots use the evaluated frame, including
        on failed checks; the screenshot reuse policy is described by photograph().

        Returns:
            dict[str, Any]: Captured page record appended to pages, including identification and
                English verification diagnostics when enabled. The returned record is the same
                object stored in pages. A verified matching item-name insertion may append a
                second record credited to the item entry. Unidentified English pages receive a
                problem field instead of a successful verification result.

        Raises:
            AssertionError: If no video frame has been captured.
            RuntimeError: If the retained frame lacks a required memory or register snapshot,
                including when capturing that snapshot failed.
            ValueError: If compiled text has a cyclic jump chain, glyph records contain invalid
                coordinates or font offsets, a required VRAM snapshot or RGB frame has invalid
                dimensions, menu columns lie outside the text buffer, or a screenshot has
                invalid PNG dimensions. Also raised if item-name indexing finds an invalid
                original-ROM hash or source metadata.
            IndexError: If compiled text or a followed jump extends beyond the ROM.
            KeyError: If compiled text contains an unsupported F8 command.
            OSError: If a selected screenshot directory or image cannot be written. Also raised
                if item-name indexing cannot read the Japanese ROM or translation sources.
        """
        start = self.get_current_text_start()
        entry, number = self.find_entry_at(start)
        room = list(self.get_current_room())

        # Japanese text is evidence of reachability; only English text uses the
        # installed Latin font and its pixel-verification path.
        if not self.english:
            self.run(8)  # let the page settle before photographing it
            page = dict(start=f"{start:06X}" if start >= 0 else None, entry=entry, page=number, frame=self.frame, room=room)
            self.photograph(page)
            self.pages.append(page)
            return page

        # Reconstruct what the renderer drew from its glyph trace, then compare
        # that expected ink with the captured frame.
        glyphs = instrument.read_glyph_log(self.S)
        page: dict[str, Any] = dict(
            start=f"{start:06X}" if start >= 0 else None,
            entry=entry,
            page=number,
            frame=self.frame,
            room=room,
            glyphs=len(glyphs),
        )
        if entry is None:
            page["problem"] = "text start is not a translated entry"
            self.pages.append(page)
            return page

        wanted, _ = decode_literal_glyphs(self.release, start)
        drawn = [g["font_offset"] // 22 for g in glyphs if g["bank"] not in (0x7E, 0x7F)]
        dx, dy, threshold = get_capture_window(entry)
        expected, outside = ink.reconstruct_dialogue_ink(self.release, glyphs)
        best: tuple[int, int] | None = None
        best_placement: dict[str, Any] | None = None
        best_errors: dict[str, int] = {}
        best_frame = self.frame
        best_rows: list[bytes] | None = None
        settle = 150 if dy == 105 else 30  # centred captions fade in slowly
        for attempt in range(settle):  # allow fades and portraits to settle
            rows = self.emu.read_rgb_rows()
            missing, off = ink.count_missing_dialogue_pixels(rows, expected, dx, dy, threshold)
            # Require both the dialogue-plane check and the hardware-recorded
            # placement. Bright pixels elsewhere cannot supply another scroll.
            hardware = screen_ink.check_screen_pixels(
                self.release,
                glyphs,
                rows,
                self.emu.read_frame_memory(3),
                label_starts=[
                    column * 12
                    for columns in getattr(self, "entries", {})
                    .get(entry, {})
                    .get("menu_columns", [])
                    for column in columns
                ],
                layers=capture_text_layers(
                    self.release,
                    self.emu.read_frame_memory(0),
                    self.emu.read_frame_memory(2),
                    self.emu.read_frame_registers(),
                    len(rows),
                ),
            )
            missing = max(missing, hardware["missing_screen_ink"])
            errors = {
                key: hardware[key]
                for key in (
                    "unexpected_screen_ink",
                    "missing_vram_ink",
                    "unexpected_vram_ink",
                    "clipped_ink",
                    "incomplete_copies",
                )
            }
            off += hardware["outside_ink"]
            missing += sum(errors.values())
            if best is None or missing + off < best[0] + best[1]:
                best = (missing, off)
                best_placement = hardware["placement"]
                best_errors = errors
                best_frame = self.frame
                best_rows = rows

            if not missing and not off:
                break

            if attempt + 1 < settle:
                self.run(1)

        if best is None:
            raise AssertionError("No frame was examined for text verification")

        page.update(
            frame=best_frame,
            literal_match=wanted == drawn,
            ink=len(expected),
            missing_ink=best[0],
            outside_ink=outside + best[1],
            pixel_placement=best_placement,
            pixel_errors=best_errors,
            pixel_method="recorded-display-registers",
        )
        page["verified"] = (
            bool(glyphs) and page["literal_match"] and not best[0] and not page["outside_ink"]
        )
        self.photograph(page, rows=best_rows)
        self.pages.append(page)
        inserted = [g["font_offset"] // 22 for g in glyphs if g["bank"] in (0x7E, 0x7F)]
        if inserted and page["verified"]:
            # An item name the game copied into this page (e.g. "Got CardKey1."): the item the
            # speaking object holds ($0E5A per actor), credited when the inserted glyphs are exactly
            # its name's ROM glyphs. Its ink was checked with the page.
            if self.items is None:
                self.items = index_item_names(self.release)

            # Trace $0B identifies the speaking actor whose item byte is indexed.
            name = self.items.get(self.W[0x0E5A + self.T[0x0B]])
            if name and name[1] == inserted:
                self.pages.append(dict(page, entry=name[0], page=1, inserted_in=entry))

        return page

    def play_text(self, until: Callable[[], bool], limit: int = 40000) -> bool:
        """
        Advance the game's own text with A, verifying each page, until `until()` holds.

        A page is checked when the text-terminator trace counter changes while the window
        remains open, then dismissed with A. The predicate is checked at the start of each
        iteration, before that iteration advances any frames. A zero limit returns False without
        calling it; there is no extra check after the final iteration. Exceptions from the
        predicate and page verification propagate. Emulated frames, captured pages and saved
        screenshots are not rolled back when playback fails.

        Args:
            until (Callable[[], bool]): Zero-argument predicate; playback stops when it returns
                a truthy value.
            limit (int, optional): Maximum playback-loop iterations before returning False. A
                page-confirmation iteration runs check_page(), including its settling frames,
                then advances four confirmation frames. Other iterations advance one frame.
                Defaults to 40000.

        Returns:
            bool: True if until() becomes truthy before the loop-iteration budget is exhausted;
                otherwise False.

        Raises:
            ValueError: If limit is not a nonnegative integer (excluding booleans), compiled
                text has a cyclic jump, glyph coordinates or font offsets are invalid, frame
                dimensions are invalid, menu columns lie outside the text buffer, or a
                screenshot has invalid PNG dimensions. Also raised if item-name indexing finds
                an invalid original-ROM hash or source metadata.
            RuntimeError: If page verification requires a memory or register snapshot that was
                not captured, or capturing it failed.
            AssertionError: If page verification needs a video frame before one has been
                captured.
            IndexError: If compiled text or a followed jump extends beyond the ROM.
            KeyError: If compiled text contains an unsupported F8 command.
            OSError: If a selected screenshot directory or image cannot be written. Also raised
                if item-name indexing cannot read the Japanese ROM or translation sources.
        """
        validate_frame_count(limit)
        ends = self.T[0x30]
        for _ in range(limit):
            if until():
                return True

            # Trace $30 changes when text ends. Capture that completed page before
            # pressing A, and require a new end event before advancing another page.
            if self.T[0x30] != ends and self.is_waiting_for_input():
                ends = self.T[0x30]
                self.check_page()
                self.run(4, {A})
                continue

            self.run(1)

        return False

    def boot_to_roam(self) -> None:
        """
        Advance the startup screens and opening scene until free roaming is reached.

        This method expects the session to be at the initial loaded-ROM state. It advances the
        current emulator without resetting or reloading the cartridge. Start confirms the
        default names; opening dialogue is captured through play_text(). The trace counters are
        cleared after naming, while existing page records are retained. Frames, captures and
        screenshots remain changed if a later step raises an exception.

        Raises:
            RuntimeError: If naming does not finish within 3600 emulated frames, opening
                playback reaches its loop limit before free roaming, or page verification
                requires a missing or failed memory/register snapshot.
            ValueError: If compiled text has a cyclic jump, glyph coordinates or font offsets
                are invalid, frame dimensions are invalid, menu columns lie outside the text
                buffer, or a screenshot has invalid PNG dimensions. Also raised if item-name
                indexing finds an invalid original-ROM hash or source metadata.
            AssertionError: If page verification needs a video frame before one has been
                captured.
            IndexError: If compiled text or a followed jump extends beyond the ROM.
            KeyError: If compiled text contains an unsupported F8 command.
            OSError: If a selected screenshot directory or image cannot be written. Also raised
                if item-name indexing cannot read the Japanese ROM or translation sources.
        """
        self.run(900)

        for _ in range(4):
            self.run(4, {START})
            self.run(300)

        naming_frames = 0
        while self.get_current_room()[0] == 22:  # naming screens: Start confirms
            if naming_frames >= 3600:
                raise RuntimeError("Naming screens did not finish within 3600 frames")

            self.run(4, {START})
            self.run(36)
            naming_frames += 40

        # Discard title and naming traces before collecting opening dialogue.
        # The cached roaming state needs sustained player control, not a brief idle gap.
        self.clear_trace()
        idle = 0

        def has_reached_roaming() -> bool:
            """
            Require 600 consecutive idle predicate checks in the opening field room.

            Each call increments the shared counter while room (1, 0) is idle and the page list
            is nonempty; any failed condition resets it. These are predicate evaluations, not a
            count of every emulated frame. Existing page records also satisfy the nonempty-list
            condition.

            Returns:
                bool: Whether the opening has remained idle in room (1, 0) long enough to use it
                    as the reusable free-roaming start.
            """
            nonlocal idle
            idle = (
                idle + 1
                if self.get_current_room() == (1, 0) and not self.is_busy() and self.pages
                else 0
            )
            return idle >= 600

        if not self.play_text(has_reached_roaming):
            raise RuntimeError("The opening scene did not reach free roaming")
