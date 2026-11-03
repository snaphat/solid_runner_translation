"""
GameplayController drives actor interactions, room changes, menu choices, and dialogue playback through a running emulator.

GameplayController extends the emulator session with actor interaction, room loading, menu
choices and task-timed input. It advances the game's event and text engines, waits for control
to return, and reports idle periods and requested stopping points. Declared fixtures are applied
at the relevant native state transitions.

Persistent prerequisite state and spawn tracing support full-run calibration. Route, battle, and
screen runners choose the interactions. This controller waits until the game accepts input,
advances events, and passes completed pages to each runner's observation method.
"""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Sequence,
    TypedDict,
    cast,
)

from core.rom.convert_addresses import (
    convert_pointer_bytes_to_file,
    convert_snes_to_file,
    resolve_bank_pointer,
)
from core.text import read_translation_files as script_files
from core.text.read_compiled_text import decode_literal_glyphs
from core.validation.validate_integers import is_integer
from emulator.validate_controller_input import (
    BTN,
    collect_button_ids,
    validate_frame_count,
)
from verification.game import apply_battle_fixtures as battle_fixtures
from verification.game import navigate_maze as maze
from verification.game.emulator_session import BOOTED, A, GameSession

# Unheadered ROM text offsets -> final digit glyph for the native 3/2/1 countdown.
# These automatic displays must not be mistaken for dialogue awaiting a button.
COUNTDOWN = {0x12BF39: 0xA6, 0x12BF48: 0xA5, 0x12BF57: 0xA4}

# Consecutive idle playback checks required to stop after player control returns.
SETTLED = 120

# Emulated frames between detecting a completed page/menu and submitting its input.
PAGE_WAIT, MENU_WAIT = 16, 24


class PlaybackStopped(Exception):
    """
    Signal deliberate playback termination at a requested page or stable battle.

    Runners catch this exception as a stopping boundary, not an emulator failure. No payload is
    attached; the configured controller target/halt state identifies which boundary was
    requested.
    """


class NativeMenu(TypedDict, total=False):
    """
    Native choice-window state used by automatic dialogue input.

    All fields are optional while discovery is in progress. find_following_menu() returns
    options/cancel; its caller adds id. An empty dictionary represents an untranslated countdown
    page, not a selectable choice window.

    Attributes:
        id (str): Owning entry ID or identifier of an untranslated control page.
        options (int): Number of selectable choices.
        cancel (bool): Whether the menu accepts B as cancellation.
    """

    id: str
    options: int
    cancel: bool


class PendingInput(TypedDict):
    """
    Deferred input owed to the currently visible page or menu.

    Attributes:
        due (int): Session frame at which input may be submitted.
        menu (NativeMenu | None): Choice window to answer, or None to confirm text.
    """

    due: int
    menu: NativeMenu | None


class DialogueOutcome(TypedDict):
    """
    Dialogue playback completion state and the choices made during that playback.

    Attributes:
        status (str): Completion or stopping reason, such as idle or frame_limit.
        choices (list[dict[str, Any]]): Ordered menu IDs, option counts and selected
            indices or the cancel action recorded during playback.
    """

    status: str
    choices: list[dict[str, Any]]


def plan_menu_inputs(options: int, selected: object, cursor: int = 0) -> list[str]:
    """
    Plan directional presses for the native two-, three- or four-option menu.

    Four-option menus use the row/column bit layout at $81:D1AE and assume the cursor starts at
    zero. Two- and three-option menus move from the current cursor, which a previously held
    direction may already have shifted.

    Args:
        options (int): Native menu option count: two, three or four. Integer
            subclasses are accepted; booleans are rejected.
        selected (object): Desired zero-based integer option index. Integer
            subclasses are accepted; booleans and nonintegers are rejected.
        cursor (int, optional): Current zero-based cursor for two- and three-option
            menus; out-of-range integer values use zero. Integer subclasses are
            accepted; booleans and nonintegers are rejected. Four-option menus do not use its value,
            but its type is still checked.
            Defaults to 0.

    Returns:
        list[str]: Directional button names that move the native menu cursor to
            selected.

    Raises:
        ValueError: If options is not an integer count of two, three or four,
            selected is not an integer within the menu, or cursor is not an integer. Booleans are rejected for these values.
    """
    # bool inherits from int but does not represent a menu count or selection.
    if not is_integer(options) or options not in (2, 3, 4):
        raise ValueError("Unsupported original menu layout")
    if not is_integer(selected) or not 0 <= selected < options:
        raise ValueError("Choice outside the original menu")

    if not is_integer(cursor):
        raise ValueError("Menu cursor must be an integer, not a boolean")

    # Four-choice menus arrange indices as row/column bits rather than a
    # vertical list: Down toggles bit 0 and Right toggles bit 1.
    if options == 4:
        return (["down"] if selected & 1 else []) + (["right"] if selected & 2 else [])

    if options in (2, 3):
        cursor = cursor if 0 <= cursor < options else 0
        return (
            ["down"] * (selected - cursor) if selected >= cursor else ["up"] * (cursor - selected)
        )

    raise ValueError("Unsupported original menu layout")


def find_following_menu(rom: bytes | bytearray, terminator: int) -> NativeMenu | None:
    """
    Find a native 44/45 menu command after text, following local and long gotos.

    Args:
        rom (bytes | bytearray): Event bytecode scanned after the text
            terminator.
        terminator (int): File offset of the text terminator preceding a
            possible menu command. The scan starts at the following byte and
            follows at most sixteen event positions. It needs eight readable bytes
            at each position; an unrecognized command, short tail, or repeated
            position ends the search without a menu.

    Returns:
        NativeMenu | None: Native menu layout with options count and cancel
            permission; None if no matching command follows.

    Raises:
        ValueError: If terminator is a boolean or noninteger, or lies outside
            the supplied ROM. Integer subclasses are accepted.
    """
    if not is_integer(terminator) or not 0 <= terminator < len(rom):
        raise ValueError("Text terminator lies outside the ROM")
    pointer = terminator + 1
    seen: set[int] = set()
    for _ in range(16):
        if pointer in seen or pointer + 8 > len(rom):
            break

        seen.add(pointer)
        raw = rom[pointer : pointer + 8]

        # Opcode 44 declares a menu; 45 3A 01 70 stores its result in SRAM $013A.
        # The option byte's high bit permits cancellation.
        if raw[0] == 0x44 and raw[2:6] == bytes.fromhex("45 3a 01 70"):
            return {"options": raw[1] & 127, "cancel": bool(raw[1] & 128)}

        # Follow event gotos while looking for the menu: 42 stays in this bank,
        # whereas 43 supplies a complete banked address.
        if raw[0] == 0x42:
            pointer = resolve_bank_pointer(raw[1] | raw[2] << 8, pointer)
        elif raw[0] == 0x43:
            pointer = convert_pointer_bytes_to_file(raw[1], raw[2], raw[3])
        else:
            break

    return None


class GameplayController(GameSession):
    """
    Drive event playback, actor interaction and deterministic controller timing.

    GameSession manages the emulator and its memory buffers. This controller tracks the active
    task's input actions and prerequisite scenes. Before stopping when targets are reached, it
    passes captures to observe_page(), which subclasses implement.

    Attributes:
        task (Optional[dict]): Active route definition, set before playback.
        entry_setup (Optional[dict]): Room-specific entry preparation selected for
            the active task.
        prerequisite_pages (list[dict]): Native captures retained from earlier
            scenes required by the active task.
        early_stop (bool): Permit ending ordinary tasks once all expected pages
            are checked.
        fast_text (bool): Accelerate ordinary field text, while preserving
            battle text timing.
        skip_waits (bool): Shorten event-script waits for calibrated quick
            playback.
        record_spawns (bool): Attach native script-spawn state to captured
            pages for calibration.
        record_chains (bool): Record completed prerequisite scene outcomes as
            persistent-state blobs.
        targets_left (set[tuple[str, int] | str] | None): Remaining (entry, page)
            pairs for English, or entry IDs for Japanese. None disables target
            stopping; an empty set means every requested target was checked.
    """

    early_stop = True  # a task with an 'expect' list ends once all those pages are checked (not inside a 'before')
    fast_text = (
        True  # text speed $18C1 = 0 (pages drawn at once) outside battles; battles keep Fast (1)
    )
    skip_waits = False  # script waits (opcode 02: B712,X) end after one frame (verify_gameplay.py's quick run)
    record_spawns = (
        False  # note each script the game spawns ($81:D4AE) and tag the pages it draws (full run)
    )
    record_chains = (
        False  # keep each earlier scene's outcome as persistent state (full run; set_persistent)
    )
    cache: dict[str, bytes]
    watch: str | None
    sighted: tuple[int, int, Sequence[int]] | None
    maze_goal: int
    maze_steps: int
    health_fixed: set[tuple[int, int]]
    defeated: set[tuple[int, int]]
    turn_ends: dict[tuple[int, int], int]
    at_turn_end: bool
    begin: Callable[[dict[str, Any]], None]
    in_before: int = 0
    targets_left: set[tuple[str, int] | str] | None = None

    # Prerequisite-state serialization order: (libretro memory region, byte offset, byte count).
    # Region 2 is WRAM ($7E:0000 origin); region 0 is cartridge SRAM. Keep this order stable:
    # serialize_persistent_state()/set_persistent() concatenate without headers.
    PERSISTENT = ((2, 0x1AA8, 0xC8), (2, 0x186B, 0x45), (2, 0xD027, 0x54), (0, 0x0000, 0x209))

    def __init__(
        self,
        rom: bytes | bytearray,
        scratch: Path,
        english: bool = True,
        shots: Path | None = None,
        accumulate: bool = False,
    ) -> None:
        """
        Initialize task-local input, prerequisite and spawn-trace state.

        After GameSession loads the emulator, this constructor loads translation records and
        initializes controller state. If this additional initialization fails, the emulator is
        closed before the error propagates.

        Boolean options require bool values; numeric zero and one are rejected.

        Args:
            rom (bytes | bytearray): Uninstrumented source ROM retained by reference.
            scratch (Path): Output path for the ROM file with tracing code added.
            english (bool, optional): True for English, False for the Japanese original.
                Defaults to True.
            shots (Path | None, optional): Screenshot directory; None disables saving.
            accumulate (bool, optional): Retain glyph traces across interface draws. Defaults to
                False.

        Raises:
            TypeError: If rom is not bytes or bytearray, or english or accumulate is not a
                boolean.
            OSError: If instrumentation files, the scratch ROM, native core or translation
                sources cannot be accessed.
            ValueError: If instrumentation, source metadata, text chains or required native
                memory are invalid.
            RuntimeError: If Snes9x rejects the instrumented ROM.
            IndexError: If text indexing follows an entry, operand or token outside the ROM.
            KeyError: If indexed text contains an unsupported F8 command.
        """
        if type(english) is not bool or type(accumulate) is not bool:
            raise TypeError("Boolean options require bool values: english, accumulate")
        super().__init__(rom, scratch, english=english, shots=shots, accumulate=accumulate)
        try:
            self.entries = {e["id"]: e for n in script_files.NAMES for e in script_files.load(n)}
            self.base: bytes | None = None
            self.advance = 0
            self.task: dict[str, Any] | None = None
            self.entry_setup: dict[str, Any] | None = None
            self.counted: int | None = None
            self.controller = False
            self.control_frame = 0
            self.player_control_seen = False
            self.target: str | None = None
            self.target_pages = 0
            self.menu_open = False
            self.free_handler: int | None = None
            self.chains: dict[
                str, bytes
            ] = {}  # earlier scenes already played this run: their end state
            self.chain_pages: dict[str, list[dict[str, Any]]] = {}
            self.prerequisite_pages: list[dict[str, Any]] = []
            self.spawns: dict[
                int, tuple[int, int, int, str]
            ] = {}  # slot: (script start, parent, stamp, serialized battle state)
            self.spawn_choices: dict[
                int, list[int | str]
            ] = {}  # slot: the menu answers its spawned script got
            self.chain_states: dict[
                str, str
            ] = {}  # earlier scene: its recorded persistent state (quick run)
            self.chain_records: dict[
                str, str
            ] = {}  # earlier scene: its persistent state, recorded (full run)
            self.halt: str | None = None
            self.halt_ready = 0  # 'battle': end the current earlier scene once its battle is ready
        except BaseException:
            # The base constructor has already loaded the core. Release it if
            # translation loading or controller initialization cannot finish.
            self.close()
            raise

    def poke(self, address: int, data: bytes | Iterable[int], region: int = 2) -> None:
        """
        Apply a bounded byte write at a region-relative offset.

        Args:
            address (int): Byte offset relative to the selected native memory region. Integer
                subclasses are accepted; booleans are rejected.
            data (bytes | Iterable[int]): Replacement byte values; writes are bounded by the
                selected memory region.
            region (int, optional): Libretro memory region ID: 0 is SRAM, 1 is RTC, 2 is WRAM
                and 3 is VRAM. Integer subclasses are accepted; booleans are rejected. Defaults
                to 2.

        Raises:
            ValueError: If address or region is a boolean or noninteger, the region ID is
                unsupported, the span exceeds that region, or a payload value is a boolean,
                noninteger or outside the byte range.
            TypeError: If data is not bytes or an iterable of integer byte values.
        """
        self.emu.poke(address, data, id_=region)

    def press(self, button: str, frames: int = 4, after: int = 10) -> None:
        """
        Press one named button, then resume normal automatic input handling.

        Args:
            button (str): One button name from BTN, such as a, up or start.
            frames (int, optional): Emulated frames to hold the button. Defaults to 4.
            after (int, optional): Frames after the explicit press; any active task controller
                may supply input during this interval. Defaults to 10.

        Raises:
            ValueError: If a hold or release duration is not a nonnegative integer.
            KeyError: If button is not a key in BTN.
            PlaybackStopped: If advancing emulation reaches the configured stable battle.
        """
        validate_frame_count(frames)
        validate_frame_count(after)
        self.run(frames, {BTN[button]})
        self.run(after)

    def read_actor_pointer(self, base: int, slot: int) -> int:
        """
        Read a 24-bit actor pointer from three 37-byte low/high/bank tables at base.

        Args:
            base (int): WRAM offset of the low-byte table; high and bank tables
                follow at 37-byte intervals. Integer subclasses are accepted;
                booleans are rejected.
            slot (int): Zero-based actor slot, from 0 through 36. Integer
                subclasses are accepted; booleans are rejected.

        Returns:
            int: Packed low/high/bank actor pointer.

        Raises:
            ValueError: If base or slot is a boolean or noninteger, the slot is
                outside 0 through 36, or the three tables exceed WRAM.
        """
        w = self.W
        if (
            not is_integer(slot)
            or not is_integer(base)
            or (not 0 <= slot < 37)
            or (not 0 <= base <= len(w) - 111)
        ):
            raise ValueError("Actor pointer requires a valid slot and three complete tables")

        # Actor pointers use three parallel 37-byte arrays: low byte, high byte
        # and bank, rather than adjacent three-byte records.
        return w[base + slot] | w[base + 37 + slot] << 8 | w[base + 74 + slot] << 16

    def find_talk_slots(self, talk: str, expected_root: str | None = None) -> list[int]:
        """
        Find live actors matching a talk handler and optional owner.

        Args:
            talk (str): Six-digit hexadecimal SNES bus address of the actor talk
                handler.
            expected_root (str | None, optional): Optional owner/root script
                address used to distinguish actors sharing a handler. None
                disables owner filtering; otherwise exactly six ASCII hexadecimal
                digits are required.

        Returns:
            list[int]: Matching zero-based actor slots in ascending slot order.

        Raises:
            ValueError: If talk or a supplied expected_root is not a string of
                exactly six ASCII hexadecimal digits.
        """
        # Reject prefixes, signs and whitespace that int(..., 16) would accept.
        for address in (talk,) if expected_root is None else (talk, expected_root):
            if (
                not isinstance(cast(object, address), str)
                or len(address) != 6
                or any(char not in "0123456789abcdefABCDEF" for char in address)
            ):
                raise ValueError("Actor handler addresses must contain six hexadecimal digits")
        wanted = int(talk, 16)
        root = int(expected_root, 16) if expected_root is not None else None
        return [
            x
            for x in range(37)
            if self.read_actor_pointer(0xB4C2, x) == wanted
            and (root is None or self.read_actor_pointer(0xB228, x) == root)
        ]

    def find_text_waiters(self) -> list[int]:
        """
        Return zero-based actor slots waiting on an event text page.

        Returns:
            list[int]: Actor slots whose native text-page wait state is active.
        """
        w = self.W

        # Actor flag bit 6 marks a text wait; the parallel wait-counter byte
        # must be exactly one before the actor is ready for confirmation input.
        return [x for x in range(37) if w[0xB1B9 + x] & 0x40 and w[0xB712 + x] == 1]

    def is_player_free(self) -> bool:
        """
        Check whether the player actor has regained its cached idle-field handler.

        Returns:
            bool: Whether the current player actor uses the previously observed idle
                field handler; False until that handler has been established.
        """
        return (
            self.free_handler is not None
            and self.read_actor_pointer(0x970, self.W[0x173A]) == self.free_handler
        )

    def is_busy(self) -> bool:
        """
        Check text/window waits and pending controller steps that block field input.

        Returns:
            bool: Whether text, dialogue-window flags or an unfinished controller
                sequence currently block idle field playback.
        """
        w = self.W
        return bool(w[0x18B0] & 1 or w[0x181A] & 2 or self.find_text_waiters()) or (
            self.controller and self.control_frame < self.count_controller_frames()
        )

    def set_flags(self, flags: Iterable[dict[str, Any]]) -> None:
        """
        Apply address/mask/value save-flag fixtures.

        Args:
            flags (Iterable[dict[str, Any]]): SRAM fixtures with address, mask and value. Zero
                masks are skipped; actor synchronization offsets are forbidden.

        Raises:
            ValueError: If a flag has an invalid address, byte value or mask, or targets actor
                synchronization state.
            KeyError: If a flag record lacks its required address, mask or value field.
        """
        flags = list(flags)
        # Validate the complete fixture batch before applying any save changes.
        for f in flags:
            address, mask, value = f["address"], f["mask"], f["value"]
            if any((not is_integer(v)) for v in (address, mask, value)) or not (
                0 <= address < len(self.S) and 0 <= mask <= 255 and 0 <= value <= 255
            ):
                raise ValueError("Save flag address, mask or value is out of range")
            if value & ~mask:
                raise ValueError("Save flag value sets bits outside its mask")
            if mask and 0x209 <= address <= 0x212:
                raise ValueError("Actor synchronization flags cannot be forced")

        for f in flags:
            if not f["mask"]:
                continue

            # Preserve every bit outside the declared mask. The fixture value
            # contains the replacement bits in their final bit positions.
            old = self.S[f["address"]]
            self.poke(f["address"], [(old & ~f["mask"]) | f["value"]], region=0)

    def load_room(self, room: Sequence[int]) -> None:
        """
        Request a (stage, variant) room load through native field initialization.

        This writes the requested stage/variant and sets the room-load flag. It does not advance
        emulation; the caller must run frames or dialogue playback for native room
        initialization to execute.

        Args:
            room (Sequence[int]): Two unsigned byte values: stage number, then room variant.

        Raises:
            ValueError: If the room is not exactly two byte values.
        """
        if len(room) != 2 or any(
            (not is_integer(value)) or not 0 <= value <= 255 for value in room
        ):
            raise ValueError("Room must contain exactly two bytes: stage and variant")
        self.poke(0x1AA8, room)
        self.poke(0x1719, [self.W[0x1719] | 0x80])

    def clear_log(self) -> None:
        """
        Clear instrumentation and the last counted text command.
        """
        self.clear_trace()
        self.counted = None

    def start(self) -> list[dict[str, Any]]:
        """
        Reach free roaming and restore the shared starting state for this runner.

        The game boots once per runner class and source-ROM hash in each worker process. Later
        calls load that cached emulator state and share its prerequisite and battle-start
        caches. This instance's pages are cleared without resetting its Python frame counter.

        Returns:
            list[dict[str, Any]]: Shallow copies of the cached opening capture records.

        Raises:
            RuntimeError: If boot, naming or opening-scene playback does not finish, or an
                emulator state cannot be saved or loaded. Also raised if page verification
                requires a missing or failed frame snapshot.
            AssertionError: If the cached free-roaming state is missing. Also raised if page
                verification needs a video frame before one has been captured.
            ValueError: If a text chain is cyclic, glyph coordinates or font offsets are
                invalid, frame/VRAM dimensions are invalid, a menu column is outside the buffer,
                source metadata is invalid, or the original-ROM hash differs.
            IndexError: If verified text or a followed jump extends beyond the ROM.
            KeyError: If verified text contains an unsupported F8 command.
            OSError: If page screenshots cannot be saved or item-name sources cannot be read.
        """
        key = (type(self).__name__, hashlib.sha1(self.release).hexdigest())
        if key not in BOOTED:
            self.boot_to_roam()
            BOOTED[key] = (
                self.emu.save_state(),
                self.read_actor_pointer(0x970, self.W[0x173A]),
                self.pages,
            )

        self.base, self.free_handler, opening = BOOTED[key]  # the player's idle field handler

        # Earlier scenes and starts already reached in this process are kept with it as well.
        self.chains = BOOTED.setdefault(key + ("chains",), self.chains)
        if hasattr(self, "cache"):
            self.cache = BOOTED.setdefault(key + ("cache",), self.cache)

        if self.base is None:
            raise AssertionError("Free-roaming state has not been initialized")

        self.emu.load_state(self.base)
        self.pages = []
        return [dict(p) for p in opening]

    def save_field_snapshot(self) -> None:
        """
        Run the native field snapshot routine and restore the player.

        Temporary event bytecode is written at WRAM $7000 and a completion byte at $7020. The
        player actor's handler, script pointer and wait bytes are restored even if playback
        raises. The native field snapshot and the temporary WRAM bytes remain changed.

        Raises:
            RuntimeError: If the native snapshot routine did not run on the player.
            PlaybackStopped: If advancing emulation reaches the configured stable battle.
        """
        # Install a temporary WRAM script that invokes the game's field snapshot
        # operation and sets a completion byte at $7E:7020.
        self.poke(0x7020, [0])
        self.poke(0x7000, bytes.fromhex("05 37 cf 80 25 20 70 7e 01 02 ff 00"))
        actor = self.W[0x173A]
        addresses = [
            base + actor for base in (0x970, 0x995, 0x9BA, 0xB297, 0xB2BC, 0xB2E1, 0xB712, 0xB1B9)
        ]

        # Retain the player actor's handler, script pointer and wait flags so
        # the temporary script can run without replacing its ongoing behavior.
        saved = {a: self.W[a] for a in addresses}

        for address, value in zip(addresses[:3], bytes.fromhex("6d 80 83")):
            self.poke(address, [value])

        for address, value in zip((0xB297, 0xB2BC, 0xB2E1), bytes.fromhex("00 70 7e")):
            self.poke(address + actor, [value])

        self.poke(0xB712 + actor, [0])
        self.poke(0xB1B9 + actor, [0])
        try:
            self.run(2)
        finally:
            # A playback boundary or emulator error must not leave the temporary
            # snapshot script installed on the player's actor.
            for address, value in saved.items():
                self.poke(address, [value])

        if self.W[0x7020] != 1:
            raise RuntimeError("Native field snapshot did not run on the player")

    def relocate_player(self, x: int, y: int, room: Sequence[int] | None = None) -> None:
        """
        Position the player using the native field-return loader.

        This method saves the field snapshot, writes the player/partner return packet in WRAM,
        and requests the destination room. The caller advances emulation to let the native
        loader apply that packet. Completed memory writes are not rolled back if a later
        operation fails.

        Args:
            x (int): Native field X coordinate in pixels; negative values are clamped to zero.
                The maximum is 0xFFF7, leaving eight pixels for the partner in the 16-bit X
                field.
            y (int): Native field Y coordinate in pixels; negative values are clamped to zero.
                The maximum is 0x0FFF; the remaining packet bits hold direction flags.
            room (Sequence[int] | None, optional): Destination [stage, variant]; None uses the
                active task stage and variant.

        Raises:
            ValueError: If the room or coordinate types are invalid, or the clamped coordinates
                exceed the native packet fields.
            AssertionError: If no room was supplied and there is no active replay task.
            RuntimeError: If the native field snapshot routine did not run on the player.
            PlaybackStopped: If advancing emulation reaches the configured stable battle.
        """
        if room is None:
            if self.task is None:
                raise AssertionError("Operation requires an active replay task")
            room = [self.task["stage"], self.task["variant"]]
        if len(room) != 2 or any(
            (not is_integer(value)) or not 0 <= value <= 255 for value in room
        ):
            raise ValueError("Room must contain exactly two bytes: stage and variant")
        if (not is_integer(x)) or (not is_integer(y)):
            raise ValueError("Field coordinates must be integer pixel positions")
        x, y = max(0, x), max(0, y)
        # The partner is eight pixels to the right; Y shares its upper nibble
        # with direction flags in the native field-return packet.
        if x > 0xFFF7 or y > 0xFFF:
            raise ValueError("Coordinates do not fit the native field-return packet")
        self.save_field_snapshot()

        # Use the game's field-return packet so the next room load rebuilds actor
        # and camera state through its own loader.
        packet = bytearray(self.W[0xD029:0xD040])
        packet[0] = max(0, min(max(0, self.W[0xA4] - 16), (x - 120) // 16))
        packet[1] = max(0, min(max(0, self.W[0xA5] - 10), (y - 80) // 16))
        for offset, px, py in (
            (7, x, y),
            (11, max(0, x - 8), max(0, y - 8)),
            (15, x + 8, max(0, y - 8)),
        ):
            packet[offset : offset + 2] = px.to_bytes(2, "little")
            direction = packet[offset + 3] & 0xF0
            packet[offset + 2 : offset + 4] = (py | (direction << 8)).to_bytes(2, "little")

        self.poke(0xD029, packet)

        self.poke(0x1AA8, room)
        self.poke(0x1719, [self.W[0x1719] | 0xA0])

    def talk(self, actor: int) -> None:
        """
        Start an actor conversation through native field state.

        Args:
            actor (int): Zero-based live actor slot whose interaction is invoked.

        Raises:
            ValueError: If the actor slot is not an integer from 0 through 36.
        """
        if (not is_integer(actor)) or not 0 <= actor < 37:
            raise ValueError("Actor slot must be between 0 and 36")
        self.poke(0x181A, [self.W[0x181A] | 2])
        self.poke(0x171A, [self.W[0x171A] | 8])
        self.poke(0x173A, [0])
        self.poke(0xB1B9 + actor, [self.W[0xB1B9 + actor] | 4])
        self.poke(0x18C1, [0])

    def remember_actor_position(self) -> None:
        """
        Remember the last observed position of the requested actor.

        Actors can unload off-screen. Their remembered position allows the spatial loader to
        restore them when the player approaches.

        Raises:
            AssertionError: If no actor talk-handler address is configured in watch.
        """
        if self.watch is None:
            raise AssertionError("Actor observation requires a talk-handler address")

        for x in self.find_talk_slots(
            self.watch, (self.task or {}).get("root")
        ):  # the task's own actor only
            w = self.W
            self.sighted = (
                w[0xA73 + x] | w[0xA98 + x] << 8,
                w[0xAE2 + x] | w[0xB07 + x] << 8,
                list(self.get_current_room()),
            )

    def approach_sighted(self, talk: str) -> list[int]:
        """
        Approach a remembered actor through native field relocation.

        Args:
            talk (str): Six-digit hexadecimal SNES bus address of the actor talk handler.

        Returns:
            list[int]: Matching live actor slots after approach/setup; empty if no eligible
                actor is available.

        Raises:
            PlaybackStopped: If playback reaches the configured text target or stable battle.
            ValueError: If a talk-handler or supplied actor-root address is not six hexadecimal
                digits. Also raised if the remembered position cannot fit the native
                field-return packet. Also raised if a text chain is cyclic, glyph coordinates or
                font offsets are invalid, frame/VRAM dimensions are invalid, a menu column is
                outside the buffer, source metadata is invalid, or the original-ROM hash
                differs.
            RuntimeError: If the native field snapshot routine does not run on the player. Also
                raised if page verification requires a missing or failed frame snapshot.
            AssertionError: If page verification needs a video frame before one has been
                captured.
            IndexError: If verified text or a followed jump extends beyond the ROM.
            KeyError: If verified text contains an unsupported F8 command.
            OSError: If page screenshots cannot be saved or item-name sources cannot be read.
        """
        if not self.sighted or self.is_busy():
            return []

        x, y, room = self.sighted  # in the room where it was seen, not where the player is now
        self.relocate_player(x, y + 24, room)

        if self.play_dialogue("approach", 12000)["status"] != "idle":
            return []

        self.press("up", 2, 12)
        self.play_dialogue("approach", 12000)
        return self.find_talk_slots(talk, (self.task or {}).get("root"))

    def catch_early(self, task: dict[str, Any]) -> list[int]:
        """
        Catch an eligible actor before its room settles.

        Some NPCs leave before the room falls quiet. This method waits until their handler is
        live and ordinary interaction is possible.

        Args:
            task (dict[str, Any]): Complete route task passed to the runner's begin callback
                before searching. talk is the hexadecimal handler address; expected_actor_root
                optionally distinguishes actors sharing it.

        Returns:
            list[int]: Matching actor slots once the field permits interaction; empty if room
                text starts first or the 6000-frame wait expires.

        Raises:
            PlaybackStopped: If advancing emulation reaches the configured stable battle.
            ValueError: If a talk-handler or supplied actor-root address is not six hexadecimal
                digits.
        """
        self.begin(task)

        for _ in range(6000):
            self.run(1)
            candidates = self.find_talk_slots(task["talk"], task.get("expected_actor_root"))

            # Wait for a talk actor before the room emits its own text. WRAM $1719
            # bits C0 indicate a pending transition, and trace $0C counts text starts.
            if (
                candidates
                and not self.is_busy()
                and not self.W[0x1719] & 0xC0
                and self.T[0x0C] == 0
            ):
                return candidates

            if self.T[0x0C]:  # the room started its own text first
                return []

        return []

    def await_talk(self, task: dict[str, Any], frames: int) -> bool | None:
        """
        Wait for an eligible actor before another text page starts.

        Args:
            task (dict[str, Any]): Actor query with hexadecimal talk handler and optional
                expected_actor_root; used while initialization runs.
            frames (int): Maximum frames to wait for an idle field with the requested actor
                handler.

        Returns:
            bool | None: True once the actor is available and the field is idle; None if the
                field becomes busy, the text-start count changes, or the frame budget expires.

        Raises:
            ValueError: If the frame budget is not a nonnegative integer. Also raised if a
                talk-handler or supplied actor-root address is not six hexadecimal digits.
            PlaybackStopped: If advancing emulation reaches the configured stable battle.
        """
        # Remember the text-start count: a new native text or busy state means
        # the opportunity to approach this actor without interrupting the room has passed.
        validate_frame_count(frames)
        scans = self.T[0x0C]
        for _ in range(frames):
            self.run(1)

            if self.is_busy() or self.T[0x0C] != scans:
                return None

            if not self.W[0x1719] & 0xC0 and self.find_talk_slots(
                task["talk"], task.get("expected_actor_root")
            ):
                return True

        return None

    def identify_special_screen(self) -> str | None:
        """
        Return battle, mission_results or None for ordinary field play.

        Returns:
            str | None: battle or mission_results for recognized special screens;
                None for ordinary field playback.
        """
        # Native game mode 120 identifies battle playback. Results instead
        # use the dedicated stage/room pair 23/7.
        if self.W[0x179C] == 120:
            return "battle"

        return "mission_results" if self.get_current_room() == (23, 7) else None

    def get_opponent_health_fixture(self) -> int | None:
        """
        Return the selected native or story opponent-health setting.

        A nonzero native setting takes precedence; otherwise use the story setting. None and
        zero leave the fixture inactive. Integer subclasses are accepted; booleans are rejected.

        Returns:
            int | None: Native enemy-health override, falling back to the task's
                story-battle HP override; None or zero when neither is enabled.

        Raises:
            ValueError: If either configured value is a boolean, noninteger or
                outside the unsigned 16-bit range.
        """
        task = self.task or {}
        for name in ("native_enemy_health_fixture", "story_enemy_hp"):
            value = task.get(name)
            if value is not None and (not is_integer(value) or not 0 <= value <= 0xFFFF):
                raise ValueError(f"{name} must be an integer 0-65535, not a boolean")
        return task.get("native_enemy_health_fixture") or task.get("story_enemy_hp")

    def start_script(self, script: bytes | Iterable[int]) -> None:
        """
        Install supplied event bytecode for the idle carrier actor to run.

        This sets the script pointer and clears waits without advancing emulation. The caller
        must run frames to execute the installed script.

        Args:
            script (bytes | Iterable[int]): Event-script bytecode copied to WRAM offset 0x7000
                and assigned to carrier actor slot 2.

        Raises:
            TypeError: If script is not bytes or an iterable of integer byte values.
            ValueError: If a script value is not an integer byte, is a boolean, or the script
                exceeds WRAM.
        """
        self.poke(0x7000, script)

        for address, value in zip((0xB297, 0xB2BC, 0xB2E1), bytes.fromhex("00 70 7e")):
            self.poke(address + 2, [value])

        self.poke(0xB712 + 2, [0])
        self.poke(0xB1B9 + 2, [0])

    def serialize_persistent_state(self) -> str:
        """
        Return configured persistent WRAM/SRAM ranges concatenated as a hex string.

        Returns:
            str: Hexadecimal bytes concatenated in PERSISTENT order for later
                restoration by set_persistent().
        """
        return b"".join(
            bytes((self.W if region == 2 else self.S)[at : at + size])
            for region, at, size in self.PERSISTENT
        ).hex()

    def set_persistent(self, recorded: str) -> None:
        """
        Restore a prerequisite scene's recorded persistent state.

        The free-roaming state is restored first, followed by the recorded persistent ranges and
        derived field-return values used by $80:8DA7.

        Args:
            recorded (str): Hexadecimal persistent-state blob produced by
                serialize_persistent_state().

        Raises:
            ValueError: If the recorded hex data does not exactly fill the persistent ranges.
            AssertionError: If the free-roaming snapshot has not been initialized.
            PlaybackStopped: If advancing emulation reaches the configured stable battle.
            RuntimeError: If the emulator rejects the saved base state.
        """
        if self.base is None:
            raise AssertionError("Free-roaming state has not been initialized")

        data = bytes.fromhex(recorded)
        if len(data) != sum(size for _, _, size in self.PERSISTENT):
            raise ValueError("Persistent state does not match the configured memory ranges")
        self.emu.load_state(self.base)

        # Apply only serialized persistent ranges to a clean roaming snapshot;
        # transient actor state is rebuilt by the native field-return code.
        at = 0
        for region, address, size in self.PERSISTENT:
            self.poke(address, data[at : at + size], region=region)
            at += size

        w = self.W
        self.poke(0x07BC, [w[0x1AA8], w[0x1AA9]], region=0)
        self.poke(0xD05D, [w[0x1AA8]])
        self.poke(0xD05C, [w[0x1AA9]])
        self.poke(0x170C, [(w[0x1AA8] - 1) & 0xFF])
        self.poke(0x07BE, [w[0x1AB2], w[0x1AB3], w[0x1AB4], w[0x1AB5]], region=0)
        self.run(1)

    def start_cutscene(self, slot: int, address: int) -> None:
        """
        Start an actor script using native room-cutscene conventions.

        This method follows the $81:C14A room-cutscene convention: it installs the scene
        handler, updates actor/scene flags and clears waits.

        Args:
            slot (int): Actor slot from 0 through 36 on which to start the scene script.
            address (int): SNES bus address of the scene script, from 0 through 0xFFFFFF.

        Raises:
            ValueError: If the actor slot is not an integer from 0 through 36, or the script
                address is not an integer from 0 through 0xFFFFFF. Booleans are rejected.
        """
        if (
            (not is_integer(slot))
            or (not is_integer(address))
            or not 0 <= slot < 37
            or not 0 <= address <= 0xFFFFFF
        ):
            raise ValueError("Invalid actor slot or 24-bit script address")
        w = self.W

        # Actor handler and script pointers are stored as separate low/high/bank
        # arrays. Install the native $83:8049 scene handler before the scene bytecode.
        for base, value in zip((0x0970, 0x0995, 0x09BA), (0x49, 0x80, 0x83)):
            self.poke(base + slot, [value])

        self.poke(0x181C, [(w[0x181C] + 1) & 0xFF])

        for base, value in zip((0xB297, 0xB2BC, 0xB2E1), address.to_bytes(3, "little")):
            self.poke(base + slot, [value])

        # Apply the actor/scene flags and clear both wait fields using the
        # native room-cutscene setup, rather than only replacing the script pointer.
        self.poke(0xB5A0 + slot, [0xD9])
        self.poke(0xB5C5 + slot, [0xD9])
        self.poke(0xB1B9 + slot, [(w[0xB1B9 + slot] | 0x08) & ~0x40])
        self.poke(0xB1DE + slot, [w[0xB1DE + slot] | 0x10])
        self.poke(0x1447 + slot, [slot + 1])
        self.poke(0x171A, [w[0x171A] | 0x20])
        self.poke(0xB712 + slot, [0])
        self.poke(0xB737 + slot, [0])

    def spawn_script(self, address: int, parent: int) -> int:
        """
        Allocate and initialize a script actor using native conventions.

        The method selects the first free actor slot from 0x16 through 0x23 and initializes the
        parent, actor stamp and two-frame delay using the $81:D4AE spawn convention.

        Args:
            address (int): SNES bus address of the script to spawn, from 0 through 0xFFFFFF.
            parent (int): Parent actor slot from 0 through 36 assigned to the new script.

        Returns:
            int: Native actor slot allocated to the spawned script.

        Raises:
            ValueError: If the parent slot is not an integer from 0 through 36, or the script
                address is not an integer from 0 through 0xFFFFFF. Booleans are rejected.
            StopIteration: If no native actor slot is free for the spawned script.
        """
        if (
            (not is_integer(parent))
            or (not is_integer(address))
            or not 0 <= parent < 37
            or not 0 <= address <= 0xFFFFFF
        ):
            raise ValueError("Invalid parent slot or 24-bit script address")
        w = self.W

        # Use a free native spawned-script slot rather than overwriting one of
        # the fixed room actors.
        slot = next(s for s in range(0x16, 0x24) if not w[0x0995 + s])
        stamp = w[0x1707] or 1
        self.poke(0x1707, [(stamp + 1) & 0xFF])
        self.poke(0x146C + slot, [stamp])
        self.poke(0x01E5, [parent])
        self.poke(0x1491 + slot, [parent])

        for base, value in zip((0xB297, 0xB2BC, 0xB2E1), address.to_bytes(3, "little")):
            self.poke(base + slot, [value])

        self.poke(0xB712 + slot, [0])
        self.poke(0xB737 + slot, [0])
        self.poke(0x0F38 + slot, [2])

        for base, value in zip((0x0970, 0x0995, 0x09BA), (0xFC, 0xD4, 0x81)):
            self.poke(base + slot, [value])

        self.poke(0x181A, [w[0x181A] & ~1])
        return slot

    def is_page_finished(self) -> bool:
        """
        Check that the traced terminator belongs to the current page.

        Menus, map labels and dynamic insertions read their own terminators while the start
        pointer may still name an earlier dialogue page.

        Returns:
            bool: True when the traced terminator belongs to the current translated
                page.
        """
        t = self.T
        start = self.get_current_text_start()
        term = convert_snes_to_file(t[0x32] | t[0x33] << 8 | t[0x34] << 16)

        # Names copied into WRAM and unrelated text draws can update the terminator
        # trace. Require a known ROM page and agreement with its decoded terminator.
        if t[0x34] in (0x7E, 0x7F) or (
            start not in self.starts if self.english else self.find_entry_at(start)[0] is None
        ):
            return False

        try:
            return decode_literal_glyphs(self.release, start)[1] == term
        except (ValueError, IndexError, KeyError):
            return False

    def identify_control_page(self) -> NativeMenu | None:
        """
        Identify untranslated native menu or countdown waits.

        Returns:
            NativeMenu | None: Menu layout (with native ID), an empty dict for a
                countdown page, or None for an ordinary/unready page.
        """
        t, rom = self.T, self.release
        if t[0x34] in (0x7E, 0x7F):
            return None

        start = self.get_current_text_start()
        term = convert_snes_to_file(t[0x32] | t[0x33] << 8 | t[0x34] << 16)

        # Translated entries use their registered page logic. This fallback handles
        # only native waits that do not belong to the translation tables.
        if self.find_entry_at(start)[0] is not None:
            return None

        try:
            if decode_literal_glyphs(rom, start)[1] != term:
                return None
        except (ValueError, IndexError, KeyError):  # not text: a stale pointer
            return None

        # FE can precede a native choice menu. Countdown text has a separate
        # known byte pattern; other unregistered text must not acquire invented menu behavior.
        if rom[term] == 0xFE:
            menu = find_following_menu(rom, term)
            if menu:
                menu["id"] = f"native-{start:06X}"

            return menu

        if (
            start in COUNTDOWN
            and rom[start - 1] == 0x18
            and rom[start : start + 12]
            == bytes([0]) + bytes([1]) * 9 + bytes([COUNTDOWN[start], 0xFF])
        ):
            return {}

        return None

    def count_controller_frames(self) -> int:
        """
        Return the total frame count in the active task's controller_steps.

        Returns:
            int: Sum of frame counts in the active task's controller_steps, or zero
                when no sequence is configured.

        Raises:
            ValueError: If a step duration is a boolean, noninteger or negative.
        """
        steps = (self.task or {}).get("controller_steps", [])
        for step in steps:
            validate_frame_count(step["frames"])
        return sum(step["frames"] for step in steps)

    def is_controller_blocked(self) -> bool:
        """
        Check text, loading and player-control gates that prevent task input.

        When a task requests wait_for_player_control, the first successful release sets
        player_control_seen so later calls do not reapply that one-time gate.

        Returns:
            bool: Whether text, room loading or the required player-control release
                currently prevents submission of the next controller-sequence frame.
        """
        w = self.W
        if w[0x18B0] & 1 or self.find_text_waiters() or w[0x1719] & 0x80:
            return True

        if (self.task or {}).get("wait_for_player_control") and not self.player_control_seen:
            # After a conversation the field keeps $171A bits 01/02/08 until the window closes; a
            # held direction then never starts the walk. Wait once for the release.
            text_busy = (
                w[0x181A] & 2 or self.find_text_waiters()
            )  # (is_busy() also counts these pending steps)
            finished = any(p.get("phase") == "conversation" for p in self.pages) and not text_busy
            if w[0x171A] & 0x0B or self.control_frame == 0 and not finished:
                return True

            self.player_control_seen = True

        return False

    def get_controller_inputs(self) -> set[int]:
        """
        Return libretro button IDs, counting only unblocked sequence frames.

        Returns:
            set[int]: Buttons for the next unblocked sequence frame. Blocked calls
                return an empty set without advancing the sequence counter. Unblocked
                calls increment control_frame even after every configured step is past.

        Raises:
            ValueError: If a controller-step duration is a boolean, noninteger or negative.
            AssertionError: If no task is active after the input gates clear.
            KeyError: If a step names a button not present in BTN.
        """
        if self.is_controller_blocked():
            return set()

        # Advance the task's input timeline only after native input gates clear;
        # blocked frames must not consume the scheduled movement.
        self.count_controller_frames()
        age = self.control_frame
        self.control_frame += 1

        if self.task is None:
            raise AssertionError("Operation requires an active replay task")

        for step in self.task["controller_steps"]:
            if age < step["frames"]:
                return {BTN[b] for b in step["buttons"]}

            age -= step["frames"]

        return set()

    def run(self, n: int = 1, held: Iterable[int] = ()) -> None:
        """
        Advance task input, capture bookkeeping and declared fixtures.

        Args:
            n (int, optional): Emulated frames to advance. Defaults to 1.
            held (Iterable[int], optional): Explicit libretro button IDs. A nonempty set
                overrides task-controller input; empty allows the active controller sequence to
                supply input when eligible. Defaults to ().

        Raises:
            ValueError: If a frame count or controller-step duration is not a nonnegative
                integer excluding booleans, a button ID is unsupported, or an opponent-health
                setting is a boolean, noninteger or outside the unsigned 16-bit range.
            TypeError: If held is not iterable.
            PlaybackStopped: If the configured stable battle state is reached.
        """
        validate_frame_count(n)
        held_buttons = collect_button_ids(held)
        self.get_opponent_health_fixture()
        for _ in range(n):
            buttons = held_buttons.copy()
            if (
                not buttons
                and self.controller
                and self.control_frame < self.count_controller_frames()
                and not self.menu_open
            ):
                buttons = self.get_controller_inputs()

            w = self.W
            if self.fast_text:
                # Only while a dialogue window is open outside battles; battles read the menu's Fast
                # setting.
                speed = 0 if w[0x181A] & 2 and w[0x179C] != 120 else 1
                if w[0x18C1] in (0, 1, 2, 4) and w[0x18C1] != speed:
                    self.poke(0x18C1, [speed])

            if (
                self.skip_waits
                and not self.in_before
                and not (self.task or {}).get("controller_steps")
                and not (w[0x18B0] & 1 or w[0x181A] & 2)
            ):
                # Opcode 02 stores a frame countdown at $B712 + actor slot; FF means
                # stopped. Reducing an active countdown to one advances the script
                # next frame. The conditions above preserve normal timing during
                # text display, scheduled controller input, and prerequisite scenes.
                # Those waits can allow graphics uploads to finish before text draws.
                for slot in range(0x25):
                    if 1 < w[0xB712 + slot] < 0xFF:
                        self.poke(0xB712 + slot, [1])

            if self.get_opponent_health_fixture() and not (self.task or {}).get("lose_fixture"):
                battle_fixtures.refill_player_health(self)

            super().run(1, buttons)

            if self.record_spawns:
                # A script spawned by $81:D4AE takes a new actor stamp ($146C) on a slot from $16
                # and waits in the handler $81:D4FC before its first command: there its pointer is
                # the script's start.
                for slot in range(0x16, 0x24):
                    stamp = w[0x146C + slot]
                    fresh = (w[0x0970 + slot], w[0x0995 + slot], w[0x09BA + slot]) == (
                        0xFC,
                        0xD4,
                        0x81,
                    )
                    if fresh and self.spawns.get(slot, (0, 0, None, ""))[2] != stamp:
                        start = w[0xB297 + slot] | w[0xB2BC + slot] << 8 | w[0xB2E1 + slot] << 16

                        # Save battle state ($1820-$18AF), outcome flags ($16BC-$16BD),
                        # and story flags ($70:0000-$0212) before this script runs.
                        # Its branches may read and then reset these values, so a
                        # later snapshot would not reproduce the same decisions.
                        state = (
                            bytes(w[0x1820:0x18B0])
                            + bytes(w[0x16BC:0x16BE])
                            + bytes(self.S[0x000:0x213])
                        ).hex()
                        self.spawns[slot] = (start, w[0x1491 + slot], stamp, state)
                        self.spawn_choices[slot] = []

            if self.get_opponent_health_fixture():
                battle_fixtures.finish_battle(
                    self, after_turns=6
                )  # the battle's own captions first, then the win

            if self.halt == "battle":
                ready = (
                    w[0x179C] == 120
                    and w[0x183B] == 0x40
                    and not GameplayController.is_busy(self)
                    and not w[0x1719] & 0x80
                )
                self.halt_ready = self.halt_ready + 1 if ready else 0
                if self.halt_ready >= 30:
                    raise PlaybackStopped

    def check_page(self) -> dict[str, Any]:
        """
        Capture and observe a page before enforcing its target stop.

        GameSession records the page before observe_page() receives that same dictionary.
        Exceptions from the hook propagate. Captures and any hook changes remain recorded when
        the target stop raises PlaybackStopped.

        Returns:
            dict[str, Any]: Captured page after adding spawn state and running the observe_page
                hook; may stop at a requested target.

        Raises:
            PlaybackStopped: If the requested stable battle/target boundary is reached.
            AssertionError: If page verification needs a video frame before one has been
                captured.
            RuntimeError: If page verification requires a missing or failed frame snapshot.
            ValueError: If a text chain is cyclic, glyph coordinates or font offsets are
                invalid, frame/VRAM dimensions are invalid, a menu column is outside the buffer,
                source metadata is invalid, or the original-ROM hash differs.
            IndexError: If verified text or a followed jump extends beyond the ROM.
            KeyError: If verified text contains an unsupported F8 command.
            OSError: If page screenshots cannot be saved or item-name sources cannot be read.
        """
        page = super().check_page()
        if self.record_spawns and "{F812}" in (
            self.entries.get(page.get("entry") or "", {"en": ""})
        ).get("en", ""):
            # The opponent's name ({F812}) is read through the opponent number $187A ($84:ABE1):
            # keep it, so the line can be shown directly with the same name (direct.json).
            page["state"] = {"187A": self.W[0x187A]}

        # Route state must be updated even when this page completes a target
        # and immediately transfers control through PlaybackStopped below.
        self.observe_page(page)

        if (
            self.target
            and page.get("entry") == self.target
            and page.get("page") == self.target_pages
        ):
            raise PlaybackStopped

        return page

    def observe_page(self, page: dict[str, Any]) -> None:
        """
        Handle a captured page before target-stop evaluation; the base hook does nothing.

        Args:
            page (dict[str, Any]): Newly captured page record, supplied before target-stop
                evaluation. The base hook leaves it unchanged.
        """
        pass

    def play_dialogue(
        self,
        phase: str,
        max_frames: int = 12000,
        quiet_target: int = 1200,
        plan: Sequence[int | Literal["cancel"]] | None = None,
    ) -> DialogueOutcome:
        """
        This method plays dialogue and enables scheduled controller inputs for the matching task
        phase.

        play_text_loop() handles page checks and menu choices. When this phase matches the
        task's controller sequence, its frame counter starts at zero. Scheduled input is
        disabled on exit, including when playback raises an exception.

        Args:
            phase (str): Capture-phase label; controller steps run when it matches the
                configured controller phase.
            max_frames (int, optional): Playback budget measured by the session frame counter;
                the final input/settling step may cross this boundary. Defaults to 12000.
            quiet_target (int, optional): Consecutive idle playback-loop checks required to
                finish playback. Defaults to 1200.
            plan (Sequence[int | Literal['cancel']] | None, optional): Ordered zero-based menu
                selections used in encountered-menu order; missing selections default to zero. A
                setup's per-entry menu selections or maze route takes precedence. Use cancel
                only where the native menu supports it.

        Returns:
            DialogueOutcome: status string and ordered choices made during playback. Outcomes
                include idle, frame_limit and explicit choice failures.

        Raises:
            PlaybackStopped: If playback reaches the configured text target or stable battle
                state.
            ValueError: If a playback or idle-frame budget, menu definition, or planned
                selection is invalid. Also raised if a text chain is cyclic, glyph coordinates
                or font offsets are invalid, frame/VRAM dimensions are invalid, a menu column is
                outside the buffer, source metadata is invalid, or the original-ROM hash
                differs.
            RuntimeError: If a requested maze goal is disconnected from the current cell. Also
                raised if page verification requires a missing or failed frame snapshot.
            AssertionError: If page verification needs a video frame before one has been
                captured.
            IndexError: If verified text or a followed jump extends beyond the ROM.
            KeyError: If verified text contains an unsupported F8 command.
            OSError: If page screenshots cannot be saved or item-name sources cannot be read.
        """
        validate_frame_count(max_frames)
        validate_frame_count(quiet_target)
        # Task metadata names controller phases differently from the playback loop.
        # Activate timed input only for the matching phase, starting its frame count at zero.
        wanted = (self.task or {}).get("controller_phase", "initialization")
        phases = {
            "registered_conversation": "conversation",
            "native_tile_interaction": "tile",
            "native_spawn": "spawn",
            "native_controller_interaction": "controller",
        }
        active = bool(self.count_controller_frames()) and phases.get(wanted, wanted) == phase
        if active:
            self.controller, self.control_frame = True, 0

        try:
            return self.play_text_loop(phase, max_frames, quiet_target, plan)
        # Timed input must stop even if playback raises or reaches a deliberate
        # stop boundary, so held inputs cannot leak into another phase.
        finally:
            if active:
                self.controller = False

    def play_text_loop(
        self,
        phase: str,
        max_frames: int = 12000,
        quiet_target: int = 1200,
        plan: Sequence[int | Literal["cancel"]] | None = None,
    ) -> DialogueOutcome:
        """
        This method advances text, answers planned menus, and waits for gameplay to become idle.

        Completed pages are checked after their terminators are read. Confirmation input follows
        after PAGE_WAIT frames, or MENU_WAIT for a menu. Menu selections are submitted only
        while the game is ready to accept them. play_dialogue(), rather than this method,
        activates phase-specific controller sequences.

        Playback may end when all targets are recorded or after SETTLED idle checks with player
        control restored. The task's wait_full setting instead requires quiet_target idle
        playback checks. Required event and room-exit conditions must still be satisfied.

        Args:
            phase (str): Label attached to captured pages and used to select required_by_phase
                completion requirements.
            max_frames (int, optional): Playback budget measured by the session frame counter;
                the final input/settling step may cross this boundary. Defaults to 12000.
            quiet_target (int, optional): Consecutive idle playback-loop checks required to
                finish playback. Defaults to 1200.
            plan (Sequence[int | Literal['cancel']] | None, optional): Ordered zero-based menu
                selections used in encountered-menu order; missing choices default to zero.
                Per-entry setup plans and maze routing can override them.

        Returns:
            DialogueOutcome: status string and ordered choices made during playback. Outcomes
                include idle, frame_limit, repeated_script, choice_plan_exhausted,
                maze_route_exhausted, and choice_selection_failed. Completed captures and
                choices remain recorded even for a non-idle result.

        Raises:
            ValueError: If a playback/idle budget is invalid, a native menu lacks its required
                identity/options, or a planned selection is invalid. Also raised if a text chain
                is cyclic, glyph coordinates or font offsets are invalid, frame/VRAM dimensions
                are invalid, a menu column is outside the buffer, source metadata is invalid, or
                the original-ROM hash differs.
            PlaybackStopped: If reading a page or advancing frames reaches the configured text
                target or stable battle state.
            RuntimeError: If a requested maze goal is disconnected from the current cell. Also
                raised if page verification requires a missing or failed frame snapshot.
            AssertionError: If page verification needs a video frame before one has been
                captured.
            IndexError: If verified text or a followed jump extends beyond the ROM.
            KeyError: If verified text contains an unsupported F8 command.
            OSError: If page screenshots cannot be saved or item-name sources cannot be read.
        """
        if plan is not None and any(
            value != "cancel" and (not is_integer(value) or not 0 <= value <= 3) for value in plan
        ):
            raise ValueError("Menu plans must contain integer indices 0-3 or cancel")
        validate_frame_count(max_frames)
        validate_frame_count(quiet_target)
        task = self.task or {}
        quiet, began = 0, self.frame
        choices: list[dict[str, Any]] = []
        visits: collections.Counter[tuple[Any, ...]] = collections.Counter()
        pending: PendingInput | None = None

        if self.targets_left == set():
            return {"status": "idle", "choices": []}  # every target page is already checked

        self.menu_open = False
        choice_count: collections.Counter[str] = collections.Counter()
        ends = self.T[0x30]
        screen_age, battle_advance = 0, 0
        while self.frame - began < max_frames:
            held: set[int] = (
                {A}
                if self.advance > 0
                else {BTN[task["battle_input_fixture"]]}
                if battle_advance
                else set()
            )

            if self.advance:
                self.advance -= 1

            if battle_advance:
                battle_advance -= 1

            self.run(1, held)

            if self.watch and self.frame % 8 == 0:
                self.remember_actor_position()

            if self.T[0x30] != ends and self.is_page_finished():
                ends = self.T[0x30]

                if self.T[0x0C] == self.counted:  # this text command's page was already checked
                    continue

                self.counted = self.T[0x0C]

                # An A owed to the previous page must not confirm the next page
                # or select its default menu option.
                self.advance = 0

                # Detect loops by text terminator and actor, with additional state below
                # for mazes and battles where repeated text can indicate real progress.
                key: tuple[Any, ...] = (
                    self.T[0x32] | self.T[0x33] << 8 | self.T[0x34] << 16,
                    self.T[0x0B],
                )

                if (self.entry_setup or {}).get("maze_goals"):
                    key += (self.W[0x1814], self.W[0xBD])

                if (task.get("controller_steps") or self.get_opponent_health_fixture()) and self.W[
                    0x179C
                ] == 120:
                    key += (self.W[0x1822],)

                if self.get_opponent_health_fixture():
                    # Captions temporarily change screen mode in the original ROM too.
                    # Completed battle turns remain progress throughout those overlays.
                    key += (tuple(sorted(self.turn_ends.items())),)

                visits[key] += 1

                if visits[key] > (10 if (self.entry_setup or {}).get("maze_goals") else 4):
                    return {"status": "repeated_script", "choices": choices}

                page = self.check_page()
                if self.record_spawns and self.T[0x0B] in self.spawns:
                    # The script's start and parent, and the battle's state at the spawn and now (a
                    # script may wait for the fight to reach a point before this line).
                    start, parent, _, state = self.spawns[self.T[0x0B]]

                    # with the story flags and actor sync values (save RAM $70:0000-$0212) its
                    # branches and waits read (ops 30-35, 89).
                    now = (
                        bytes(self.W[0x1820:0x18B0])
                        + bytes(self.W[0x16BC:0x16BE])
                        + bytes(self.S[0x000:0x213])
                    ).hex()
                    page["spawn"] = [
                        f"{start:06X}",
                        parent,
                        state,
                        now,
                        list(self.spawn_choices.get(self.T[0x0B], [])),
                    ]

                page["phase"] = phase
                if self.targets_left:
                    self.targets_left.discard(
                        (page.get("entry"), page.get("page")) if self.english else page.get("entry")
                    )

                    if not self.targets_left:
                        self.run(16)
                        return {"status": "idle", "choices": choices}

                menu = None
                entry = self.entries.get(page.get("entry") or "")
                if (
                    entry
                    and entry.get("end")
                    and entry["id"] not in task.get("manual_menu_entries", [])
                ):
                    menu = find_following_menu(self.release, int(entry["end"], 16))
                    if menu:
                        menu["id"] = entry["id"]

                if page.get("entry") not in task.get("manual_menu_entries", []):
                    # A menu left to the task's own steps gets no automatic button at all.
                    pending = {"due": self.frame + (MENU_WAIT if menu else PAGE_WAIT), "menu": menu}
                    self.menu_open = bool(
                        menu
                    )  # the task's own steps wait while a menu is answered
            elif self.T[0x30] != ends and (control := self.identify_control_page()) is not None:
                # Untranslated original control text: answered or advanced, never credited.
                ends = self.T[0x30]

                if self.T[0x0C] != self.counted:
                    self.counted = self.T[0x0C]
                    pending = {
                        "due": self.frame + (MENU_WAIT if control else PAGE_WAIT),
                        "menu": control or None,
                    }

            if pending and self.frame >= pending["due"]:
                menu = pending["menu"]
                selected = 0
                if menu:
                    if "id" not in menu or "options" not in menu:
                        raise ValueError(
                            "Automatic menu input requires an entry ID and option count"
                        )

                    selected = (plan or [])[len(choices)] if len(plan or []) > len(choices) else 0
                    choice_plan = (self.entry_setup or {}).get("choice_plans", {}).get(menu["id"])
                    if choice_plan:
                        number = choice_count[menu["id"]]
                        choice_count[menu["id"]] += 1

                        if number >= len(choice_plan):
                            return {"status": "choice_plan_exhausted", "choices": choices}

                        selected = choice_plan[number]

                    if menu["id"] == "09EBEF" and (self.entry_setup or {}).get("maze_goals"):
                        selected = maze.choose_option(self)
                        if selected is None:
                            return {"status": "maze_route_exhausted", "choices": choices}

                    choices.append(dict(id=menu["id"], options=menu["options"], selected=selected))

                    if self.record_spawns and self.T[0x0B] in self.spawns:
                        self.spawn_choices.setdefault(self.T[0x0B], []).append(
                            selected
                        )  # its answers so far

                    if selected == "cancel":
                        self.press("b", 4, 20)
                    else:
                        for button in plan_menu_inputs(menu["options"], selected, self.S[0x13A]):
                            self.press(button, 4, 20)

                        if self.S[0x13A] != selected:
                            return {"status": "choice_selection_failed", "choices": choices}

                self.advance = 0 if menu and selected == "cancel" else 4
                self.menu_open = False
                pending = None

            special = self.identify_special_screen()
            if special == "battle" and self.get_opponent_health_fixture() and not self.is_busy():
                battle_fixtures.lower_enemy_health(self)

            if self.entry_setup and special:
                # Battles and mission results wait for a button without an event-text waiter.
                screen_age += 1

                if screen_age % 90 == 0 and not pending and not self.is_busy():
                    if special == "battle" and task.get("battle_input_fixture"):
                        battle_advance = 4
                    else:
                        self.advance = 4
            else:
                screen_age = 0

            if (
                self.is_busy()
                or pending
                or self.advance
                or battle_advance
                or (self.entry_setup and special)
                or self.W[0x1719] & 0xC0
            ):  # room load and fade-in
                quiet = 0
            else:
                quiet += 1

            if quiet >= quiet_target or (
                quiet >= SETTLED
                and quiet_target > SETTLED
                and self.is_player_free()
                and not task.get("wait_full")
            ):
                required = (self.entry_setup or {}).get("required_event_ids", []) + (
                    self.task or {}
                ).get("required_by_phase", {}).get(phase, [])
                seen = {p.get("entry") for p in self.pages}
                if not set(required) <= seen or (self.entry_setup or {}).get(
                    "required_leave_room"
                ) == list(self.get_current_room()):
                    continue

                return {"status": "idle", "choices": choices}

        return {"status": "frame_limit", "choices": choices}
