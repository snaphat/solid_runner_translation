"""
This module advances through arena conversations and battles until the requested announcement or character level is reached.

The helpers select rounds through Riff's conversation and apply declared encounter or
experience fixtures. They change emulator memory and controller state, capture announcement
pages through the supplied controller, and return an outcome for task evaluation. Level
progression advances battles without capturing pages itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.text.read_compiled_text import iter_dialogue_pages
from core.validation.validate_integers import is_integer

if TYPE_CHECKING:
    from verification.game.control_gameplay import GameplayController

from verification.game import apply_battle_fixtures as battle_fixtures
from verification.game.control_gameplay import BTN, PlaybackStopped
from verification.game.emulator_session import A
from verification.tasks import validate_requirements as requirements

# SNES bus address of Riff's arena talk script; used to find his active actor slot.
RIFF_ARENA = 0x948D07


def advance_intro(game: GameplayController, page: dict[str, Any]) -> None:
    """
    Reapply the selected encounter flags after the arena introduction.

    The one-time introduction enables both named encounters. This helper restores the encounter
    selection requested by the task.

    Args:
        game (GameplayController): Controller whose arena task supplies
            quest_flags. Its final two flag records select the round and encounter;
            their masks are reapplied to SRAM after the introduction without
            replacing unrelated bits. Other controller state is unchanged here.
        page (dict[str, Any]): Capture record with entry, page, verification and identity
            fields; read without mutation. Only entry 0A09A8 triggers the reset.

    Raises:
        ValueError: If the active task's setup or requirement fields are invalid,
            or selected save flags have invalid addresses, masks or values.
            Boolean numeric values and incomplete flag records are rejected.
        KeyError: If the active arena task lacks quest_flags.
    """
    if page.get("entry") == "0A09A8" and (game.task or {}).get("route") == "arena":
        # Preserve save bits outside each declared mask when restoring the round selection.
        assert game.task is not None
        requirements.validate_definition(game.task)
        game.set_flags(game.task["quest_flags"][-2:])


def count_entry_pages(rom: bytes | bytearray, slot: int) -> int:
    """
    Count the compiled FD/18 page chain beginning at an original slot.

    The reader follows relocated text to each terminator and continues only across FD/18.
    It does not check whether traversal reaches another registered translation entry.
    Callers must supply a known announcement chain. Repeated page starts are rejected
    as continuation cycles.

    Arena progression uses the count to determine how many announcement pages must be observed
    before the task can stop.

    Args:
        rom (bytes | bytearray): ROM containing the announcement page
            chain.
        slot (int): Unheadered ROM file offset at the start of the entry slot.
            Integer subclasses are accepted; booleans are rejected. The buffer is not modified.

    Returns:
        int: Number of pages traversed from the supplied slot.

    Raises:
        TypeError: If rom is not bytes or bytearray, or slot is not an integer or is a boolean.
        ValueError: If a text jump or dialogue continuation chain is cyclic.
        IndexError: If a text token or jump target is outside rom.
        KeyError: If an F8 command is unsupported.
    """
    # No registered-entry boundaries apply within a caller-selected announcement.
    # The shared iterator stops at repeated starts; without a final terminator,
    # that stop means a continuation cycle rather than a complete announcement.
    for pages, _, end in iter_dialogue_pages(rom, slot, {}):
        if rom[end : end + 2] != b"\xfd\x18":
            return pages

    raise ValueError("Arena announcement continuation chain is cyclic")


def enter_round(game: GameplayController, task: dict[str, Any]) -> str:
    """
    Enter a native arena round and check its target announcement.

    The controller applies the listed progression flags and answers Riff's fee/class menus.
    The game selects the opponent; playback stops when the final target announcement page
    is captured. Result evaluation separately checks that all required pages verified.

    Args:
        game (GameplayController): Controller used to initialize the arena,
            answer Riff and capture the target announcement. Its begin callback must
            prepare a route task. Emulator memory, task state and captures change;
            target-stop settings are restored after announcement playback. The caller
            must clear the preceding task's captures before starting a new task, as
            Japanese page numbering counts observations already stored in game.pages.
        task (dict[str, Any]): Arena definition with class_index, name_profile,
            quest_flags and the hexadecimal target announcement ID. class_index is
            0 through 2, excluding booleans. name_profile is default or widest;
            widest writes six wide glyphs into the runner-name buffer. Route initialization
            also requires key, route=arena, and stage/variant integers from 0 through 255.
            Integer subclasses are accepted; booleans are rejected. Read without mutation.

    Returns:
        str: idle when the requested announcement boundary is reached;
            otherwise blocked_active_initialization, actor_unavailable or
            target_not_reached.

    Raises:
        ValueError: If the task's numeric fields, name profile, target ID or flags are invalid,
            or its announcement chain is cyclic. Setup and playback validation errors propagate.
        TypeError: If the announcement ROM buffer is not bytes or bytearray.
        KeyError: If a required task field is missing or announcement text uses an unsupported F8 command.
        IndexError: If announcement text or a followed jump extends beyond the ROM.
        AssertionError: If begin requires a base state that has not been prepared, or capture
            validation needs a video frame before one has been captured.
        RuntimeError: If room setup cannot finish or a required frame snapshot is unavailable.
        OSError: If setup sources or captured screenshots cannot be read or written.
        PlaybackStopped: If setup or the 24-frame interval after the announcement
            reaches a separately configured stop. Announcement-playback stops are
            handled here; a stop during the later interval propagates after restoring
            the target-stop settings.
    """
    requirements.validate_definition(task)
    if task["name_profile"] not in ("default", "widest"):
        raise ValueError("Arena name_profile must be default or widest")
    target = task["target"]
    if not isinstance(target, str) or not target:
        raise ValueError("Arena target must be a hexadecimal entry ID")
    target_pages = count_entry_pages(game.release, int(target, 16))
    # Required selection fields must be present before begin can change emulator state.
    class_index = task["class_index"]
    flags = task["quest_flags"][-2:]
    game.begin(task)

    if task["name_profile"] == "widest":
        # Runner-name WRAM buffer: six wide glyphs followed by its FF terminator.
        game.poke(0x1ACF, b"\x98" * 6 + b"\xff")

    # Finish initialization before applying the round-selection flags; room
    # setup can overwrite them before the arena attendant reads the current round.
    prep = game.play_dialogue("initialization", 24000)
    if prep["status"] != "idle" or game.is_busy():
        return "blocked_active_initialization"

    game.set_flags(flags)

    slots = game.find_talk_slots(f"{RIFF_ARENA:06X}")
    if len(slots) != 1:
        return "actor_unavailable"

    # Request a stop on the announcement's final page, then confirm that this
    # page actually caused the stop rather than a separate battle-stop condition.
    previous_target, previous_pages = game.target, game.target_pages
    game.target, game.target_pages = target, target_pages
    first_page = len(game.pages)
    try:
        game.talk(slots[0])
        game.play_dialogue("entry", 10000, 120, [0, class_index, 0])
        game.play_dialogue("arena", 12000, 12000)
        return "target_not_reached"
    except PlaybackStopped:
        # Stable-battle stops use the same exception; only the requested page is success.
        if not any(
            page.get("entry") == target and page.get("page") == target_pages
            for page in game.pages[first_page:]
        ):
            return "target_not_reached"
        game.run(24)
        return "idle"
    finally:
        game.target, game.target_pages = previous_target, previous_pages


def advance_levels(game: GameplayController, task: dict[str, Any]) -> str:
    """
    Continue native battles and level-up code to a target level.

    Args:
        game (GameplayController): Controller whose battle HP, earned
            experience, task state, and controller input are changed while reaching
            the requested level. This loop does not capture/verify text pages itself.
        task (dict[str, Any]): Progression task whose level selects the original
            experience threshold. level is a required integer from 1 through 255,
            excluding booleans; integer subclasses are accepted. The original task
            dictionary is retained by the controller but is not modified here.

    Returns:
        str: idle after applying experience and reaching a quiet field at the
            requested level, experience_not_applied if that field state is
            reached without the fixture, or frame_limit if playback times out.

    Raises:
        ValueError: If the task's setup fields or required level are invalid, or the
            three-byte experience threshold lies outside the release ROM. Emulator
            input and memory-write validation errors also propagate.
        PlaybackStopped: If the controller reaches a separately configured stable-battle stop.
    """
    requirements.validate_definition(task)
    level = task.get("level")
    if not is_integer(level) or not 1 <= level <= 255:
        raise ValueError(
            "Arena progression level must be an integer from 1 through 255"
        )
    threshold_offset = 0x1E4A2 + (level - 1) * 3
    if threshold_offset + 3 > len(game.release):
        raise ValueError("Arena experience threshold lies outside the release ROM")

    # The ROM table at file offset $01E4A2 stores three-byte thresholds per level.
    # The earned-experience WRAM buffer at $18AD receives this value only while
    # the post-battle experience handler ($90:E577) and completion flag are active.
    threshold = int.from_bytes(
        game.release[threshold_offset : threshold_offset + 3], "little"
    )
    game.task = task
    w, applied, quiet, age = game.W, False, 0, 0
    seen: set[tuple[int, int]] = set()

    # Bound progression to 200,000 frames; success also requires 180 quiet field frames.
    for frame in range(200000):
        # Actor handler addresses use parallel low/high/bank WRAM arrays, with 37 slots.
        handlers = {
            w[0x970 + i] | w[0x995 + i] << 8 | w[0x9BA + i] << 16 for i in range(37)
        }
        if not applied and 0x90E577 in handlers and w[0x16BD] & 0x20:
            game.poke(0x18AD, threshold.to_bytes(3, "little"))
            applied = True

        if w[0x1835] == 0xCA and w[0x1836] == 0xD1:
            # $1835/$1836 identify the idle battle handler. HP words use
            # $189C (current enemy) and $1898 (maximum); $16BD bit 5 marks defeat.
            # Track each opponent/round pair so the bounded health fixture is applied
            # once per fight, without repeatedly changing health during native combat.
            opponent = (w[0x182C], game.S[0xC4])
            hp, maximum = w[0x189C] | w[0x189D] << 8, w[0x1898] | w[0x1899] << 8
            if opponent not in seen and 0 < hp <= maximum and not w[0x16BD] & 0x20:
                game.poke(0x189C, (1).to_bytes(2, "little"))
                seen.add(opponent)

            battle_fixtures.refill_player_health(game)

        text = bool(w[0x18B0] & 1 or game.find_text_waiters())
        if text:
            # Pulse confirm for four of every eighteen frames while dialogue is active.
            held: set[int] = {A} if frame % 18 < 4 else set()
        else:
            # Advance combat for forty of every 130 non-dialogue frames.
            held = {A, BTN["right"]} if age % 130 < 40 else set()
            age += 1

        game.run(1, held)

        # A level increase alone is insufficient: require the battle and text to
        # finish and the field state to remain quiet before accepting progression.
        # $179C excludes battle mode 120; actor zero must use field-script bank
        # $83, and the current level byte at $1883 must meet the requested level.
        text = bool(w[0x18B0] & 1 or game.find_text_waiters())
        field = (
            w[0x179C] != 120 and w[0x9BA] == 0x83 and w[0x1883] >= level and not text
        )
        quiet = quiet + 1 if field else 0
        if quiet >= 180:
            return "idle" if applied else "experience_not_applied"

    return "frame_limit"
