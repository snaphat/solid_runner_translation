"""
This module registers controller programs that navigate terminal jobs, articles, library records, and the disc viewer and capture their pages.

Availability fixtures expose specific lists or text variants. Programs navigate native job
requests, news, library records and the disc viewer, recording each page as the game draws it.
Factory callbacks keep category and variant choices separate from the shared screen driver's
capture and evaluation behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from core.validation.validate_integers import is_integer

if TYPE_CHECKING:
    from verification.screens.register_screen_programs import ScreenProgram
    from verification.screens.run_screen_checks import ScreenActions
from verification.screens.programs.controller_sequences import (
    ALL_AVAILABLE,
    TERMINAL,
    catch_drawn,
    read_game_mode,
    walk,
)

SCRIPTS: dict[str, ScreenProgram] = {}


def capture_terminal_options(d: ScreenActions) -> None:
    """
    Capture the help line for each terminal main-menu option.

    Re-enters options with Up/Down because the first help line draws when the menu opens.

    Args:
        d (ScreenActions): Driver restored to the terminal-open start.
    """
    d.run(30)
    walk(
        d,
        [
            ("down", 60),
            ("down", 60),
            ("down", 60),
            ("down", 60),
            ("up", 60),
            ("up", 60),
            ("up", 60),
            ("up", 60),
        ],
        "menu",
    )


def capture_empty_terminal(d: ScreenActions) -> None:
    """
    Capture terminal lists with their availability flags cleared.

    Args:
        d (ScreenActions): Driver restored to the terminal-open start.
    """
    d.run(30)

    for lo, n in [(0x141, 18), (0x153, 30), (0x171, 30), (0x18F, 20), (0x1A3, 38)]:
        d.poke(
            lo,
            [0] * n,
            0,
            "nothing available/read (SRAM unlock bytes; bit 7 available, bit 6 read)",
        )

    # Open each category after clearing its availability flags so the native
    # empty-list messages are captured instead of unlocked record contents.
    walk(
        d,
        [
            ("a", 220),
            ("b", 150),
            ("down", 60),
            ("a", 220),
            ("b", 150),
            ("down", 60),
            ("a", 220),
            ("b", 150),
            ("down", 60),
            ("a", 220),
            ("a", 90),
            ("right", 60),
            ("right", 60),
        ],
        "empty",
    )


SCRIPTS["terminal-main"] = {
    "state": TERMINAL,
    "run": capture_terminal_options,
    "description": "Job terminal main menu (native tile 18, room 2/3): help line per option.",
}


SCRIPTS["terminal-empty"] = {
    "state": TERMINAL,
    "run": capture_empty_terminal,
    "description": "Job terminal lists with no records available.",
    "fixtures": [
        "SRAM $70:0141-0152, 0153-0170, 0171-018E, 018F-01A2, 01A3-01C8 = 00 (nothing available)"
    ],
}


def unlock_terminal_records(d: ScreenActions, value: int = 0x80) -> None:
    """
    Apply availability, current-job and funds fixtures to the terminal.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        d (ScreenActions): Driver with the job terminal open.
        value (int, optional): Record-availability byte: bit 7 available, bit 6
            read, bit 0 second text variant.
            Defaults to 128.

    Raises:
        ValueError: If record availability is outside 0 through 255.
            Also raised if value is a boolean or noninteger.
    """
    if not is_integer(value):
        raise ValueError("value must be an integer, not a boolean")
    if not 0 <= value <= 255:
        raise ValueError("Record availability must be a byte")

    for lo, n in ALL_AVAILABLE:
        d.poke(
            lo,
            [value] * n,
            0,
            f"every record available (SRAM unlock bytes = {value:02X})",
        )

    d.poke(0x13B, [0xFF], 0, "no current job")
    d.poke(0x1AD6, (999999).to_bytes(3, "little"), 2, "funds 999999")


def read_records(
    d: ScreenActions,
    tag: str,
    step_down: bool,
    count: int,
    limit: int = 12,
    cursor_resets: bool = False,
) -> None:
    """
    This function opens records in a terminal list and captures their pages.

    Game-mode changes indicate when a record has closed and the list is visible again. The same
    input and capture timing is used for both languages. Playback stops after count attempts or
    when an opening A press leaves the mode unchanged. Reaching limit confirmation steps does
    not prove that a record's pages were exhausted and does not raise an exception.

    Counts require integers excluding booleans; integer subclasses are accepted. step_down and
    cursor_resets require bool values.

    Args:
        d (ScreenActions): Driver at an open terminal record list.
        tag (str): Category label for capture labels and descriptions.
        step_down (bool): Move the cursor down after each record when read
            records remain listed.
        count (int): Record limit; stops when none can be opened.
        limit (int, optional): Maximum page-confirmation steps per record,
            including the first page.
            Defaults to 12.
        cursor_resets (bool, optional): Return to the ith list position after
            each record when the game resets its cursor.
            Defaults to False.

    Raises:
        TypeError: If step_down or cursor_resets is not a boolean.
        ValueError: If the record count is negative or the page limit is not positive.
            Also raised if any integer argument is a boolean or noninteger.
    """
    if type(step_down) is not bool or type(cursor_resets) is not bool:
        raise TypeError("Boolean options require bool values: step_down, cursor_resets")
    if not is_integer(count) or not is_integer(limit):
        raise ValueError("count, limit must be integers, not booleans")
    # Some native lists reset the cursor after each record. Navigate from the
    # top again for those lists instead of assuming the selection was preserved.
    if count < 0 or limit <= 0:
        raise ValueError("Record count must be nonnegative and page limit positive")

    for i in range(count):
        if cursor_resets:  # the job list redraws with the cursor on job 0
            for k in range(i):
                d.clear()
                d.press("down", 4, 60)
                d.capture(
                    f"{tag}-{i:02d}-to{k:02d}", f"{tag} list, Down {k + 1} of {i}"
                )

        # Remember the list's native mode so repeated A inputs stop as soon as the
        # record closes, rather than reopening it or entering another menu.
        list_mode = read_game_mode(d)
        d.clear()
        d.press("a", 4, 180)
        d.capture(f"{tag}-{i:02d}-p0", f"{tag} record {i}")
        record_mode = read_game_mode(d)
        if record_mode == list_mode:
            break  # nothing opened: list exhausted

        for p in range(1, limit):
            d.clear()
            d.press("a", 4, 180)
            d.capture(f"{tag}-{i:02d}-p{p}", f"{tag} record {i}, A #{p}")

            if read_game_mode(d) != record_mode:  # back on the list
                break

        if step_down and not cursor_resets:
            d.clear()
            d.press("down", 4, 80)
            d.capture(f"{tag}-{i:02d}-down", f"{tag} list after Down")


def open_terminal_list(d: ScreenActions, menu: int, tag: str) -> None:
    """
    Select a top-level terminal category and capture its initial list.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        d (ScreenActions): Driver at the terminal main-menu origin after fixtures
            have been applied; the trace is cleared before entering the list.
        menu (int): Down presses selecting jobs 0, news 1, information 2 or library 3.
        tag (str): Category label used in the list's capture label and context.

    Raises:
        ValueError: If the terminal menu index is outside 0 through 3.
            Also raised if menu is a boolean or noninteger.
    """
    if not is_integer(menu):
        raise ValueError("menu must be an integer, not a boolean")
    if not 0 <= menu < 4:
        raise ValueError("Terminal menu must be between 0 and 3")

    for _ in range(menu):
        d.press("down", 4, 30)
    d.clear()
    d.press("a", 4, 220)
    d.capture(f"{tag}-list", f"{tag} list")


def create_terminal_list_capture(
    menu: int,
    tag: str,
    step_down: bool,
    count: int,
    value: int = 0x80,
    funds: int | None = 999999,
) -> Callable[[ScreenActions], None]:
    """
    Create a terminal-list callback with availability and funds fixtures.

    Integer inputs accept subclasses; booleans are rejected.

    Boolean options require bool values; numeric zero and one are rejected.

    Args:
        menu (int): Down presses from the terminal main-menu origin: jobs 0,
            news 1, information 2, library 3.
        tag (str): Category label for capture labels and descriptions.
        step_down (bool): Move the cursor down after each record when read
            records remain listed.
        count (int): Record limit; stops when none can be opened.
        value (int, optional): Record-availability byte: bit 7 available, bit 6
            read, bit 0 second text variant.
            Defaults to 128.
        funds (int | None, optional): Funds fixture written as a three-byte
            little-endian WRAM value after unlock_terminal_records() has set
            999999. None leaves that unlock-helper value in place; it does not
            preserve the original start-state funds.
            Defaults to 999999.

    Returns:
        Callable[[ScreenActions], None]: ScreenActions callback applying availability flags and
            capturing records.

    Raises:
        TypeError: If step_down is not a boolean.
        ValueError: If menu is outside 0 through 3, count is negative, value is outside
            0 through 255, or supplied funds are outside 0 through 999999.
            Also raised if menu, count, value or supplied funds is a boolean or
            noninteger.
    """
    if funds is not None and not is_integer(funds):
        raise ValueError("Terminal funds must be an integer, not a boolean")
    if type(step_down) is not bool:
        raise TypeError("Boolean options require bool values: step_down")
    if not is_integer(menu) or not is_integer(count) or not is_integer(value):
        raise ValueError("menu, count, value must be integers, not booleans")

    if (
        not 0 <= menu < 4
        or count < 0
        or not 0 <= value <= 255
        or (funds is not None and not 0 <= funds <= 999999)
    ):
        raise ValueError(
            "Terminal setup requires menu 0-3, a nonnegative record count, availability 0-255 and optional funds 0-999999"
        )

    def run(d: ScreenActions) -> None:
        """
        Capture a configured terminal list with availability and funds fixtures.

        Args:
            d (ScreenActions): Driver at an open terminal; record flags, funds and
                menu inputs are applied before capture.
        """
        d.run(30)
        unlock_terminal_records(d, value)
        if funds is not None:
            d.poke(0x1AD6, funds.to_bytes(3, "little"), 2, f"funds {funds}")

        # Select the requested top-level category only after applying its record
        # and funds fixtures; opening the list consumes those native values.
        open_terminal_list(d, menu, tag)
        read_records(d, tag, step_down, count)

    return run


# Availability records use bit 7 for available and bit 6 for already read;
# 80 and C0 exercise the unread and read variants through native menus.
UNLOCK_FIXTURE = "SRAM availability bytes $70:0141 (jobs, 18), 0153 (news, 30), 0171 (info, 30), 018F (people, 20), 01A3 (foes, 38) = {v}; $70:013B = FF (no job); $7E:1AD6 funds"


SCRIPTS["terminal-requests"] = {
    "state": TERMINAL,
    "run": create_terminal_list_capture(0, "req", True, 18),
    "description": "Every job request: list, details pages.",
    "fixtures": [UNLOCK_FIXTURE.format(v="80")],
}


SCRIPTS["terminal-news"] = {
    "state": TERMINAL,
    "run": create_terminal_list_capture(1, "news", False, 31),
    "description": "Every news item read in turn.",
    "fixtures": [UNLOCK_FIXTURE.format(v="80")],
}


SCRIPTS["terminal-info"] = {
    "state": TERMINAL,
    "run": create_terminal_list_capture(2, "info", False, 31),
    "description": "Every information item bought and read in turn.",
    "fixtures": [UNLOCK_FIXTURE.format(v="80")],
}


SCRIPTS["terminal-info-broke"] = {
    "state": TERMINAL,
    "run": create_terminal_list_capture(2, "info", False, 1, funds=0),
    "description": "Information with no funds.",
    "fixtures": [UNLOCK_FIXTURE.format(v="80") + " = 0"],
}


def create_library_capture(
    category: int, tag: str, count: int, value: int = 0xC0
) -> Callable[[ScreenActions], None]:
    """
    Create a callback for Data Library records with configured availability flags.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        category (int): Right presses selecting Topics (0), People
            (1) or Foes (2).
        tag (str): Category label for capture labels and descriptions.
        count (int): Record limit; stops when none can be opened.
        value (int, optional): Availability/read flag byte, including bit 0
            for the second text variant; C0 lists already-read records.
            Defaults to 192.

    Returns:
        Callable[[ScreenActions], None]: ScreenActions callback applying availability flags and
            capturing records.

    Raises:
        ValueError: If category is outside 0 through 2, count is negative, or value is
            outside 0 through 255.
            Also raised if any integer argument is a boolean or noninteger.
    """
    if not is_integer(category) or not is_integer(count) or not is_integer(value):
        raise ValueError("category, count, value must be integers, not booleans")

    if not 0 <= category < 3 or count < 0 or not 0 <= value <= 255:
        raise ValueError(
            "Library setup requires category 0-2, a nonnegative record count and availability 0-255"
        )

    def run(d: ScreenActions) -> None:
        """
        Capture records from the configured Data Library category.

        Args:
            d (ScreenActions): Driver at an open terminal, used to select the
                category and read its records.
        """
        d.run(30)
        unlock_terminal_records(d, value)

        # The Data Library is the fourth top-level option. Its inner categories
        # use horizontal selection, unlike the terminal's vertical main menu.
        for _ in range(3):
            d.press("down", 4, 30)

        d.press("a", 4, 220)

        for _ in range(category):
            d.press("right", 4, 60)

        d.clear()
        d.press("a", 4, 90)
        d.capture(f"{tag}-list", f"data library {tag} list")
        read_records(d, tag, True, count)

    return run


for _cat, _tag, _n in [(0, "topics", 60), (1, "people", 20), (2, "foes", 38)]:
    SCRIPTS[f"terminal-lib-{_tag}"] = {
        "state": TERMINAL,
        "run": create_library_capture(_cat, _tag, _n),
        "description": f"Data library {_tag}: every record and page.",
        "fixtures": [UNLOCK_FIXTURE.format(v="C0 (available and read)")],
    }


def accept_first_job(d: ScreenActions) -> None:
    """
    Capture the first job's offer and accept it through the native choice menu.

    The program starts at the terminal and records the pages leading to the Accept/Pass prompt
    before selecting Accept.

    Args:
        d (ScreenActions): Driver restored to the terminal-open start.
    """
    d.run(30)
    unlock_terminal_records(d)
    d.press("a", 4, 220)

    # Open the first offer, choose acceptance in its native menu, then try
    # another request with the accepted-job state still active.
    walk(
        d,
        [
            ("a", 180),
            ("a", 180),
            ("left", 60),
            ("a", 240),
            ("a", 180),
            ("a", 180),
            ("a", 180),
            ("down", 80),
            ("a", 240),
            ("a", 180),
            ("a", 180),
        ],
        "acc",
    )


SCRIPTS["terminal-accept"] = {
    "state": TERMINAL,
    "run": accept_first_job,
    "description": "Accept job 0, then open another job.",
    "fixtures": [UNLOCK_FIXTURE.format(v="80")],
}


for _cat, _tag, _n in [(0, "topics", 60), (1, "people", 20), (2, "foes", 37)]:
    SCRIPTS[f"terminal-lib-{_tag}-v1"] = {
        "state": TERMINAL,
        "run": create_library_capture(_cat, _tag, _n, 193),
        "description": f"Data library {_tag}: every record, second-text variant.",
        "fixtures": [UNLOCK_FIXTURE.format(v="C1 (available, read, bit 0)")],
    }


for _menu, _tag, _down, _n in [
    (0, "req", True, 18),
    (1, "news", False, 31),
    (2, "info", False, 31),
]:
    SCRIPTS[f"terminal-{_tag}-v1"] = {
        "state": TERMINAL,
        "run": create_terminal_list_capture(_menu, _tag, _down, _n, 129, funds=None),
        "description": f"Terminal {_tag}: every record, second-text variant.",
        "fixtures": [UNLOCK_FIXTURE.format(v="81 (available, bit 0)")],
    }


def capture_disc_viewer(d: ScreenActions) -> None:
    """
    Capture native disc-viewer dialogue from the terminal tile.

    Waits 30 frames before each sample without clearing the trace; capture adds three settling
    frames. Every fourth observation confirms with a four-frame A press.

    Args:
        d (ScreenActions): Driver restored to the terminal-disc-before start.
    """
    d.clear()
    d.run(4, ("a",))

    # Capture multiple draws per advance because the viewer can animate or
    # replace text between pages; A is submitted only every fourth sample.
    for k in range(64):
        d.run(30)
        d.capture(f"d-{k:02d}", f"disc viewer, step {k}")

        if k % 4 == 3:
            d.run(4, ("a",))


SCRIPTS["terminal-disc"] = {
    "state": "terminal-disc-before",
    "run": capture_disc_viewer,
    "description": "Terminal tile 18 (room 2/3) with the disc obtained: native recording viewer.",
    "fixtures": [
        "quest flags (start state): $70:0016 b80=0, $70:0141 b20=0, $70:001B b08=1, $70:0143 b40=0, $70:0146 b40=0"
    ],
}


def capture_job_requests(d: ScreenActions) -> None:
    """
    Sample up to sixteen job requests, returning from the reset cursor each time.

    Args:
        d (ScreenActions): Driver restored to the terminal-open start.
    """
    d.run(30)
    unlock_terminal_records(d)
    d.clear()
    d.press("a", 4, 220)
    d.capture("req-list", "request list")

    # This request list restores its cursor when a record closes. Let the
    # record reader advance from that reset origin instead of assuming cursor retention.
    read_records(d, "req", True, 16, cursor_resets=True)


SCRIPTS["terminal-requests-all"] = {
    "state": TERMINAL,
    "run": capture_job_requests,
    "description": "Every job request (16): Down to each from the reset cursor, details pages.",
    "fixtures": [UNLOCK_FIXTURE.format(v="80")],
}


def capture_disc_media_label(d: ScreenActions) -> None:
    """
    Capture the disc-viewer media label immediately after drawing.

    Args:
        d (ScreenActions): Driver restored to the terminal-disc-before start.
    """
    d.clear()
    d.run(4, ("a",))  # terminal tile with the disc

    for k in range(16):  # advance "Let's check out the disc." until the viewer
        d.run(30)

        if any("03B2F9" in r["ids"] for r in d.read_records()):
            break

        if k % 3 == 2:
            d.run(4, ("a",))

    d.clear()  # the viewer redraws: catch it from a clean trace
    catch_drawn(d, "03B2F9", frames=600)


SCRIPTS["disc-media-drawn"] = {
    "state": "terminal-disc-before",
    "run": capture_disc_media_label,
    "description": 'Terminal disc viewer: captured right after "Media: S-MD..." is drawn.',
    "fixtures": [
        "quest flags (start state): $70:0016 b80=0, $70:0141 b20=0, $70:001B b08=1, $70:0143 b40=0, $70:0146 b40=0"
    ],
}


def create_article_capture(
    menu: int, tag: str, availability: int
) -> Callable[[ScreenActions], None]:
    """
    Create a callback that browses the News or Information article list.

    The callback unlocks thirty records and samples up to thirty-one record openings through
    read_records(), with its default twelve-step page limit. Captured target evidence, not
    callback completion, determines coverage.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        menu (int): Down presses from the terminal main-menu origin: News 1 or
            Information 2. Jobs and Library are not article categories.
        tag (str): Category label for capture labels and descriptions.
        availability (int): SRAM offset of the 30-byte availability array:
            0x153 for News or 0x171 for Information.

    Returns:
        Callable[[ScreenActions], None]: ScreenActions callback applying availability flags and
            capturing records.

    Raises:
        ValueError: If the article category does not match the availability array used
            by the game.
            Also raised if any integer argument is a boolean or noninteger.
    """
    if not is_integer(menu) or not is_integer(availability):
        raise ValueError("menu, availability must be integers, not booleans")

    if menu not in (1, 2) or availability != (0x153 if menu == 1 else 0x171):
        raise ValueError(
            "Article category does not match its native availability array"
        )

    def run(d: ScreenActions) -> None:
        """
        Capture available articles from the configured terminal category.

        Args:
            d (ScreenActions): Driver at an open terminal; availability fixtures
                and article navigation update its captures.
        """
        d.run(30)
        d.poke(
            availability,
            [0x80] * 30,
            0,
            "legal article availability flags; unread articles",
        )
        d.poke(
            0x1AD6,
            (999999).to_bytes(3, "little"),
            2,
            "legal funds for information purchases",
        )

        # Select News or Information, then read records through native pagination
        # until the list no longer opens another article.
        open_terminal_list(d, menu, tag)
        read_records(d, tag, False, 31)

    return run


SCRIPTS["article-pages-news"] = {
    "state": TERMINAL,
    "run": create_article_capture(1, "news", 339),
    "description": "Article pagination repair: every News article, with all pages and return to list.",
    "fixtures": [
        "SRAM $70:0153-0170 = 80 (available, unread); WRAM $7E:1AD6 funds = 999999"
    ],
}


SCRIPTS["article-pages-info"] = {
    "state": TERMINAL,
    "run": create_article_capture(2, "info", 369),
    "description": "Article pagination repair: every Information article, with all pages and return to list.",
    "fixtures": [
        "SRAM $70:0171-018E = 80 (available, unread); WRAM $7E:1AD6 funds = 999999"
    ],
}
