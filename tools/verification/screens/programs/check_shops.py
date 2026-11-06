"""
This module registers controller programs that browse shop and garage lists, perform transactions, and capture their messages.

Programs browse stock, sale quantities, upgrades and weapon swaps through native menus. Factory
callbacks select a stock category or story chapter. Declared funds, inventory and part fixtures
expose the text being checked; the screen driver captures and grades the native displays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

from core.validation.validate_integers import is_integer

if TYPE_CHECKING:
    from verification.screens.register_screen_programs import ScreenProgram
    from verification.screens.run_screen_checks import ScreenActions
from verification.screens.programs.controller_sequences import (
    FULL_INVENTORY,
    UPGRADE_PARTS,
    catch_drawn,
    walk,
)

SCRIPTS: dict[str, ScreenProgram] = {}


def select_sale_category(d: ScreenActions, category_down: Literal[0, 1]) -> None:
    """
    Apply the inventory fixture and select a category in the native Sell menu.

    Args:
        d (ScreenActions): Driver at the weapon-shop start. The calling screen
            program must declare FULL_INVENTORY in its registered fixtures to
            document the inventory override.
        category_down (Literal[0, 1]): Items 0 or Arms 1; the caller confirms the
            category after positioning the cursor and clearing any earlier trace.

    Raises:
        ValueError: If category_down is a boolean, a noninteger, or not 0 or 1.
    """
    # Stock every saleable item before opening Sell; the native category
    # and quantity menus still decide which records and descriptions appear.
    if not is_integer(category_down) or category_down not in (0, 1):
        raise ValueError("Sale category must be Items (0) or Arms (1)")

    fixture = FULL_INVENTORY
    d.poke(fixture["address"], fixture["data"], fixture["region"], fixture["why"])
    d.run(20)
    d.press("down", 4, 40)
    d.press("a", 4, 120)
    for _ in range(category_down):
        d.press("down", 4, 40)


def create_sale_list_capture(
    category_down: Literal[0, 1], steps: int, tag: str
) -> Callable[[ScreenActions], None]:
    """
    Create a callback that captures each highlighted sell-list entry.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        category_down (Literal[0, 1]): Down presses selecting Items (0) or Arms
            (1) in the Sell category menu.
        steps (int): Number of list positions to capture, including the
            initially highlighted entry.
        tag (str): Category label used in capture labels and capture
            descriptions.

    Returns:
        Callable[[ScreenActions], None]: Callback that captures successive sell-list
            entries in the configured category.

    Raises:
        ValueError: If the sale category is neither Items (0) nor Arms (1), or the step
            count is not positive.
            Also raised if category_down or steps is a boolean or noninteger.
    """
    if not is_integer(steps):
        raise ValueError("steps must be an integer, not a boolean")

    if not is_integer(category_down) or category_down not in (0, 1) or steps <= 0:
        raise ValueError(
            "Sale-list capture requires a valid category and positive step count"
        )

    def run(d: ScreenActions) -> None:
        """
        Walk the configured shop sale list and capture each selection.

        Args:
            d (ScreenActions): Driver at the weapon-shop start, used to submit
                sale-menu inputs and record the list.
        """
        # The helper writes FULL_INVENTORY to game memory and selects Items or
        # Arms. Browse that category so the game draws each entry's description
        # and stats for the highlighted selection.
        select_sale_category(d, category_down)

        d.clear()
        d.press("a", 4, 120)  # the list, first entry highlighted with its description
        d.capture(f"{tag}-000", f"Sell {tag}: list top")

        for i in range(1, steps):
            d.clear()
            d.press("down", 4, 40)
            d.capture(f"{tag}-{i:03d}", f"Sell {tag}: cursor {i}")

    return run


SCRIPTS["shop-sell-arms"] = {
    "state": "weapon-shop",
    "run": create_sale_list_capture(1, 95, "arms"),
    "fixtures": [FULL_INVENTORY],
    "description": "Weapon shop > Sell > Arms with every weapon owned: each weapon highlighted (name list, description, stats).",
}


SCRIPTS["shop-sell-items"] = {
    "state": "weapon-shop",
    "run": create_sale_list_capture(0, 40, "items"),
    "fixtures": [FULL_INVENTORY],
    "description": "Weapon shop > Sell > Items with every item owned: each item highlighted (name list, description).",
}


def capture_garage_upgrades(d: ScreenActions) -> None:
    """
    Capture garage Upgrade menus with parts available.

    Args:
        d (ScreenActions): Driver restored to the riff-garage start.
    """
    f = UPGRADE_PARTS
    d.poke(f["address"], f["data"], f["region"], f["why"])
    d.run(20)
    d.press("down", 4, 40)
    d.press("down", 4, 40)
    d.clear()
    d.press("a", 4, 120)  # Upgrade list, first part and its description
    d.capture("up-00", "garage Upgrade: list top")

    # Clear each prior draw before moving the cursor, so captures are
    # attributed to the newly selected part rather than accumulated list text.
    for i in range(1, 30):
        d.clear()
        d.press("down", 4, 40)
        d.capture(f"up-{i:02d}", f"garage Upgrade: cursor {i}")


SCRIPTS["garage-upgrade"] = {
    "state": "riff-garage",
    "run": capture_garage_upgrades,
    "fixtures": [UPGRADE_PARTS],
    "description": "Riff garage > Upgrade with one of each upgrade part; each part highlighted (name list, description).",
}


def capture_garage_options(d: ScreenActions) -> None:
    """
    Capture the main garage menu choices.

    Visits Add, each armor type in Swap, Upgrade and Quit.

    Args:
        d (ScreenActions): Driver restored to the riff-garage start.
    """
    d.run(20)

    # Visit Add first, then return and enter Swap. The later menu walk
    # continues to the remaining garage options from that same native cursor state.
    walk(d, [("a", 120), ("b", 120), ("down", 40), ("a", 120)], "g")

    for i in range(6):
        d.clear()
        d.press("down", 4, 40)
        d.capture(f"swap-{i}", f"garage Swap: armor type {i + 1}")

    walk(
        d,
        [("b", 120), ("down", 40), ("a", 120), ("b", 120), ("down", 40), ("a", 150)],
        "g2",
    )


def capture_shop_options(d: ScreenActions) -> None:
    """
    Capture the shop menu choices and help text.

    Visits Buy/Items, a Repair-L purchase, the Arms mount picker, Sell and Leave.

    Args:
        d (ScreenActions): Driver restored to the weapon-shop start.
    """
    d.run(20)

    # Follow Buy/Items through confirmation, then return through the Arms
    # mount picker, Sell and Leave. Waits allow each native redraw to finish.
    walk(
        d,
        [
            ("a", 120),
            ("a", 120),
            ("a", 120),
            ("a", 150),
            ("a", 150),
            ("b", 120),
            ("b", 120),
            ("down", 40),
            ("a", 120),
            ("b", 120),
            ("b", 120),
            ("down", 40),
            ("a", 120),
            ("a", 120),
            ("a", 150),
            ("a", 150),
            ("a", 150),
            ("b", 120),
            ("b", 120),
            ("b", 120),
            ("down", 40),
            ("a", 200),
        ],
        "m",
    )


SCRIPTS["garage-menus"] = {
    "state": "riff-garage",
    "run": capture_garage_options,
    "description": "Riff garage: Add, Swap (each armor type), Upgrade and Quit screens.",
}


SCRIPTS["shop-menus"] = {
    "state": "weapon-shop",
    "run": capture_shop_options,
    "description": "Weapon shop: Buy/Items/quantity/purchase, Arms slot picker, Sell/quantity/sale, Leave.",
}


def capture_weapon_swap_lists(d: ScreenActions) -> None:
    """
    Capture garage Swap lists and weapon descriptions.

    Purchases each armor type, then reopens Swap to capture the fitted armor and highlighted
    description, including Regen.

    Args:
        d (ScreenActions): Driver restored to the riff-garage start.
    """
    d.poke(
        0x1AD6, (999999).to_bytes(3, "little"), 2, "funds 999999 (swap fee 30 G each)"
    )
    d.run(20)
    d.press("down", 4, 40)  # cursor on Swap

    for i in range(1, 6):
        d.press("a", 4, 120)

        for k in range(i):
            d.clear()
            d.press("down", 4, 40)
            d.capture(f"t{i}-hl{k}", f"Swap list, type {k + 1} highlighted")

        d.press("a", 4, 120)  # buy: back to the garage menu
        d.clear()
        d.press("a", 4, 120)
        d.capture(f"t{i}-now", f"Swap reopened after fitting armor type {i}")
        d.press("b", 4, 120)


SCRIPTS["garage-swap-all"] = {
    "state": "riff-garage",
    "run": capture_weapon_swap_lists,
    "description": "Riff garage Swap: fit each armor type (native purchase) and reopen; Now: label and descriptions.",
    "fixtures": ["$7E:1AD6 funds = 999999"],
}


def create_sale_capture(
    category_down: Literal[0, 1], tag: str
) -> Callable[[ScreenActions], None]:
    """
    Create a controller callback that captures the native sale-confirmation sequence.

    The callback chooses Items or Arms, selects the first available entry, enters a quantity and
    captures the confirmation. category_down selects which sale list the shared sequence opens.

    Args:
        category_down (Literal[0, 1]): Down presses selecting Items (0) or Arms
            (1) in the Sell category menu.
        tag (str): Category label used in capture labels and capture
            descriptions.

    Returns:
        Callable[[ScreenActions], None]: ScreenActions callback for the configured stock/category
            and transaction captures.

    Raises:
        ValueError: If category_down is a boolean, a noninteger, or not 0 or 1.
    """

    if not is_integer(category_down) or category_down not in (0, 1):
        raise ValueError("Sale category must be Items (0) or Arms (1)")

    def run(d: ScreenActions) -> None:
        """
        Capture the configured category's sale confirmations.

        Args:
            d (ScreenActions): Driver at the weapon-shop start; sale inputs and
                resulting captures update its runner.
        """
        select_sale_category(d, category_down)

        d.clear()
        d.press("a", 4, 150)
        d.capture("list", f"sell {tag} list")
        d.clear()
        d.press("a", 4, 150)
        d.capture("qty", f"sell {tag} quantity")
        d.press("a", 4, 0)

        # Sample immediately after confirmation: the acknowledgement can be
        # replaced by the list before a single long released wait would capture it.
        for k in range(30):
            d.run(4)
            d.capture(
                f"after-{k:02d}", f"after confirming the sale, +{7 * (k + 1)} frames"
            )

        # leaving the list after a sale: the shop script ($93:8654/$93:A3D4) branches on $1494 = 4
        walk(d, [("b", 150), ("a", 150), ("b", 150), ("b", 150)], "leave")

    return run


SCRIPTS["shop-sell-items-once"] = {
    "state": "weapon-shop",
    "run": create_sale_capture(0, "items"),
    "fixtures": [FULL_INVENTORY],
    "description": "Weapon shop: sell one item; quantity screen and thank-you line.",
}


SCRIPTS["shop-sell-arms-once"] = {
    "state": "weapon-shop",
    "run": create_sale_capture(1, "arms"),
    "fixtures": [FULL_INVENTORY],
    "description": "Weapon shop: sell one weapon; quantity screen and thank-you line.",
}


def capture_redrawn_garage_menu(d: ScreenActions) -> None:
    """
    Capture the garage menu immediately after it is redrawn.

    Args:
        d (ScreenActions): Driver restored to the riff-garage start.
    """
    d.run(20)
    d.press("a", 4, 120)  # Add screen
    d.clear()
    d.run(4, ("b",))  # back: the menu is redrawn, then Riff's line
    catch_drawn(d, "09B248")


def capture_purchase_quantity(d: ScreenActions) -> None:
    """
    Capture purchase-quantity labels immediately after drawing.

    Args:
        d (ScreenActions): Driver restored to the weapon-shop start.
    """
    d.run(20)
    d.press("a", 4, 120)  # Buy -> Items / Arms
    d.press("a", 4, 120)  # Items list
    d.clear()
    d.run(4, ("a",))  # an item: quantity screen + "How many will you buy?"
    catch_drawn(d, "09851B")


SCRIPTS["garage-menu-drawn"] = {
    "state": "riff-garage",
    "run": capture_redrawn_garage_menu,
    "description": "Riff garage: back from Add, captured right after the menu (09B248) is drawn.",
}


SCRIPTS["shop-qty-drawn"] = {
    "state": "weapon-shop",
    "run": capture_purchase_quantity,
    "description": "Weapon shop Buy > Items > item: captured right after the quantity labels are drawn.",
}


def capture_purchase_thanks(d: ScreenActions) -> None:
    """
    Capture the shopkeeper's thank-you line during an item purchase.

    Args:
        d (ScreenActions): Driver restored to the weapon-shop start.
    """
    d.run(20)
    d.press("a", 4, 120)  # Buy
    d.press("a", 4, 120)  # Items
    d.clear()
    d.press("a", 4, 0)  # an item

    for k in range(24):
        d.run(10)
        d.capture(f"t-{k:02d}", f"after choosing the item, +{13 * (k + 1)} frames")

    d.press("a", 4, 0)  # confirm the purchase

    for k in range(24):
        d.run(10)
        d.capture(f"c-{k:02d}", f"after confirming, +{13 * (k + 1)} frames")


SCRIPTS["shop-buy-thanks"] = {
    "state": "weapon-shop",
    "run": capture_purchase_thanks,
    "description": "Weapon shop: buy an item; the thanks line (09844B) as it appears.",
}


def create_weapon_stock_capture(chapter: int) -> Callable[[ScreenActions], None]:
    """
    Create a callback that browses the four mount lists for one stock selector.

    Integer inputs accept subclasses; booleans are rejected.

    Args:
        chapter (int): Stock-list selector written to SRAM 0x5C; supported
            lists are 0-3.

    Returns:
        Callable[[ScreenActions], None]: ScreenActions callback browsing each mount's weapon stock
            for the selected chapter, sampling the first position and 24 downward
            moves per mount. The callback applies stock/funds fixtures and captures
            the resulting native list; it does not independently prove every
            possible weapon was present.

    Raises:
        ValueError: If the weapon-stock chapter is outside 0 through 3.
            Also raised if chapter is a boolean or noninteger.
    """
    if not is_integer(chapter):
        raise ValueError("chapter must be an integer, not a boolean")

    if not 0 <= chapter < 4:
        raise ValueError("Weapon stock chapter must be between 0 and 3")

    def run(d: ScreenActions) -> None:
        """
        Browse and capture each mount's weapon stock for the configured chapter.

        Args:
            d (ScreenActions): Driver at the weapon-shop start; stock and funds
                fixtures are applied before browsing the four mount lists.
        """
        d.poke(
            0x5C,
            [chapter],
            0,
            f"Shop stock fixture $70:005C = {chapter} (the stock list $93:96BA reads).",
        )
        d.poke(0x1AD6, (999999).to_bytes(3, "little"), 2, "Funds fixture 999999.")
        d.run(20)
        d.press("down", 4, 40)  # Arms
        d.press("a", 4, 120)  # the mount picker

        for m, moves in enumerate(([], ["right"], ["down"], ["left"])):
            for b in moves:
                d.press(b, 4, 60)

            d.clear()
            d.press("a", 4, 120)  # the mount's stock list
            d.capture(f"m{m}-00", f"mount {m}: list top")

            for i in range(1, 25):
                d.clear()
                d.press("down", 4, 40)
                d.capture(f"m{m}-{i:02d}", f"mount {m}: cursor {i}")

            d.press("b", 4, 90)  # back to the mount picker

    return run


for _c in range(4):
    SCRIPTS[f"shop-buy-arms-{_c}"] = {
        "state": "weapon-shop",
        "run": create_weapon_stock_capture(_c),
        "description": f"Weapon shop > Buy > Arms, stock list {_c}: every mount and weapon.",
    }
