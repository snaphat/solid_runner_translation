"""
This command measures English prose and reports text that exceeds its display-window limits.

Read-only checks use the compiler's configured or inferred entry widths and row limits. Apply
mode instead checks proposed category-derived widths and saves them to translation metadata.
Both modes report oversized words and excessive rows; read-only checking leaves source files
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.files import replace_related_files as source_files
from core.text import get_text_limits as text_limits
from core.text import layout_text as text_layout
from core.text import measure_text as text_measurement
from core.text import read_translation_files as script_files
from core.text import validate_translation as script_rules


def run_cli(apply: bool = False) -> int:
    """
    Check classified non-paginated prose in the default compiled English ROM.

    The command reads system/events sources and checks entries classified as
    non-paginated prose. Read-only mode uses the compiler's current entry width and row
    allowance, including w and max_lines metadata. Apply mode substitutes proposed
    widths derived from Japanese category text, with category and entry overrides, while
    retaining the compiler's row allowance. The compiled-ROM path is build/Solid Runner
    (English).sfc relative to the process working directory.

    Boolean options require bool values; numeric zero and one are rejected.

    Args:
        apply (bool, optional): Save derived widths to translation metadata as
            well as reporting violations. Writes both system and events collections,
            including when violations remain. Entries failing word-fit are not
            assigned a width; row-count failures can still receive their derived width.
            All four source/metadata files are formatted before any replacement.
            Replacement failures trigger rollback across both collections. A process
            or operating-system crash can still leave a partial update; failed
            rollback retains recovery files. Cleanup failures after replacement
            leave the updated files in place.
            Defaults to False.

    Returns:
        int: Zero when checked windows fit, one if any word/row violations are reported.

    Raises:
        TypeError: If apply is not a boolean.
        OSError: If a ROM/source file cannot be read, or apply-mode replacement,
            rollback or temporary-file cleanup fails.
        ValueError: If source metadata, markup or required insertion measurements are
            invalid, or apply-mode destinations repeat or have no common parent path.
        AssertionError: If apply-mode formatting encounters an English line beginning
            with the reserved # comment prefix.
        KeyError: If English text contains a character with no game encoding.
    """
    if type(apply) is not bool:
        raise TypeError("Boolean options require bool values: apply")
    rom = bytearray(Path("build/Solid Runner (English).sfc").read_bytes())
    ev = script_files.load("system")
    evs = script_files.load("events")
    allp = [(e, "system") for e in ev] + [(e, "events") for e in evs]

    # Accumulate category widths from Japanese source, then apply the explicit
    # per-category and per-entry limits before checking English layout.
    width: dict[str, int] = {}
    for e, _ in allp:
        c = text_limits.classify_text_window(int(e["id"], 16))
        if c and e.get("en") and not e.get("skip"):
            w = max(
                text_limits.estimate_japanese_line_width(l.strip("　"))
                for l in e["jp"].split("\n")
            )
            width[c] = max(width.get(c, 0), w)

    names = [e["en"] for e in ev if e.get("mode") == "n" and e.get("en") and not e.get("skip")]
    name_width = text_measurement.measure_name_width(rom, names) if names else None

    for c in width:
        width[c] = min(text_limits.MAXW, max(width[c], 96))

    width.update(text_limits.OVERRIDE)
    print("window widths:", width)
    bad: list[tuple[str, str, int, str | int, str]] = []

    # Only unpaginated prose must fit a single native window. Dialogue can
    # continue onto another page and is validated by its own layout rules.
    for e, _ in allp:
        c = text_limits.classify_text_window(int(e["id"], 16))
        if not c or not e.get("en") or e.get("skip") or e.get("mode") != "p":
            continue

        # Read-only checks use the compiler's current width and row allowance.
        # Apply mode checks a proposed width before assigning it to the source entry.
        candidate = e.copy()
        if apply:
            candidate["w"] = text_limits.ENTRY_W.get(e["id"], width[c])
        constraints = script_rules.derive_entry_layout_constraints(candidate)
        w, jl = constraints["width"], constraints["lines"]
        if w is None or jl is None:
            raise ValueError("Classified prose must have a width and a visible row limit")
        try:
            laid = text_layout.wrap_text(rom, e["en"], w, paginate=False, name_width=name_width)
        except AssertionError:
            bad.append((c, e["id"], jl, "word too wide", e["en"]))
            continue

        # Count newline tokens rather than zero bytes, since zero can also be
        # an operand inside a text command or an extended glyph.
        el = sum(1 for t, _ in text_measurement.iter_text_tokens(laid) if t == bytes([0])) + 1
        if el > jl:
            bad.append((c, e["id"], jl, el, e["en"]))

        if apply:
            e["w"] = w

    for b in bad:
        print("%-20s %s limit %s lines, EN %s | %s" % b)

    print("still too tall:", len(bad))

    # Applying inferred widths updates technical metadata in both collections;
    # without apply this command only reports layout failures.
    if apply:
        system_text, system_meta = script_files.resolve_collection_paths("system")
        events_text, events_meta = script_files.resolve_collection_paths("events")
        # Format every file before replacing any collection, then use one rollback
        # operation so a failed later replacement restores the earlier files too.
        updates = [
            (system_text, script_files.format_text("system", ev).encode("utf-8")),
            (system_meta, script_files.format_meta(ev).encode("utf-8")),
            (events_text, script_files.format_text("events", evs).encode("utf-8")),
            (events_meta, script_files.format_meta(evs).encode("utf-8")),
        ]
        source_files.replace_many(updates)

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run_cli("apply" in sys.argv))
