"""
This command reads a compiled English ROM and reports invalid text jumps, terminators, and name records.

The command reads the selected image and checks its active translation entries through the
shared compiled-text reader. It reports malformed text chains, incorrect terminators and invalid
name-copy data through its exit status.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.text import validate_compiled_text as rom_text


def run_cli(path: str | os.PathLike[str]) -> int:
    """
    Read a compiled ROM and report its text-verification status.

    Args:
        path (str | os.PathLike[str]): Compiled ROM file to read and
            validate.

    Returns:
        int: Compiled-text verifier exit status: zero on success, one on
            reported structural failure. Other input/setup errors propagate.

    Raises:
        OSError: If the ROM or a translation source cannot be read.
        ValueError: If source metadata or reference resolution fails.
        IndexError: If a stored reference operand lies outside the ROM.
    """
    return rom_text.verify_compiled_text(Path(path).read_bytes())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: verify_compiled_text.py <rom.sfc>")

    sys.exit(run_cli(sys.argv[1]))
