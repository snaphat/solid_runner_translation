"""
This module reads the original Japanese ROM, verifies its hash, and rejects output paths that point to that input file.

read_original_rom() reads rom/Solid Runner (Japan).sfc and compares its SHA-256
with the required digest. A mismatch raises ValueError. A successful read returns
immutable bytes and caches them; later calls return the cached bytes without
rereading the file. Build and validation code use these bytes as their unchanged
Japanese source image.

resolve_output_path() takes a proposed output filename and returns its resolved
path. It requires an .sfc extension and rejects paths that resolve to the input
ROM or are existing hard links to it. It checks a destination; it does not create
or write the output file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib
import hashlib
import os
from functools import cache
from pathlib import Path

# Required location of the unmodified, unheadered Japanese input ROM.
PATH = Path(__file__).resolve().parents[3] / "rom" / "Solid Runner (Japan).sfc"

# Exact accepted input-image digest; read_original_rom() rejects modified or headered ROMs.
SHA256 = "c8f453cba9f638967f7cea41a95106ac675a33861f5d3414fff9ffb44ec2f3d8"


@cache
def read_original_rom() -> bytes:
    """
    Read and hash-check the original Japanese ROM on the first successful call.

    Later calls return the cached bytes even if the file has changed. Failed
    reads or hash checks are not cached. This function never writes the input.

    Returns:
        bytes: Cached original Japanese ROM bytes with the required SHA-256.

    Raises:
        ValueError: The original Japanese ROM SHA-256 differs from the required
            value.
        OSError: The original ROM cannot be opened or read.
    """
    rom = PATH.read_bytes()
    if hashlib.sha256(rom).hexdigest() != SHA256:
        raise ValueError(
            "Original ROM SHA-256 differs. Supply the unmodified, unheadered Japanese ROM; see README.md."
        )

    return rom


def resolve_output_path(value: str | os.PathLike[str]) -> pathlib.Path:
    """
    Resolve a compiler destination while protecting the original ROM.

    This is a path check, not a reservation: filesystem links can change after
    it returns. The caller remains responsible for creating parent directories
    and writing the output.

    Args:
        value (str | os.PathLike[str]): Requested .sfc destination; must
            not resolve or hard-link to the original ROM.

    Returns:
        pathlib.Path: Resolved .sfc destination distinct from the protected
            original ROM. The output need not exist; write permission is not
            tested. Existing paths are inspected to reject aliases of the input.

    Raises:
        ValueError: The destination is not .sfc or aliases the protected
            original ROM.
        OSError: Path resolution or an existing-file identity check fails.
    """
    # Resolve path aliases first; samefile below also catches hard links to
    # the protected input ROM.
    target = Path(value).resolve()
    if (
        target.suffix.lower() != ".sfc"
        or target == PATH.resolve()
        or (target.exists() and PATH.exists() and os.path.samefile(target, PATH))
    ):
        raise ValueError("Choose a .sfc output path other than the original ROM or an alias of it.")

    return target
