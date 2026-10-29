"""
This module runs Asar and assembles a temporary copy of a candidate ROM.

assemble_final_image() takes a 4 MiB ROM buffer, writes a temporary copy, runs
Asar with checksum fixing enabled, and returns the assembled bytes. It leaves
the supplied buffer unchanged and rejects assembler errors or changed image sizes.

Fixed-byte patch extraction and installation are provided by
core.rom.apply_assembly_patches.

The module locates Asar through ASAR or external/asar/asar.exe, captures its
output, and removes temporary files under build/ when their contexts exit.
Source and image arguments are relative to the repository when on its drive.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# Repository root used to resolve assembly sources, dependency paths and
# build-local scratch directories.
ROOT = Path(__file__).resolve().parents[3]
SIZE = 0x400000  # Expanded unheadered LoROM image size in bytes (4 MiB).


def resolve_asar_path() -> str:
    """
    Return the configured Asar executable path without checking the executable.

    Returns:
        str: Executable path from ASAR, or the repository's external/asar/asar.exe
            default. This function does not check existence or binary integrity.
    """
    return os.environ.get("ASAR", str(ROOT / "external" / "asar" / "asar.exe"))


def create_scratch_directory(prefix: str) -> tempfile.TemporaryDirectory[str]:
    """
    Stage assembler files below the project so their CLI paths can stay relative.

    Args:
        prefix (str): Prefix for a temporary directory created below build/.

    Returns:
        tempfile.TemporaryDirectory[str]: Context manager whose directory is removed
            on exit.

    Raises:
        ValueError: The prefix contains a path separator or drive delimiter.
    """
    if "/" in prefix or "\\" in prefix or ":" in prefix:
        raise ValueError("Scratch directory prefix must not contain path separators")
    directory = ROOT / "build"
    directory.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=directory)


def run_asar(
    source: str | os.PathLike[str], image: str | os.PathLike[str], *options: Any
) -> subprocess.CompletedProcess[str]:
    """
    Run Asar on a source file and scratch ROM.

    Run with the repository as the working directory and capture stdout/stderr.
    Source and image arguments use paths relative to that directory when they
    are on its drive; paths on other drives remain absolute.

    Args:
        source (str | os.PathLike[str]): Asar assembly source file.
        image (str | os.PathLike[str]): Scratch ROM image that Asar may
            modify.
        *options: Additional Asar command-line arguments, each passed as one
            argument.

    Returns:
        subprocess.CompletedProcess[str]: Asar exit status and captured UTF-8
            stdout/stderr; nonzero status is not raised here.
    """

    def format_path_argument(path: str | os.PathLike[str]) -> str:
        """
        Express a subprocess file path relative to the repository when possible.

        Args:
            path (str | os.PathLike[str]): Assembler input or output path;
                another Windows drive falls back to a resolved absolute path.

        Returns:
            str: Relative path from ROOT, or an absolute path when drives differ.
        """
        # Prefer repository-relative arguments so Asar need not interpret a possibly
        # non-ASCII absolute workspace path.
        try:
            return os.path.relpath(path, ROOT)
        except ValueError:  # Windows cannot form a relative path across drives.
            return str(Path(path).resolve())

    return subprocess.run(
        [resolve_asar_path(), *options, format_path_argument(source), format_path_argument(image)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def assemble_final_image(
    rom: bytes | bytearray, source: str | os.PathLike[str] | None = None
) -> bytes:
    """
    Assemble a temporary ROM copy with checksum fixing enabled and return its bytes.

    Args:
        rom (bytes | bytearray): 4 MiB pre-assembly candidate; assembly
            modifies a scratch copy.
        source (str | os.PathLike[str] | None, optional): Assembly
            entry file; None selects asm/vwf.asm.

    Returns:
        bytes: Asar's 4 MiB output image, assembled with checksum fixing enabled.

    Raises:
        ValueError: Asar fails, or the candidate/final image has the wrong
            size.
    """
    if len(rom) != SIZE:
        raise ValueError(f"Invalid pre-assembly ROM size: {len(rom)}; expected {SIZE}")

    source = Path(source) if source is not None else ROOT / "asm/vwf.asm"

    # Assemble a disposable candidate; a failed assembly must not overwrite the
    # caller's ROM image.
    with create_scratch_directory("solid-runner-asar-") as folder:
        image = Path(folder) / "candidate.sfc"
        image.write_bytes(rom)
        result = run_asar(source.resolve(), image, "--fix-checksum=on")
        if result.returncode:
            raise ValueError(f"Asar failed on {source.name}:\n{result.stdout}{result.stderr}")

        assembled = image.read_bytes()
        if len(assembled) != SIZE:
            raise ValueError(f"Invalid assembled ROM size: {len(assembled)}; expected {SIZE}")

        return assembled
