"""
This module runs Asar on assembly source files to produce a modified ROM or extract bytes for an individual repair.

assemble_final_image() takes a 4 MiB ROM buffer, writes a temporary copy, runs
Asar with checksum fixing enabled, and returns the assembled bytes. It leaves
the supplied buffer unchanged and rejects assembler errors or changed image sizes.

assemble_repair_blocks() instead assembles a named repair over two temporary
images filled with 00 and FF. Matching output bytes are collected into blocks
keyed by ROM file offset. Repairs must emit fixed bytes independently of the
input ROM; two fills cannot prove that arbitrary conditional assembly does so.

apply_assembly_repair() checks every block's destination against the original
Japanese ROM, expecting FF beyond the original image's end. It changes the
supplied ROM buffer only after every destination matches. This compares current
bytes; it does not record or detect earlier writes that left those bytes unchanged.

The module locates Asar through ASAR or external/asar/asar.exe, captures its
output, and removes temporary files under build/ when their contexts exit.
Source and image arguments are relative to the repository when on its drive.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# Repository root used to resolve repair sources, dependency paths and
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


def assemble_repair_blocks(name: str) -> dict[int, bytes]:
    """
    Assemble asm/repairs/<name>.asm and return {file offset: written bytes}.

    Repair sources must emit fixed bytes without depending on the input image.
    Assemble into 00-filled and FF-filled temporary images and collect spans
    where both outputs agree. Reject differing output bytes unless they retain
    their respective 00/FF fills. These two inputs can reveal some violations
    of the fixed-byte requirement, but cannot prove that a source never depends
    on ROM contents. Each call runs Asar twice without caching the results.

    Args:
        name (str): Repair path below asm/repairs without .asm, including
            events/ or interface/.

    Returns:
        dict[int, bytes]: Fresh mapping of written block contents keyed by
            unheadered file offset.

    Raises:
        ValueError: name escapes asm/repairs, Asar fails, its output size changes,
            or the probe images reveal an input-dependent write.
    """
    repairs = (ROOT / "asm" / "repairs").resolve()
    source = (repairs / f"{name}.asm").resolve()
    if not name or not source.is_relative_to(repairs):
        raise ValueError("Repair source must remain below asm/repairs")
    images: list[bytes] = []
    with create_scratch_directory("solid-runner-blocks-") as folder:
        # For fixed-byte repairs, matching outputs distinguish emitted bytes from
        # untouched fill, including emitted 00 and FF bytes.
        for fill in (0x00, 0xFF):
            image = Path(folder) / f"{fill:02x}.sfc"
            image.write_bytes(bytes([fill]) * SIZE)
            result = run_asar(source, image, "--fix-checksum=off", "--no-title-check")
            if result.returncode:
                raise ValueError(f"Asar failed on {source.name}:\n{result.stdout}{result.stderr}")

            assembled = image.read_bytes()
            if len(assembled) != SIZE:
                raise ValueError(
                    f"Repair assembly changed ROM size: {len(assembled)}; expected {SIZE}"
                )
            images.append(assembled)

    zeros, ones = images

    # Accept equal output bytes or the unchanged 00/FF fill pair. Other pairs
    # show that the two assembly runs did not produce a fixed repair byte.
    # Search changed spans in C first; most of each 4 MiB image is background.
    for pattern, image, other in (
        (rb"[^\x00]+", zeros, ones),
        (rb"[^\xff]+", ones, zeros),
    ):
        for span in re.finditer(pattern, image):
            if image[span.start() : span.end()] != other[span.start() : span.end()]:
                at = next(
                    offset
                    for offset in range(span.start(), span.end())
                    if image[offset] != other[offset]
                )
                raise ValueError(
                    f"Repair {name!r} has an input-dependent write at {at:06X}; "
                    "repair blocks must emit fixed bytes"
                )

    # XOR is zero only where both assembled images agree. Those spans are the
    # fixed-byte repair spans; untouched fills differ at every byte.
    same = (int.from_bytes(zeros, "big") ^ int.from_bytes(ones, "big")).to_bytes(SIZE, "big")
    return {m.start(): zeros[m.start() : m.end()] for m in re.finditer(rb"\x00+", same)}


def get_repair_ranges(name: str) -> list[tuple[int, int]]:
    """
    Return half-open file-offset write ranges for the named assembly repair.

    Args:
        name (str): Repair path below asm/repairs without .asm, including
            events/ or interface/.

    Returns:
        list[tuple[int, int]]: Half-open file-offset write ranges.
    """
    return [(at, at + len(data)) for at, data in assemble_repair_blocks(name).items()]


def apply_assembly_repair(rom: bytearray, name: str) -> None:
    """
    Apply repair blocks only if every destination matches its expected original bytes.

    Compare each destination with the original Japanese ROM, using FF beyond
    its end. Check all blocks before writing any of them. A mismatch raises an
    error without modifying the candidate. Earlier writes that left the expected
    bytes unchanged are not detected; this function does not track write history.

    Args:
        rom (bytearray): Expanded 4 MiB candidate ROM, modified only after
            all repair regions pass original-byte checks.
        name (str): Repair path below asm/repairs without .asm, including
            events/ or interface/.

    Raises:
        ValueError: The candidate image is not exactly 4 MiB, or repair assembly
            fails validation.
        AssertionError: A destination block differs from the original bytes or
            the expected FF expansion bytes.
    """
    # Reject incomplete or oversized candidates before assembling or changing bytes.
    if len(rom) != SIZE:
        raise ValueError(f"Invalid repair ROM size: {len(rom)}; expected {SIZE}")

    from core.rom import load_original_rom as original_rom

    source = original_rom.read_original_rom()
    blocks = assemble_repair_blocks(name)

    # Check all blocks before writing any, so a later mismatch cannot leave
    # this repair partially applied.
    for at, data in blocks.items():
        span = slice(at, at + len(data))
        original = source[span]
        original += b"\xff" * (len(data) - len(original))
        if not (bytes(rom[span]) == original):
            raise AssertionError((name, "already changed by an earlier step", hex(at)))

    for at, data in blocks.items():
        rom[at : at + len(data)] = data
