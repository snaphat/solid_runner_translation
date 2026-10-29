"""
This module extracts bytes written by assembly patches and installs them in a candidate ROM.

assemble_patch_blocks() runs a named source from asm/patches against 00-filled and FF-filled
scratch images. It returns file offsets and byte blocks where the outputs agree, and rejects
differences other than untouched fill bytes. Sources must emit fixed bytes; these two probes
cannot prove independence from every possible input image. get_assembly_patch_ranges() returns
the extracted spans.

apply_assembly_patch() checks every destination against the original Japanese ROM, expecting FF
in expanded space, before writing any block. The caller's buffer is unchanged when those checks
fail. Asar execution and scratch-file management are provided by core.rom.assemble_rom.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from core.rom import assemble_rom


def _validate_patch_name(name: str) -> None:
    """
    Require a string before resolving a named assembly source.

    Args:
        name (str): Source path relative to asm/patches, without .asm.

    Raises:
        TypeError: If name is not a string.
    """
    if not isinstance(cast(object, name), str):
        raise TypeError("Assembly patch name must be a string")


def assemble_patch_blocks(name: str) -> dict[int, bytes]:
    """
    This function runs an assembly patch through Asar and returns the bytes it writes, grouped by ROM file offset.

    Asar runs twice: once on a temporary ROM file filled with 00 bytes and once on a file filled
    with FF bytes. Matching output bytes identify the patch's writes. Unchanged 00/FF pairs
    identify bytes the patch did not write. Any other difference raises ValueError because the
    patch produced different bytes for the two inputs.

    Patch sources must produce fixed bytes independently of existing ROM contents. These two
    runs detect some violations of that requirement, but cannot prove independence for every
    possible input. Results are not cached.

    Args:
        name (str): Patch path below asm/patches without .asm, including
            events/ or interface/.

    Returns:
        dict[int, bytes]: Fresh mapping of written block contents keyed by
            unheadered file offset.

    Raises:
        TypeError: If name is not a string.
        ValueError: If name escapes asm/patches, Asar fails, its output size changes, or the
            probe images reveal an input-dependent write.
        OSError: If temporary files cannot be created, read, written or removed, or Asar cannot
            be started.
    """
    _validate_patch_name(name)
    patches = (assemble_rom.ROOT / "asm" / "patches").resolve()
    source = (patches / f"{name}.asm").resolve()
    if not name or not source.is_relative_to(patches):
        raise ValueError("Patch source must remain below asm/patches")
    images: list[bytes] = []
    with assemble_rom.create_scratch_directory("solid-runner-blocks-") as folder:
        # For fixed-byte patches, matching outputs distinguish emitted bytes from
        # untouched fill, including emitted 00 and FF bytes.
        for fill in (0x00, 0xFF):
            image = Path(folder) / f"{fill:02x}.sfc"
            image.write_bytes(bytes([fill]) * assemble_rom.SIZE)
            result = assemble_rom.run_asar(source, image, "--fix-checksum=off", "--no-title-check")
            if result.returncode:
                raise ValueError(f"Asar failed on {source.name}:\n{result.stdout}{result.stderr}")

            assembled = image.read_bytes()
            if len(assembled) != assemble_rom.SIZE:
                raise ValueError(
                    f"Patch assembly changed ROM size: {len(assembled)}; expected {assemble_rom.SIZE}"
                )
            images.append(assembled)

    zeros, ones = images

    # Accept equal output bytes or the unchanged 00/FF fill pair. Other pairs
    # show that the two assembly runs did not produce a fixed patch byte.
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
                    f"Patch {name!r} has an input-dependent write at {at:06X}; "
                    "patch blocks must emit fixed bytes"
                )

    # XOR is zero only where both assembled images agree. Those spans are the
    # fixed-byte patch spans; untouched fills differ at every byte.
    same = (int.from_bytes(zeros, "big") ^ int.from_bytes(ones, "big")).to_bytes(
        assemble_rom.SIZE, "big"
    )
    return {m.start(): zeros[m.start() : m.end()] for m in re.finditer(rb"\x00+", same)}


def get_assembly_patch_ranges(name: str) -> list[tuple[int, int]]:
    """
    Return half-open file-offset write ranges for the named assembly patch.

    This function calls assemble_patch_blocks(), which runs Asar on two temporary ROM files
    filled with 00 and FF bytes. It returns ranges in ascending file-offset order; adjacent
    written bytes form one range. Each call assembles the source again.

    Args:
        name (str): Patch path below asm/patches without .asm, including
            events/ or interface/.

    Returns:
        list[tuple[int, int]]: Half-open file-offset write ranges.

    Raises:
        TypeError: If name is not a string.
        ValueError: If the source path leaves asm/patches, Asar fails, assembly changes the ROM
            size, or the two probe images reveal an input-dependent write.
        OSError: If temporary files cannot be created, read, written or removed, or Asar cannot
            be started.
    """
    return [(at, at + len(data)) for at, data in assemble_patch_blocks(name).items()]


def apply_assembly_patch(rom: bytearray, name: str) -> None:
    """
    This function writes an assembly patch into the supplied ROM buffer after checking every destination.

    Destination bytes must match the original Japanese ROM, or FF beyond the original file's
    end. All destinations are checked before any writes occur, so a mismatch leaves the supplied
    buffer unchanged. The check compares byte values, not write history: it cannot detect
    earlier writes that preserved those values.

    Args:
        rom (bytearray): Expanded 4 MiB candidate ROM, modified only after
            all patch regions pass original-byte checks. Other buffer types,
            including memoryview, are rejected.
        name (str): Patch path below asm/patches without .asm, including
            events/ or interface/.

    Raises:
        TypeError: If rom is not a bytearray or name is not a string.
        ValueError: If the candidate image is not exactly 4 MiB, or patch assembly fails
            validation or the required original-ROM hash does not match.
        OSError: If the original ROM or scratch files cannot be accessed, cleanup fails, or Asar
            cannot be started.
        AssertionError: If a destination block differs from the original bytes or the expected
            FF expansion bytes.
    """
    # Validate caller inputs before loading the original ROM or running Asar.
    if not isinstance(cast(object, rom), bytearray):
        raise TypeError("Assembly patch destination must be a bytearray")
    _validate_patch_name(name)
    if len(rom) != assemble_rom.SIZE:
        raise ValueError(f"Invalid patch ROM size: {len(rom)}; expected {assemble_rom.SIZE}")

    from core.rom import load_original_rom as original_rom

    source = original_rom.read_original_rom()
    blocks = assemble_patch_blocks(name)

    # Check all blocks before writing any, so a later mismatch cannot leave
    # this patch partially applied.
    for at, data in blocks.items():
        span = slice(at, at + len(data))
        original = source[span]
        original += b"\xff" * (len(data) - len(original))
        if not (bytes(rom[span]) == original):
            raise AssertionError(f"Patch {name!r}: ROM block at {at:#x} differs from the expected original bytes")

    for at, data in blocks.items():
        rom[at : at + len(data)] = data
