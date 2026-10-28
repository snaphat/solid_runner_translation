"""
This command lists compressed graphics resources or exports their tiles as PNG previews.

The scan subcommand prints candidate banks and decoded resource sizes. The
sheets subcommand writes 2bpp and 4bpp tile-sheet images into the requested
directory when the decoded byte count contains complete tiles at that depth.
Resources with incomplete 2bpp tiles are reported on stderr and skipped. Both use the reusable graphics functions in core.graphics; the
command handles argument parsing, original-ROM loading, and console output.

Usage:
    python tools/graphics/inspect_resources.py scan
    python tools/graphics/inspect_resources.py sheets OUTDIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.graphics.compress_graphics import iter_resource_blocks, write_tile_sheet
from core.rom import load_original_rom as original_rom


def run_cli() -> None:
    """
    Parse the scan or sheets command and inspect the original ROM's resources.

    Help and argument errors are handled before the original ROM is loaded.
    Scan prints resource counts; sheets creates its output directory and writes
    preview PNGs for each compatible tile depth. The scanner skips banks whose
    pointer tables or compressed resources fail its checks. Decoded resources
    whose sizes are not multiples of 16 bytes are reported on stderr and skipped.
    ROM-loading and output-file errors propagate to the caller.
    """
    parser = argparse.ArgumentParser(description="Inspect and export native graphics resources.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan", help="List candidate graphics banks and decoded resource sizes")
    sheets = commands.add_parser("sheets", help="Export tile-sheet PNG previews")
    sheets.add_argument("outdir", type=Path, help="Directory for exported previews")
    args = parser.parse_args()
    rom = original_rom.read_original_rom()
    if args.command == "scan":
        for bank, res in iter_resource_blocks(rom):
            print(
                "bank %02X: %d blocks, %d bytes out" % (bank, len(res), sum(len(d) for _, d in res))
            )
    else:
        args.outdir.mkdir(parents=True, exist_ok=True)
        for _, res in iter_resource_blocks(rom):
            for off, data in res:
                # A 2bpp tile occupies 16 bytes; a 4bpp tile occupies 32. The
                # scanner identifies candidate resources, not their tile depth.
                if len(data) % 16:
                    print(
                        "Skipping resource %06X: %d bytes do not form complete tiles"
                        % (off, len(data)),
                        file=sys.stderr,
                    )
                    continue

                if len(data) % 32 == 0:
                    write_tile_sheet(data, 4, args.outdir / ("%06X_4.png" % off))
                write_tile_sheet(data, 2, args.outdir / ("%06X_2.png" % off))


if __name__ == "__main__":
    run_cli()
