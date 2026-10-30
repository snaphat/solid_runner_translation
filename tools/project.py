"""
This script parses the build, check, and validate commands and dispatches them to the corresponding project workflows.

build.bat forwards its arguments here. The build command compiles a ROM, check
validates the existing build, and validate runs the complete rebuild and gameplay
workflow. Commands run from the repository root and report failures as nonzero
process exit statuses.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from pipeline import run_project_commands as operations
from pipeline.run_project_commands import ENGLISH


def run_cli() -> None:
    """
    Parse project arguments and run the requested build/check/validate workflow.

    Change the process working directory to the repository. Output-directory
    creation belongs to the selected workflow; parsing help creates no outputs.
    Arguments come from sys.argv. build accepts --out; check accepts --emulator
    and --gameplay/--scenes; validate accepts --workers and --out. validate exits
    with its returned status, while build/check failures propagate to the module
    entry point, which prints the error and exits with status one.

    Raises:
        ValueError: Python is older than 3.11 or the selected workflow fails validation.
        OSError: Repository, dependency, or build files cannot be accessed.
        subprocess.CalledProcessError: A required child workflow exits nonzero.
        SystemExit: Argument parsing displays help/rejects arguments, or validate
            finishes with its workflow status.
    """
    os.chdir(ROOT)

    if sys.version_info < (3, 11):
        raise ValueError("Python 3.11 or newer is required; 64-bit Python 3.13 is tested.")

    # Each subcommand owns its options; build/check dispatch through the
    # project operations layer, while validate runs the staged full pipeline.
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build").add_argument("--out", default=str(ENGLISH))
    checking = commands.add_parser("check")
    checking.add_argument("--emulator", action="store_true")
    checking.add_argument(
        "--gameplay",
        "--scenes",
        dest="scenes",
        action="store_true",
        help="also run calibrated quick gameplay verification and write build/verification/report/index.html",
    )
    validating = commands.add_parser(
        "validate", help="rebuild and run all checks, both-ROM replays and target audits"
    )
    validating.add_argument(
        "--workers", type=int, default=12, help="parallel emulators, 1-12 (default: 12)"
    )
    validating.add_argument(
        "--out", type=Path, help="new directory for logs and results (default: under build/)"
    )

    # Parse once, then return the selected operation's exit status to the shell.
    args = parser.parse_args()
    if args.command == "validate":
        from pipeline.run_full_validation import run_full_validation

        sys.exit(run_full_validation(args.workers, args.out))

    {"build": operations.build_rom, "check": operations.check_project}[args.command](args)


if __name__ == "__main__":
    try:
        run_cli()
    except (ValueError, OSError, subprocess.CalledProcessError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
