"""
This module implements the build and check commands by preparing dependencies and running the required compiler or validation processes.

The build command validates the original ROM, selects the pinned Asar binary and
invokes compilation. The check command verifies an existing build, runs unit
tests and optionally boots the emulator or replays gameplay. Subprocess failures
propagate to the public command instead of leaving a successful exit status.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import argparse
    import pathlib
import os
import struct
import subprocess
import sys

from core.rom import load_original_rom as original_rom
from pipeline.setup import install_dependencies as dependencies

ROOT = dependencies.ROOT
ROM = original_rom.PATH
ORIGINAL_SHA256 = original_rom.SHA256
ENGLISH = ROOT / "build/Solid Runner (English).sfc"

# Each check must exit cleanly and print its line of success.
CHECKS = (
    ("audit_rom_changes", "0 problems"),
    ("verify_compiled_text", "bad 0"),
    ("check_text_windows", "still too tall: 0"),
    ("check_translation_rules", "script rules: 0 problems"),
)


def run_subprocess(
    args: Sequence[str | os.PathLike[str]],
    cwd: str | os.PathLike[str] = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    """
    Run one child process and require a zero exit status.

    Arguments are stringified and passed directly without a shell. Standard
    input, output, and error are inherited; no output is captured or returned.

    Args:
        args (Sequence[str | os.PathLike[str]]): Process argument vector,
            without shell quoting or a shell command string.
        cwd (str | os.PathLike[str], optional): Child working directory;
            defaults to the repository root.
        env (dict[str, str] | None, optional): Complete child environment
            mapping; None inherits the current environment.

    Raises:
        subprocess.CalledProcessError: The child process exits with a nonzero
            status.
        OSError: The executable cannot be started or cwd cannot be used.
    """
    subprocess.run([str(a) for a in args], cwd=cwd, env=env, check=True)


def validate_original_rom() -> pathlib.Path:
    """
    Return the original ROM path after checking its existence and required SHA-256.

    Returns:
        pathlib.Path: Original Japanese ROM path after existence and hash
            validation.

    Raises:
        ValueError: The original Japanese ROM is missing or has the wrong SHA-256.
        OSError: The input cannot be opened or read for hashing.
    """
    if not ROM.is_file():
        raise ValueError(
            f"Place your original Japanese ROM at {ROM}. See README.md for its required SHA-256."
        )

    if dependencies.hash_file(ROM) != ORIGINAL_SHA256:
        raise ValueError(
            "Original ROM SHA-256 differs. Supply the unmodified, unheadered Japanese ROM; see README.md."
        )

    return ROM


def build_rom(args: argparse.Namespace) -> None:
    """
    Resolve Asar and compile to args.out after validating the original ROM.

    Args:
        args (argparse.Namespace): Parsed build options; out is the requested
            ROM destination, checked against the protected original input path.
            Parent directories are created by the compiler, not this dispatcher.

    Raises:
        ValueError: Original-ROM, dependency, or destination checks fail.
        OSError: Input/dependency/output files cannot be accessed or a child cannot start.
        subprocess.CalledProcessError: Compilation exits nonzero.
    """
    validate_original_rom()
    asar = dependencies.install_dependency("asar")
    out = original_rom.resolve_output_path(args.out)

    # Pass the resolved assembler and UTF-8 mode to the build subprocess;
    # the caller environment and repository source files remain unchanged.
    env = dict(os.environ, ASAR=str(asar), PYTHONUTF8="1")
    run_subprocess(
        [sys.executable, ROOT / "tools/pipeline/compilation/compile_rom.py", out], env=env
    )
    print(f"Built {out}\nSHA-256 {dependencies.hash_file(out)}")


def check_project(args: argparse.Namespace) -> None:
    """
    Run static, unit and requested emulator/gameplay checks.

    Args:
        args (argparse.Namespace): Parsed check options; emulator enables smoke
            checks and scenes enables quick gameplay verification. Either option
            first runs boot/naming smoke checks. Both must be present attributes.
            Checks always use the default ENGLISH build path, not a custom build output.

    Raises:
        ValueError: No build is present, a required check fails, or emulator
            checks use an incompatible Python architecture.
        OSError: Source/build/dependency files cannot be read or a child cannot start.
        subprocess.CalledProcessError: Unit tests, smoke checks, or gameplay verification
            exit nonzero.
    """
    validate_original_rom()

    if not ENGLISH.is_file():
        raise ValueError("Run build.bat first.")

    env = dict(os.environ, ASAR=str(dependencies.install_dependency("asar")), PYTHONUTF8="1")

    # Require both process success and the check's success summary, so a tool
    # that exits before completing its audit cannot pass silently.
    for tool, expected in CHECKS:
        command = [sys.executable, ROOT / f"tools/checks/{tool}.py"]
        if tool in ("audit_rom_changes", "verify_compiled_text"):
            command.append(ENGLISH)

        result = subprocess.run(
            [str(v) for v in command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
        )
        print(result.stdout, end="")

        if result.returncode or expected not in result.stdout.splitlines():
            raise ValueError(f"{tool} check failed")

    run_subprocess(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"], env=env
    )

    if args.emulator or args.scenes:
        if struct.calcsize("P") != 8:
            raise ValueError("The pinned Snes9x core needs 64-bit Python.")

        env = dict(env, SOLID_RUNNER_CORE=str(dependencies.install_dependency("snes9x")))
        run_subprocess(
            [sys.executable, ROOT / "tools/verification/checks/check_boot_and_naming.py"], env=env
        )

        if args.scenes:
            # Quick verification combines calibrated displays with native tasks.
            run_subprocess(
                [sys.executable, ROOT / "tools/verification/verify_gameplay.py"], env=env
            )
