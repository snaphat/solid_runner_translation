"""
This installer downloads the external binaries listed in dependencies.lock.json and verifies their archive and binary hashes.

The installer verifies existing files before reuse. Missing or invalid binaries
are downloaded from their pinned URLs, and archive and binary hashes are checked
before installation. Dependencies remain under external/; they are not source
files or Python package requirements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import pathlib
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class DependencySpec(TypedDict):
    """
    Pinned download, archive member and integrity requirements for one dependency.

    Attributes:
        version (str): Release label displayed during installation.
        url (str): Download URL for the archive containing the binary.
        archive_sha256 (str): Expected digest of the complete downloaded archive.
        member (str): Archive member containing the executable or core library.
        binary_sha256 (str): Expected digest of the selected binary.
        destination (str): Binary installation path relative to the repository
            root; install_dependency() requires the resolved path to remain under external/.
    """

    version: str
    url: str
    archive_sha256: str
    member: str
    binary_sha256: str
    destination: str


def hash_file(path: str | os.PathLike[str]) -> str:
    """
    Return the hexadecimal SHA-256 digest of the file at path.

    Args:
        path (str | os.PathLike[str]): File to read in full as bytes for
            hashing.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest of the complete file contents.

    Raises:
        OSError: The file cannot be opened or read.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_dependency(name: str) -> pathlib.Path:
    """
    Reuse a verified installed binary or install it from a verified archive.

    A cached archive is reused only when its digest matches the lock file.
    Missing or mismatched archives are downloaded again before extraction.
    The binary and any Asar license files are written directly, not as one atomic
    file transaction. A later write failure may leave some installed files.

    Args:
        name (str): Dependency key from dependencies.lock.json, such as asar or
            snes9x.

    Returns:
        pathlib.Path: Verified executable or core DLL installed under
            external/.

    Raises:
        ValueError: The platform is not Windows, the destination escapes external/,
            or a downloaded archive or extracted binary fails its pinned checksum.
        KeyError: name or the required archive member is missing.
        OSError: Lock-file, cache, installed-file, or network access fails.
        zipfile.BadZipFile: The verified archive cannot be read as a ZIP file.
    """
    if os.name != "nt":
        raise ValueError(
            'The pinned binaries target Windows. See "Platforms" under Development in README.md.'
        )

    spec: DependencySpec = json.loads(
        (ROOT / "dependencies.lock.json").read_text(encoding="utf-8")
    )[name]
    target = (ROOT / spec["destination"]).resolve()
    if not target.is_relative_to((ROOT / "external").resolve()):
        raise ValueError("Dependency destination must remain under external/")

    # An existing filename is insufficient: reuse the binary only when its
    # contents match the pinned binary digest.
    if target.is_file() and hash_file(target) == spec["binary_sha256"]:
        return target

    cache = ROOT / "external/downloads" / spec["url"].rsplit("/", 1)[1]
    cache.parent.mkdir(parents=True, exist_ok=True)

    if not cache.is_file() or hash_file(cache) != spec["archive_sha256"]:
        print(f"Downloading {name} {spec['version']} from GitHub...", flush=True)
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "SolidRunner-setup"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()

        if hashlib.sha256(payload).hexdigest() != spec["archive_sha256"]:
            raise ValueError(f"{name}: archive checksum mismatch; nothing installed")

        # Cache only an archive that passed its pinned digest; the selected member
        # is checked separately before installation.
        cache.write_bytes(payload)

    # Extract only the expected binary and Asar's accompanying licenses/docs.
    # Do not execute arbitrary archive paths or extract unverified archives.
    with zipfile.ZipFile(cache) as archive:
        payload = archive.read(spec["member"])
        if hashlib.sha256(payload).hexdigest() != spec["binary_sha256"]:
            raise ValueError(f"{name}: binary checksum mismatch; nothing installed")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

        if name == "asar":
            for item in (
                "LICENSE",
                "license-gpl.txt",
                "license-lgpl.txt",
                "license-wtfpl.txt",
                "README.txt",
            ):
                (target.parent / item).write_bytes(archive.read(item))

    return target
