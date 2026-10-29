"""
This module replaces a group of files with supplied bytes and restores earlier contents if a replacement fails.

The caller supplies destination paths and replacement bytes. Before replacing
any destination, replace_many() writes all new contents and copies existing
files into a temporary directory. This lets the translation writer update its
text file and JSON metadata together and undo completed replacements if a
later os.replace() call raises OSError.

If rollback also fails, the remaining backups and recovery.json are retained
and their directory is reported in the error. Each replacement uses os.replace();
a process or operating-system crash can still leave only some files updated.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable


def replace_many(updates: Iterable[tuple[str | os.PathLike[str], bytes]]) -> None:
    """
    Replace an iterable of (path, bytes) pairs with rollback on write failure.

    Stages new contents and predecessor copies before replacement. A failed
    rollback retains recovery files and names their directory in the error.
    Individual replacements are atomic; the group is not atomic across a
    process or operating-system crash. Use this for related source files on the
    same filesystem: staging uses their common parent, and os.replace() cannot
    move files across filesystems. Backups preserve file bytes or symlink targets,
    not permissions, timestamps, or other filesystem metadata.

    Args:
        updates (Iterable[tuple[str | os.PathLike[str], bytes]]): Destination path and replacement
            bytes pairs, consumed once before staging. An empty iterable does nothing.
            Parents must exist; destination directory entries must be
            distinct, including aliases through parent symlinks or junctions.
            A final-component symlink is replaced, not followed, and rollback
            restores the link itself even if its target is missing. Separate
            hard-link directory entries are independent destinations.

    Raises:
        OSError: Staging, replacement or cleanup fails. Replacement failures
            trigger rollback; a failed rollback retains its recovery directory.
            If cleanup also fails, its error and directory are attached to the
            original exception. A cleanup failure after successful replacement
            is reported without undoing the completed updates.
        ValueError: Destinations repeat, or their parent paths do not have a
            common path, for example across different drives.
    """
    resolved = [(Path(path).absolute(), content) for path, content in updates]
    if not resolved:
        return

    # Resolve parent aliases, not the final component: os.replace replaces a
    # destination symlink itself, while directory aliases reach the same entry.
    destinations = [
        os.path.normcase(str(path.parent.resolve() / path.name)) for path, _ in resolved
    ]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Related file updates must have distinct destinations")

    directory = os.path.commonpath([str(path.parent) for path, _ in resolved])

    # Stage every new file and rollback copy before touching any destination.
    scratch = Path(tempfile.mkdtemp(prefix=".source-save-", dir=directory))
    preserve = False
    failure_in_progress: BaseException | None = None
    try:
        staged: list[Path] = []
        backups: list[Path | None] = []
        for index, (target, content) in enumerate(resolved):
            candidate = Path(scratch) / f"new-{index}"
            candidate.write_bytes(content)
            staged.append(candidate)
            backup = (
                Path(scratch) / f"old-{index}" if target.exists() or target.is_symlink() else None
            )
            if backup is not None:
                # Back up the directory entry rather than following a symlink.
                # Its relative target must be restored unchanged during rollback.
                shutil.copyfile(target, backup, follow_symlinks=False)

            backups.append(backup)

        # Write recovery instructions before any destination changes, so a later
        # disk/locking failure cannot prevent retaining an intelligible backup.
        (scratch / "recovery.json").write_text(
            json.dumps(
                [
                    dict(target=str(target), backup=backup.name if backup is not None else None)
                    for (target, _), backup in zip(resolved, backups)
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        replaced: list[int] = []
        try:
            for index, (target, _) in enumerate(resolved):
                os.replace(staged[index], target)
                replaced.append(index)

        except OSError as failure:
            # Undo only destinations already replaced, in reverse order. Missing
            # predecessors are restored by removing the newly created destination.
            rollback_errors: list[str] = []
            for index in reversed(replaced):
                target = resolved[index][0]
                try:
                    backup = backups[index]
                    if backup is None:
                        target.unlink()
                    else:
                        os.replace(backup, target)
                except OSError as error:
                    rollback_errors.append(f"{target}: {error}")

            if rollback_errors:
                preserve = True
                raise OSError(
                    f"Source update failed: {failure}. Rollback failed for "
                    + "; ".join(rollback_errors)
                    + f". Recovery files retained in {scratch}; see recovery.json. "
                    "A null backup means the target did not exist before this save; "
                    "a backup already moved back is no longer in the recovery folder."
                ) from failure

            raise
    except BaseException as error:
        # Keep the operation's failure primary even if removing scratch files
        # also fails. This includes interruption while staging or replacing.
        failure_in_progress = error
        raise
    finally:
        if not preserve:
            try:
                shutil.rmtree(scratch)
            except OSError as cleanup_error:
                if failure_in_progress is None:
                    raise
                failure_in_progress.add_note(
                    f"Temporary-file cleanup failed for {scratch}: {cleanup_error}"
                )
