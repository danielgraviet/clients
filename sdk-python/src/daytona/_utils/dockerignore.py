# Copyright Daytona Platforms Inc.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for filtering Docker build contexts with ``.dockerignore`` rules."""

from __future__ import annotations

import os
import tarfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from pathspec import GitIgnoreSpec


def load_spec(dockerignore_path: str | None) -> GitIgnoreSpec | None:
    """Load a ``.dockerignore`` file into a matchable spec, or ``None`` if there isn't one."""
    if dockerignore_path is None or not os.path.isfile(dockerignore_path):
        return None

    with open(dockerignore_path, encoding="utf-8") as dockerignore_file:
        lines = dockerignore_file.read().splitlines()
    # Docker treats a single dot as a no-op for historical reasons.
    return GitIgnoreSpec.from_lines("gitignore", (line for line in lines if line.strip() != "."))


def is_ignored(path: Path, context_root: Path, spec: GitIgnoreSpec | None, *, is_dir: bool = False) -> bool:
    """Check whether ``path`` is excluded by ``spec``, a spec loaded by `load_spec`.

    ``path`` and ``context_root`` are normalized to absolute paths before comparison, so callers
    don't need to agree in advance on whether they're passing absolute or relative paths.
    """
    if spec is None:
        return False
    try:
        relative = Path(os.path.abspath(path)).relative_to(os.path.abspath(context_root)).as_posix()
    except ValueError:
        # Docker ignore rules only apply to paths under the build context.
        return False
    if relative == ".":
        return False
    return spec.match_file(relative + ("/" if is_dir else ""))


def iter_context_entries(
    source_path: str,
    archive_base_path: str,
    dockerignore_path: str | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield source and archive paths for entries included in a build context."""
    spec = load_spec(dockerignore_path)
    source = Path(source_path)
    context_root = Path(dockerignore_path).parent if dockerignore_path else source

    if not source.is_dir():
        if not is_ignored(source, context_root, spec):
            yield str(source), archive_base_path
        return

    included_dirs: set[Path] = {source}
    included_files: list[Path] = []
    for root, dirs, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        # Don't prune ignored directories: a later negation rule may include a file below one.
        dirs[:] = sorted(dirs)
        for dirname in dirs:
            directory = root_path / dirname
            if not is_ignored(directory, context_root, spec, is_dir=True):
                included_dirs.add(directory)
        for filename in sorted(files):
            file_path = root_path / filename
            if not is_ignored(file_path, context_root, spec):
                included_files.append(file_path)
                parent = file_path.parent
                while parent != source and source in parent.parents:
                    included_dirs.add(parent)
                    parent = parent.parent

    for directory in sorted(included_dirs):
        relative = directory.relative_to(source)
        archive_path = (
            archive_base_path
            if str(relative) == "."
            else str(PurePosixPath(archive_base_path) / PurePosixPath(relative.as_posix()))
        )
        yield str(directory), archive_path
    for file_path in included_files:
        relative = file_path.relative_to(source)
        yield str(file_path), str(PurePosixPath(archive_base_path) / PurePosixPath(relative.as_posix()))


def add_context_to_tar(
    tar: tarfile.TarFile,
    source_path: str,
    archive_base_path: str,
    dockerignore_path: str | None = None,
) -> None:
    """Add filtered context entries without recursively re-walking ignored paths."""
    for local_path, archive_path in iter_context_entries(source_path, archive_base_path, dockerignore_path):
        tar.add(local_path, arcname=archive_path, recursive=False)
