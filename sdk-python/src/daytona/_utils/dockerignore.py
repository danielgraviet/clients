# Copyright Daytona Platforms Inc.
# SPDX-License-Identifier: Apache-2.0

"""Filtering of Docker build contexts with ``.dockerignore`` rules.

Matching follows Docker's own rules (moby/patternmatcher and the BuildKit context walker), not
``.gitignore`` rules: every pattern is anchored at the context root, ``*`` and ``?`` never cross a
``/``, ``**`` spans any number of directories, a later ``!`` rule re-includes what an earlier rule
excluded, and a rule that matches a directory excludes everything below it.
"""

from __future__ import annotations

import os
import posixpath
import re
import tarfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum

# Regexp metacharacters that have no meaning in a Docker pattern, so they match literally.
_LITERAL_REGEX_CHARS = frozenset(".+()|{}$")
# Characters that make a pattern more than a plain path prefix.
_WILDCARD_CHARS = frozenset("*[]?^\\")


class _MatchType(Enum):
    EXACT = "exact"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    REGEX = "regex"


@dataclass(frozen=True)
class _Pattern:
    cleaned: str
    exclusion: bool
    match_type: _MatchType
    regex: re.Pattern[str] | None

    def match(self, path: str) -> bool:
        if self.match_type is _MatchType.EXACT:
            return path == self.cleaned
        if self.match_type is _MatchType.PREFIX:
            return path.startswith(self.cleaned[:-2])
        if self.match_type is _MatchType.SUFFIX:
            suffix = self.cleaned[2:]
            return path.endswith(suffix) or (suffix.startswith("/") and path == suffix[1:])
        if self.regex is None:
            return False
        found = self.regex.match(path)
        return found is not None and found.end() == len(path)

    def without_trailing_glob(self) -> str:
        return self.cleaned.removesuffix("/**").removesuffix("/*")


def _clean(pattern: str) -> str:
    """Lexically clean a pattern the way Go's ``filepath.Clean`` does."""
    cleaned = posixpath.normpath(pattern)
    # normpath keeps exactly two leading slashes, Clean folds them into one.
    return "/" + cleaned.lstrip("/") if cleaned.startswith("//") else cleaned


def _compile(cleaned: str, exclusion: bool) -> _Pattern | None:
    """Translate one pattern into a matcher, or ``None`` if it is not a valid pattern."""
    regex = "^"
    match_type = _MatchType.EXACT
    position = 0
    index = 0
    while position < len(cleaned):
        char = cleaned[position]
        position += 1
        if char == "*":
            if position < len(cleaned) and cleaned[position] == "*":
                position += 1
                # "**/" behaves like "**".
                if position < len(cleaned) and cleaned[position] == "/":
                    position += 1
                if position == len(cleaned):
                    if match_type is _MatchType.EXACT:
                        match_type = _MatchType.PREFIX
                    else:
                        regex += ".*"
                        match_type = _MatchType.REGEX
                else:
                    regex += "(.*/)?"
                    match_type = _MatchType.REGEX
                if index == 0:
                    match_type = _MatchType.SUFFIX
            else:
                regex += "[^/]*"
                match_type = _MatchType.REGEX
        elif char == "?":
            regex += "[^/]"
            match_type = _MatchType.REGEX
        elif char in _LITERAL_REGEX_CHARS:
            regex += "\\" + char
        elif char == "\\":
            if position < len(cleaned):
                regex += "\\" + cleaned[position]
                position += 1
                match_type = _MatchType.REGEX
            else:
                regex += "\\"
        elif char in "[]":
            regex += char
            match_type = _MatchType.REGEX
        else:
            regex += char
        index += 1

    if match_type is not _MatchType.REGEX:
        return _Pattern(cleaned, exclusion, match_type, None)
    try:
        return _Pattern(cleaned, exclusion, match_type, re.compile(regex + "$"))
    except re.error:
        return None


def _parse(content: str) -> list[_Pattern]:
    """Parse ``.dockerignore`` content into patterns, in file order."""
    patterns: list[_Pattern] = []
    for line_number, raw_line in enumerate(content.split("\n")):
        line = raw_line.removesuffix("\r")
        if line_number == 0:
            line = line.removeprefix("\ufeff")
        if line.startswith("#"):
            continue
        line = line.strip()
        if not line:
            continue
        if os.sep == "\\":
            # On Windows a backslash is a path separator, never an escape.
            line = line.replace("\\", "/")
        exclusion = line.startswith("!")
        if exclusion:
            line = line[1:].strip()
        if line:
            line = _clean(line)
            # A leading slash is redundant: every pattern is relative to the context root.
            if len(line) > 1 and line.startswith("/"):
                line = line[1:]
        if exclusion and not line:
            # Docker rejects a bare "!"; skip it rather than fail the whole build.
            continue
        compiled = _compile(line, exclusion)
        if compiled is not None:
            patterns.append(compiled)
    return patterns


@dataclass(frozen=True)
class DockerIgnore:
    """The ``.dockerignore`` rules of one build context."""

    context_root: str
    patterns: tuple[_Pattern, ...]
    _has_exclusions: bool = field(init=False)
    _only_prefix_exclusions: bool = field(init=False)

    def __post_init__(self) -> None:
        exclusions = [pattern for pattern in self.patterns if pattern.exclusion]
        object.__setattr__(self, "_has_exclusions", bool(exclusions))
        object.__setattr__(
            self,
            "_only_prefix_exclusions",
            not any(_WILDCARD_CHARS.intersection(pattern.without_trailing_glob()) for pattern in exclusions),
        )

    @classmethod
    def load(cls, dockerignore_path: str | None) -> DockerIgnore | None:
        """Read the rules at ``dockerignore_path``.

        Returns ``None`` when there is no such file or it has no rules, so that callers can keep
        the unfiltered behaviour for contexts without a ``.dockerignore``.
        """
        if dockerignore_path is None or not os.path.isfile(dockerignore_path):
            return None
        with open(dockerignore_path, encoding="utf-8", errors="surrogateescape") as dockerignore_file:
            patterns = _parse(dockerignore_file.read())
        if not patterns:
            return None
        return cls(os.path.dirname(os.path.abspath(dockerignore_path)), tuple(patterns))

    def relative(self, path: str) -> str | None:
        """Return ``path`` relative to the context root, or ``None`` if it lies outside it."""
        try:
            relative = os.path.relpath(os.path.abspath(path), self.context_root)
        except ValueError:  # a different drive on Windows
            return None
        relative = relative.replace(os.sep, "/")
        if relative == ".." or relative.startswith("../"):
            return None
        return relative

    def excludes(self, relative: str) -> bool:
        """Whether a context-relative path is excluded, directly or through a parent directory."""
        if relative == ".":
            # Docker never excludes the context root itself.
            return False
        parent = posixpath.dirname(relative)
        parent_dirs = parent.split("/") if parent else []
        matched = False
        for pattern in self.patterns:
            # An exclusion can only change an already matched path, an inclusion an unmatched one.
            if pattern.exclusion != matched:
                continue
            match = pattern.match(relative) or any(
                pattern.match("/".join(parent_dirs[: depth + 1])) for depth in range(len(parent_dirs))
            )
            if match:
                matched = not pattern.exclusion
        return matched

    def may_include_below(self, relative_dir: str) -> bool:
        """Whether a later ``!`` rule could re-include something inside an excluded directory."""
        if not self._has_exclusions:
            return False
        if not self._only_prefix_exclusions:
            return True
        dir_slash = relative_dir + "/"
        return any(
            (pattern.without_trailing_glob() + "/").startswith(dir_slash)
            for pattern in self.patterns
            if pattern.exclusion
        )


@dataclass
class _PendingDir:
    """An excluded directory that is archived only if something below it is re-included."""

    local_path: str
    archive_path: str
    emitted: bool = False


def _is_real_dir(path: str) -> bool:
    return os.path.isdir(path) and not os.path.islink(path)


def _walk(
    local_path: str,
    archive_path: str,
    relative: str | None,
    dockerignore: DockerIgnore,
    pending: list[_PendingDir],
) -> Iterator[tuple[str, str]]:
    is_dir = _is_real_dir(local_path)
    if relative is not None and dockerignore.excludes(relative):
        if not is_dir or not dockerignore.may_include_below(relative):
            return
        pending = [*pending, _PendingDir(local_path, archive_path)]
    else:
        for directory in pending:
            if not directory.emitted:
                directory.emitted = True
                yield directory.local_path, directory.archive_path
        pending = []
        yield local_path, archive_path

    if not is_dir:
        return
    # Same order as tarfile's own recursive add, so an archive that excludes nothing is unchanged.
    for name in sorted(os.listdir(local_path)):
        child_relative = None if relative is None else (name if relative == "." else f"{relative}/{name}")
        yield from _walk(
            os.path.join(local_path, name),
            posixpath.join(archive_path, name),
            child_relative,
            dockerignore,
            pending,
        )


def iter_context_entries(
    source_path: str, archive_base_path: str, dockerignore: DockerIgnore
) -> Iterator[tuple[str, str]]:
    """Yield ``(local path, archive path)`` for every entry of a source that the rules keep.

    Entries come in the order ``tarfile`` adds them. Symlinks are kept as links and never
    followed, and a source outside the context root is not filtered at all.
    """
    yield from _walk(source_path, archive_base_path, dockerignore.relative(source_path), dockerignore, [])


def add_context_to_tar(
    tar: tarfile.TarFile, source_path: str, archive_base_path: str, dockerignore: DockerIgnore
) -> None:
    """Add the entries of a source that the rules keep to ``tar``."""
    for local_path, archive_path in iter_context_entries(source_path, archive_base_path, dockerignore):
        tar.add(local_path, arcname=archive_path, recursive=False)


def hash_context_entries(
    update: Callable[[bytes], None], source_path: str, archive_base_path: str, dockerignore: DockerIgnore
) -> None:
    """Feed exactly the entries `add_context_to_tar` archives into a hash's ``update``."""
    for local_path, archive_path in iter_context_entries(source_path, archive_base_path, dockerignore):
        name = os.fsencode(archive_path)
        if os.path.islink(local_path):
            update(b"l\0" + name + b"\0" + os.fsencode(os.readlink(local_path)) + b"\0")
        elif os.path.isdir(local_path):
            update(b"d\0" + name + b"\0")
        else:
            update(b"f\0" + name + b"\0" + str(os.path.getsize(local_path)).encode() + b"\0")
            with open(local_path, "rb") as file:
                while chunk := file.read(8192):
                    update(chunk)
