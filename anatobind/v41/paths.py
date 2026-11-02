"""Private output path checks, redaction and atomic private writes.

`resolve_private_output` and `write_private_json` descend from
anatobind/aur/data_contract.py (sober235/Foundation_Model @4deeead,
blob c29f2570) with the private root passed explicitly instead of a module
constant. Error messages carry codes only, never the offending path.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from anatobind.v41.schema import TASK_STATUSES

REDACTED = "<redacted>"


class PrivatePathError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def redact(text: str, secrets: Iterable[str]) -> str:
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return text


def _is_git_dir(directory: Path) -> bool:
    git = directory / ".git"
    if git.is_file() or (git / "HEAD").is_file():
        return True
    return (directory / "HEAD").is_file() and (directory / "objects").is_dir() and (directory / "config").is_file()


def _inside_git(path: Path) -> bool:
    return any(_is_git_dir(parent) for parent in (path, *path.parents))


def resolve_private_output(path: Path, root: Path, repo: Path) -> Path:
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise PrivatePathError("PRIVATE_PATH_TRAVERSAL")
    resolved = raw.resolve()
    root_resolved = Path(root).expanduser().resolve()
    if resolved == root_resolved:
        raise PrivatePathError("PRIVATE_PATH_IS_ROOT")
    repo_resolved = Path(repo).expanduser().resolve()
    if resolved == repo_resolved or resolved.is_relative_to(repo_resolved):
        raise PrivatePathError("PRIVATE_PATH_IN_GIT")
    if not resolved.is_relative_to(root_resolved):
        raise PrivatePathError("PRIVATE_PATH_OUTSIDE_ROOT")
    if _inside_git(resolved):
        raise PrivatePathError("PRIVATE_PATH_IN_GIT")
    return resolved


def _make_private_dirs(directory: Path) -> None:
    missing = []
    parent = directory
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for item in reversed(missing):
        item.mkdir(mode=0o700)
    if directory.stat().st_mode & 0o077:
        directory.chmod(0o700)


def write_private_json(path: Path, value: Any) -> None:
    path = Path(path)
    _make_private_dirs(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with os.fdopen(os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise PrivatePathError("PRIVATE_FILE_EXISTS") from None
    finally:
        if tmp.exists():
            tmp.unlink()


def write_task_record(path: Path, record: dict) -> None:
    if not isinstance(record, dict) or not isinstance(record.get("task_id"), str):
        raise PrivatePathError("TASK_RECORD_INVALID")
    if record.get("status") not in TASK_STATUSES:
        raise PrivatePathError("TASK_RECORD_INVALID")
    body = dict(record)
    body.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_private_json(Path(path), body)
