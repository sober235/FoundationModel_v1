"""v5.1 private output files (plan_v5 T09c; Proposal §2.1 "私有数组引用和哈希").

Every array of a PerceptionOutput or ModuleOutput (maps, acceptance and reason maps, instance masks) is written
once as its own .npy (allow_pickle=False, mode 600, never overwritten); the record JSON keeps every other field
as is and replaces each array by an ArrayRef {file, sha256, dtype, shape}. File names start with a hash prefix of
observation_uid, never the identifier itself. Loading checks each array against its reference, then validates
the whole output again.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

from .contracts_v5 import (
    ModuleOutput, PerceptionOutput, V5Error, _check_schema, array_sha256, validate_module_output_v5,
    validate_perception_output_v5)
from .paths import PrivatePathError, resolve_private_output, write_private_bytes, write_private_json

RECORD_SCHEMA = "perception_record"
_FILE = re.compile(r"[0-9a-f]{16}(\.[A-Za-z0-9_]+)+\.npy")


def output_prefix_v5(observation_uid: str) -> str:
    return hashlib.sha256(observation_uid.encode("utf-8")).hexdigest()[:16]


def _slots(body: dict) -> list[tuple]:
    """Where arrays live: the voxel maps, acceptance/reason task maps and instance masks."""
    slots = [(key,) for key in ("anatomy_map", "abnormality_map", "tissue_state_map") if body.get(key) is not None]
    for group in ("acceptance", "reason"):
        slots += [(group, key) for key in ("anatomy", "abnormality", "state") if key in body[group]]
    slots += [("abnormal_instances", i, "mask") for i in range(len(body.get("abnormal_instances", [])))]
    return slots


def _get(body, slot: tuple):
    for key in slot:
        body = body[key]
    return body


def _set(body, slot: tuple, value) -> None:
    for key in slot[:-1]:
        body = body[key]
    body[slot[-1]] = value


def _path(slot: tuple) -> str:
    return "$" + "".join(f"[{k}]" if isinstance(k, int) else f".{k}" for k in slot)


def _containers(value):
    """Copy dicts and lists, keep leaves (arrays included) as they are."""
    if isinstance(value, dict):
        return {key: _containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_containers(item) for item in value]
    return value


def save_output_v5(output, out_dir: Path, private_root: Path, repo: Path) -> Path:
    """Write the arrays, then the record; nothing is written if any target exists or lies outside the root."""
    if not isinstance(output, (PerceptionOutput, ModuleOutput)):
        raise V5Error("INVALID_TYPE", "$")
    body = {field.name: getattr(output, field.name) for field in dataclasses.fields(output)}
    prefix = output_prefix_v5(body["observation_uid"])
    record_path = resolve_private_output(Path(out_dir) / f"{prefix}.record.json", private_root, repo)
    record = _containers(body)
    files = {}
    for slot in _slots(body):
        array = _get(body, slot)
        if not isinstance(array, np.ndarray):
            raise V5Error("ARRAY_REQUIRED", _path(slot))
        name = f"{prefix}." + ".".join(str(k) for k in slot) + ".npy"
        files[slot] = (resolve_private_output(Path(out_dir) / name, private_root, repo), array)
        _set(record, slot, {"file": name, "sha256": array_sha256(array), "dtype": str(array.dtype),
                            "shape": [int(s) for s in array.shape]})
    _check_schema(record, RECORD_SCHEMA)
    if record_path.exists() or any(path.exists() for path, _ in files.values()):
        raise PrivatePathError("PRIVATE_FILE_EXISTS")
    for path, array in files.values():
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        write_private_bytes(path, buffer.getvalue())
    write_private_json(record_path, record)
    return record_path


def load_output_v5(record_path: Path, scope):
    """Read a record and its arrays; every array must match its reference (sha256, dtype, shape)."""
    record_path = Path(record_path)
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        raise V5Error("RECORD_UNREADABLE", "$") from None
    _check_schema(record, RECORD_SCHEMA)
    body = _containers(record)
    for slot in _slots(record):
        ref, path = _get(record, slot), _path(slot)
        if not _FILE.fullmatch(ref["file"]):
            raise V5Error("INVALID_REF", f"{path}.file")
        try:
            array = np.load(record_path.parent / ref["file"], allow_pickle=False)
        except FileNotFoundError:
            raise V5Error("ARRAY_MISSING", path) from None
        except (OSError, ValueError):
            raise V5Error("ARRAY_HASH_MISMATCH", path) from None
        if array_sha256(array) != ref["sha256"] or str(array.dtype) != ref["dtype"] \
                or [int(s) for s in array.shape] != ref["shape"]:
            raise V5Error("ARRAY_HASH_MISMATCH", path)
        _set(body, slot, array)
    if record["output_kind"] == "joint":
        return validate_perception_output_v5(body, scope)
    return validate_module_output_v5(body, scope)
