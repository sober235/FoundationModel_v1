"""v5.1 private output files: one .npy per array, a record with hashed references (plan_v5 T09c)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import stat
from pathlib import Path

import numpy as np
import pytest

from anatobind.v41.contracts_v5 import (
    PerceptionOutput, V5Error, array_sha256, validate_module_output_v5, validate_perception_output_v5,
    validate_scope_v5)
from anatobind.v41.paths import PrivatePathError
from anatobind.v41.perception_io import load_output_v5, save_output_v5
from tests.v5.factories import make_joint_output, make_module_output, make_module_scope, make_scope

REPO = Path(__file__).resolve().parents[2]
PREFIX = hashlib.sha256(b"obs-synth-1").hexdigest()[:16]


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _same(a, b):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    else:
        assert a == b


def _same_output(a, b):
    assert type(a) is type(b)
    for field in dataclasses.fields(a):
        _same(getattr(a, field.name), getattr(b, field.name))


def _joint():
    scope = validate_scope_v5(make_scope())
    return scope, validate_perception_output_v5(make_joint_output(scope), scope)


def test_saved_output_roundtrips_with_hashes(tmp_path):
    scope, output = _joint()
    root = tmp_path / "private"
    path = save_output_v5(output, root / "outputs", root, REPO)
    assert path == (root / "outputs" / f"{PREFIX}.record.json").resolve()
    names = sorted(p.name for p in path.parent.iterdir())
    assert all(name.startswith(PREFIX + ".") for name in names) and not [n for n in names if "obs-synth" in n]
    record = json.loads(path.read_text())
    assert record["anatomy_map"] == {"file": f"{PREFIX}.anatomy_map.npy", "sha256": array_sha256(output.anatomy_map),
                                     "dtype": str(output.anatomy_map.dtype), "shape": list(output.anatomy_map.shape)}
    assert record["abnormal_instances"][0]["mask"]["file"] == f"{PREFIX}.abnormal_instances.0.mask.npy"
    assert record["acceptance"]["state"]["file"] == f"{PREFIX}.acceptance.state.npy"
    assert record["acceptance"]["relations"] == {"inst-1": True} and record["observation_uid"] == "obs-synth-1"
    loaded = load_output_v5(path, scope)
    assert isinstance(loaded, PerceptionOutput)
    _same_output(loaded, output)
    module_scope = validate_scope_v5(make_module_scope("A"))
    module = validate_module_output_v5(make_module_output(module_scope, "A"), module_scope)
    path = save_output_v5(module, root / "modules", root, REPO)
    assert json.loads(path.read_text())["abnormality_map"] is None
    _same_output(load_output_v5(path, module_scope), module)


def test_tampered_array_rejected(tmp_path):
    scope, output = _joint()
    root = tmp_path / "private"
    path = save_output_v5(output, root / "outputs", root, REPO)
    target = path.parent / f"{PREFIX}.anatomy_map.npy"
    original = np.load(target)
    changed = original.copy()
    changed.flat[0] = 2 if changed.flat[0] != 2 else 1
    np.save(target, changed)
    _raises("ARRAY_HASH_MISMATCH", "$.anatomy_map", load_output_v5, path, scope)
    np.save(target, original.astype(np.int32 if original.dtype != np.int32 else np.int64))   # same values, new dtype
    _raises("ARRAY_HASH_MISMATCH", "$.anatomy_map", load_output_v5, path, scope)
    np.save(target, original)
    load_output_v5(path, scope)
    mask = path.parent / f"{PREFIX}.abnormal_instances.0.mask.npy"
    mask.unlink()
    _raises("ARRAY_MISSING", "$.abnormal_instances[0].mask", load_output_v5, path, scope)
    record = json.loads(path.read_text())
    record["anatomy_map"]["file"] = "../escape.npy"
    escaped = path.with_name(f"{PREFIX}.escaped.json")
    escaped.write_text(json.dumps(record))
    _raises("INVALID_REF", "$.anatomy_map.file", load_output_v5, escaped, scope)


def test_output_never_written_inside_repo(tmp_path):
    _, output = _joint()
    with pytest.raises(PrivatePathError) as err:
        save_output_v5(output, REPO / "outputs-never", tmp_path / "private", REPO)
    assert err.value.code == "PRIVATE_PATH_IN_GIT" and not (REPO / "outputs-never").exists()
    with pytest.raises(PrivatePathError) as err:
        save_output_v5(output, tmp_path / "elsewhere", tmp_path / "private", REPO)
    assert err.value.code == "PRIVATE_PATH_OUTSIDE_ROOT" and not (tmp_path / "elsewhere").exists()


def test_output_files_private_and_not_overwritten(tmp_path):
    _, output = _joint()
    root = tmp_path / "private"
    path = save_output_v5(output, root / "outputs", root, REPO)
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in path.parent.iterdir())
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    with pytest.raises(PrivatePathError) as err:
        save_output_v5(output, root / "outputs", root, REPO)
    assert err.value.code == "PRIVATE_FILE_EXISTS"
    assert {p.name: p.read_bytes() for p in path.parent.iterdir()} == before
