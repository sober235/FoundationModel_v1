import json
import os

import pytest

from anatobind.v41.paths import (
    PrivatePathError,
    redact,
    resolve_private_output,
    write_private_json,
    write_task_record,
)


@pytest.fixture
def layout(tmp_path):
    root = tmp_path / "private_root"
    root.mkdir(mode=0o700)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    return root, repo, outside


def test_valid_private_path_is_canonical(layout):
    root, repo, _ = layout
    out = resolve_private_output(root / "audits" / "./x", root, repo)
    assert out == (root / "audits" / "x").resolve()


def test_root_itself_rejected(layout):
    root, repo, _ = layout
    with pytest.raises(PrivatePathError, match="PRIVATE_PATH_IS_ROOT"):
        resolve_private_output(root, root, repo)


def test_outside_root_rejected(layout):
    root, repo, outside = layout
    with pytest.raises(PrivatePathError, match="PRIVATE_PATH_OUTSIDE_ROOT"):
        resolve_private_output(outside / "x", root, repo)


def test_parent_traversal_rejected(layout):
    root, repo, _ = layout
    with pytest.raises(PrivatePathError, match="PRIVATE_PATH_TRAVERSAL"):
        resolve_private_output(root / "audits" / ".." / "audits" / "x", root, repo)


def test_symlink_escape_rejected(layout):
    root, repo, outside = layout
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PrivatePathError, match="PRIVATE_PATH_OUTSIDE_ROOT"):
        resolve_private_output(root / "link" / "x", root, repo)


def test_git_target_rejected(layout):
    root, repo, _ = layout
    with pytest.raises(PrivatePathError, match="PRIVATE_PATH_IN_GIT"):
        resolve_private_output(repo / "reports" / "x", root, repo)
    nested = root / "nested_repo"
    (nested / ".git").mkdir(parents=True)
    (nested / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(PrivatePathError, match="PRIVATE_PATH_IN_GIT"):
        resolve_private_output(nested / "x", root, repo)


def test_error_text_excludes_private_path(layout):
    root, repo, outside = layout
    secret = outside / "zz_patient_root" / "CASE_SECRET_000777"
    with pytest.raises(PrivatePathError) as info:
        resolve_private_output(secret, root, repo)
    assert "zz_patient_root" not in str(info.value)
    assert "CASE_SECRET_000777" not in str(info.value)


def test_redact_replaces_secrets():
    text = "failed for /p/CASE_1 at /p/CASE_1/img.nii.gz"
    assert redact(text, ["/p/CASE_1"]) == "failed for <redacted> at <redacted>/img.nii.gz"


def test_write_private_json_refuses_overwrite(layout):
    root, _, _ = layout
    target = root / "a" / "b.json"
    write_private_json(target, {"k": 1})
    assert oct(target.stat().st_mode & 0o777) == "0o600"
    assert oct((root / "a").stat().st_mode & 0o777) == "0o700"
    with pytest.raises(PrivatePathError, match="PRIVATE_FILE_EXISTS"):
        write_private_json(target, {"k": 2})
    assert json.loads(target.read_text()) == {"k": 1}


def test_task_record_requires_status_enum(layout):
    root, _, _ = layout
    with pytest.raises(PrivatePathError, match="TASK_RECORD_INVALID"):
        write_task_record(root / "t" / "task_record.json", {"task_id": "T01", "status": "DONE"})
    with pytest.raises(PrivatePathError, match="TASK_RECORD_INVALID"):
        write_task_record(root / "t" / "task_record.json", {"status": "PASS"})


def test_interrupted_task_record_leaves_no_certificate(layout, monkeypatch):
    root, _, _ = layout
    target = root / "t" / "task_record.json"

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("anatobind.v41.paths.json.dump", boom)
    with pytest.raises(OSError):
        write_task_record(target, {"task_id": "T01", "status": "IN_PROGRESS"})
    assert not target.exists()
    assert list(target.parent.iterdir()) == []
    monkeypatch.undo()
    write_task_record(target, {"task_id": "T01", "status": "IN_PROGRESS"})
    body = json.loads(target.read_text())
    assert body["status"] == "IN_PROGRESS"
    assert body["task_id"] == "T01"
    assert "written_at" in body
