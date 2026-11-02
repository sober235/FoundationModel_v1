import gzip
import json
import os
import stat

import nibabel as nib
import numpy as np
import pytest
import yaml

from anatobind.v41.inventory import (
    discover_assets,
    load_sources_config,
    summarize_assets,
    verify_asset,
)

CONFIG = {
    "schema_version": "v41.0",
    "credential_like_patterns": ["*token*", "*credential*", "*.key", ".netrc", "*secret*"],
    "archive_suffixes": [".tar.gz", ".tgz", ".zip", ".tar"],
    "sources": {
        "SRC_A": {
            "priority": "P0",
            "role": "U",
            "status": "pending_verification",
            "case_patterns": ["data/sub-*", "data/derivatives/sub-*"],
            "image_patterns": ["data/sub-*/**/*.nii.gz"],
            "label_patterns": ["data/derivatives/sub-*/**/*_msk.nii.gz"],
        },
        "SRC_B": {
            "priority": "P0",
            "role": "A",
            "status": "pending_verification",
            "case_patterns": ["image_files/*.h5", "masks/*.nii.gz"],
            "image_patterns": ["image_files/*.h5"],
            "label_patterns": ["masks/*.nii.gz"],
        },
        "SRC_EMPTY": {"priority": "P2", "role": "unknown", "status": "pending_verification"},
    },
}


def _nifti(path, data=None):
    data = np.zeros((4, 4, 4), dtype=np.int16) if data is None else data
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


@pytest.fixture
def root(tmp_path):
    base = tmp_path / "raw"
    a = base / "SRC_A" / "data"
    for i in (1, 2):
        (a / f"sub-{i:04d}" / "ses-0001" / "dwi").mkdir(parents=True)
        _nifti(a / f"sub-{i:04d}" / "ses-0001" / "dwi" / f"sub-{i:04d}_dwi.nii.gz")
        (a / "derivatives" / f"sub-{i:04d}" / "ses-0001").mkdir(parents=True)
        _nifti(a / "derivatives" / f"sub-{i:04d}" / "ses-0001" / f"sub-{i:04d}_msk.nii.gz")
    (a / "sub-0003").mkdir()
    (base / "SRC_A" / "data.zip").write_bytes(b"PK\x05\x06" + b"\0" * 18)
    (base / "SRC_A" / "download.token").write_text("do-not-read")
    b = base / "SRC_B"
    (b / "image_files").mkdir(parents=True)
    (b / "masks").mkdir()
    (b / "image_files" / "MTR_001.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 64)
    (b / "image_files" / "MTR_002.h5.truncated").write_bytes(b"\0" * 8)
    (b / "image_files.tar.gz").write_bytes(b"\x1f\x8b" + b"\0" * 8)
    _nifti(b / "masks" / "MTR_001.nii.gz")
    (base / "SRC_EMPTY").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.nii.gz").write_bytes(b"x")
    (b / "escape").symlink_to(outside, target_is_directory=True)
    return base


@pytest.fixture
def asset_fixture(root):
    assets = discover_assets(root, CONFIG)
    by_name = {a["relpath"].rsplit("/", 1)[-1]: a for a in assets}

    def pick(name):
        return by_name[name]

    pick.assets = assets
    return pick


def test_empty_directory_is_not_a_case(root):
    summary = summarize_assets(discover_assets(root, CONFIG))
    assert summary["sources"]["SRC_EMPTY"]["cases_observed"] == 0
    assert summary["sources"]["SRC_A"]["cases_observed"] == 2


def test_archive_and_extracted_copy_are_not_double_counted(root):
    assets = discover_assets(root, CONFIG)
    archives = [a for a in assets if a["kind"] == "archive"]
    assert {a["relpath"] for a in archives} == {"SRC_A/data.zip", "SRC_B/image_files.tar.gz"}
    assert all(a["case_key"] is None for a in archives)
    summary = summarize_assets(assets)
    assert summary["sources"]["SRC_B"]["cases_observed"] == 1
    assert summary["sources"]["SRC_B"]["archives"] == 1


def test_truncated_asset_is_quarantined(asset_fixture):
    report = verify_asset(asset_fixture("MTR_002.h5.truncated"), "header")
    assert report["training_eligible"] is False
    assert report["reason"] == "TRUNCATED_ASSET"
    assert report["status"] == "quarantined"
    assert asset_fixture("MTR_002.h5.truncated")["kind"] == "quarantined"


def test_symlink_escape_is_not_followed(root):
    assets = discover_assets(root, CONFIG)
    assert not any("leak" in a["relpath"] for a in assets)
    summary = summarize_assets(assets)
    assert summary["sources"]["SRC_B"]["excluded_symlinks"] == 1


def test_credential_like_files_are_counted_not_read(root):
    token = root / "SRC_A" / "download.token"
    token.chmod(0)
    try:
        assets = discover_assets(root, CONFIG)
    finally:
        token.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert not any(a["relpath"].endswith("download.token") for a in assets)
    assert summarize_assets(assets)["sources"]["SRC_A"]["excluded_credential_like"] == 1


def test_asset_records_are_classified_with_parent_source(asset_fixture):
    image = asset_fixture("sub-0001_dwi.nii.gz")
    label = asset_fixture("sub-0001_msk.nii.gz")
    assert image["kind"] == "image" and label["kind"] == "label"
    assert image["source"] == "SRC_A" and image["case_key"] == "SRC_A/data/sub-0001"
    assert label["case_key"] == "SRC_A/data/derivatives/sub-0001"
    assert image["case_id"] == label["case_id"] == "sub-0001"
    assert image["status"] == "observed" and len(image["asset_id"]) == 64


def test_header_level_is_not_full_verification(asset_fixture):
    report = verify_asset(asset_fixture("sub-0001_dwi.nii.gz"), "header")
    assert report["status"] == "header_only"
    assert report["training_eligible"] is None
    assert report["header"]["shape"] == [4, 4, 4]
    assert report["content_sha256"] is None


def test_full_decode_records_content_hash(asset_fixture):
    report = verify_asset(asset_fixture("sub-0001_dwi.nii.gz"), "full_decode")
    assert report["status"] == "verified"
    assert report["training_eligible"] is True
    assert len(report["content_sha256"]) == 64


def test_corrupt_nifti_fails_decode(asset_fixture, root):
    target = root / "SRC_A" / "data" / "sub-0002" / "ses-0001" / "dwi" / "sub-0002_dwi.nii.gz"
    target.write_bytes(gzip.compress(b"not a nifti"))
    report = verify_asset(asset_fixture("sub-0002_dwi.nii.gz"), "full_decode")
    assert report["status"] == "failed"
    assert report["training_eligible"] is False
    assert report["reason"] == "DECODE_ERROR"


def test_metadata_level_does_not_open_file(asset_fixture):
    report = verify_asset(asset_fixture("MTR_001.h5"), "metadata")
    assert report["status"] == "observed"
    assert report["size_bytes"] == 72
    with pytest.raises(ValueError, match="INVALID_VERIFY_LEVEL"):
        verify_asset(asset_fixture("MTR_001.h5"), "deep")


def test_report_has_no_private_path(asset_fixture, root):
    report = verify_asset(asset_fixture("MTR_001.h5"), "header")
    assert str(root) not in json.dumps(report)
    assert report["reason"] in ("H5_OPEN_ERROR", None)


def test_shipped_sources_config_lists_17_entries_pending_verification():
    config = load_sources_config()
    assert len(config["sources"]) == 17
    assert all(v["status"] == "pending_verification" for v in config["sources"].values())
    p0 = {k for k, v in config["sources"].items() if v["priority"] == "P0"}
    assert p0 == {"ISLES_ltr", "SKM-TEA_ltr"}
    assert yaml.safe_dump(config)


def test_verify_inventory_is_incremental_and_defaults_to_p0(root, tmp_path):
    from anatobind.v41.inventory import read_jsonl, verify_inventory, write_inventory

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    out = private / "audits" / "inv"
    write_inventory(root, CONFIG, out, private, repo)
    first = verify_inventory(root, out / "assets.jsonl", "header", out, private, repo)
    assert first["priorities"] == ["P0"]
    assert first["counts"]["header_only"] == 5 and first["counts"]["quarantined"] == 1
    second = verify_inventory(root, out / "assets.jsonl", "header", out, private, repo)
    assert second["counts"] == first["counts"]
    rows = read_jsonl(out / "asset_integrity.jsonl")
    assert all(r["status"] != "verified" for r in rows)
    assert all(r["training_eligible"] is not True for r in rows)
