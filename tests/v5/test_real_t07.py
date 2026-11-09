"""Real T07 preparation (plan_v5 §2B T07c real run; §2C T09b conventions). Synthetic records only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest
import yaml

from anatobind.v41.audits import build_registration_records
from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.geometry_qc import _csv
from anatobind.v41.migrate_v5 import migrate_v41_record_v5
from anatobind.v41.real_t07_v5 import (
    prepare_record_v5,
    read_qc_csv_v5,
    run_prepare_v5,
    skm_h5_geometry_v5,
)
from anatobind.v41.schema import SampleRecord
from anatobind.v41.source_policy_v5 import validate_source_policy_v5

CONFIG = Path("configs/v5")
ISLES_SCOPE = validate_scope_v5(yaml.safe_load((CONFIG / "scope_isles_u.yaml").read_text()))
SKM_SCOPE = validate_scope_v5(yaml.safe_load((CONFIG / "scope_skm_a.yaml").read_text()))
GEOMETRY = {"shape": [8, 8, 4], "spacing_mm": [2.0, 2.0, 2.0],
            "affine": [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]], "axis_codes": ["R", "A", "S"],
            "units": "mm"}
ISLES_ASSETS = [
    {"source": "ISLES_ltr", "kind": "image", "case_id": "c1", "relpath": "isles/c1/c1_dwi.nii.gz"},
    {"source": "ISLES_ltr", "kind": "image", "case_id": "c1", "relpath": "isles/c1/c1_adc.nii.gz"},
    {"source": "ISLES_ltr", "kind": "label", "case_id": "c1", "relpath": "isles/derivatives/c1/c1_msk.nii.gz"},
]
SKM_ASSETS = [
    {"source": "SKM-TEA_ltr", "kind": "image", "case_id": "k1", "relpath": "image_files/k1.h5"},
    {"source": "SKM-TEA_ltr", "kind": "label", "case_id": "k1", "relpath": "segmentation_masks/dicom-track/k1.nii.gz"},
]
SKM_META = {"k1": {"subject_id": 7, "scan_id": 1, "timepoint": 0, "inspected": True, "split_release": "val",
                   "boxes": [{"supercategory": "Meniscal Tear", "category_id": 1, "tissue_id": 4, "confidence": 3}]}}
A6_MAPPING = [c + 1 for c in range(6)]           # channel c -> dicom-track label value


def _isles_record() -> dict:
    return build_registration_records("isles_u", ISLES_ASSETS)[0]


def _skm_record() -> dict:
    return build_registration_records("skm_a", SKM_ASSETS, SKM_META)[0]


def _rows(source: str, refs: list, status: str = "pass") -> list:
    return [{"source": source, "case_id": "c", "image_relpath": ref, "status": status,
             "reasons": [] if status == "pass" else ["AFFINE_MISMATCH"]} for ref in refs]


def _channel(transform: str = "transpose_xy", mapping: dict | None = None, consistent: bool = True) -> dict:
    best = {"transform": transform, "consistent": consistent,
            "channel_to_label": A6_MAPPING if mapping is None else mapping}
    return {"tracks": {"dicom_track_relpath": {"best": best}, "raw_track_relpath": None}}


def _policy(source_id: str, **filled) -> object:
    entry = yaml.safe_load((CONFIG / "sources_v5.template.yaml").read_text())[source_id]
    entry.update(filled)
    return validate_source_policy_v5(source_id, entry)


def test_isles_record_is_prepared_and_migrates():
    record = _isles_record()
    prepared = prepare_record_v5(record, ISLES_SCOPE, _rows("ISLES_ltr", record["image_paths"]), GEOMETRY)
    assert isinstance(prepared, SampleRecord)
    assert (prepared.geometry_status, prepared.image_geometry) == ("pass", GEOMETRY)
    assert prepared.modality_names == ["dwi", "adc"]
    (annotation,) = prepared.annotation_records
    assert annotation.label_map_version == "isles_binary_v1"
    # Proposal_v5 §2.3: the released-reference track does not pseudo-fill "visible"
    assert annotation.input_visibility == "unknown"
    policy = _policy("isles_2022", annotation_modalities={"U": ["dwi", "adc", "flair"]}, index_evidence="qc",
                     physical_units_policy="trust_header", physical_units_evidence="header")
    migrated = migrate_v41_record_v5(prepared, policy, ISLES_SCOPE)
    assert (migrated.index_alignment_status, migrated.physical_geometry_status) == ("verified", "verified")
    assert record == _isles_record()                                      # input not modified


@pytest.mark.parametrize("rows,expected", [
    (lambda refs: _rows("ISLES_ltr", refs[:1]) + _rows("ISLES_ltr", refs[1:], "fail"), "fail"),
    (lambda refs: _rows("ISLES_ltr", refs[:1]), "not_checked"),               # one model image unchecked
    (lambda refs: _rows("SKM-TEA_ltr", refs), "not_checked"),                 # rows of another source
    (lambda refs: [], "not_checked"),
])
def test_isles_geometry_status_needs_every_model_image(rows, expected):
    record = _isles_record()
    prepared = prepare_record_v5(record, ISLES_SCOPE, rows(record["image_paths"]), GEOMETRY)
    assert prepared.geometry_status == expected


def test_skm_record_reads_h5_datasets_and_keeps_scope_tasks():
    record = _skm_record()
    h5 = record["image_paths"][0]
    prepared = prepare_record_v5(record, SKM_SCOPE, _rows("SKM-TEA_ltr", [h5]), GEOMETRY, _channel())
    assert prepared.geometry_status == "pass"
    assert prepared.image_paths == [f"{h5}#echo1", f"{h5}#echo2"]
    assert prepared.modality_names == ["qdess_e1", "qdess_e2"]
    assert prepared.modality_is_synthetic == [False, False] and prepared.available_modality_mask == [True, True]
    (annotation,) = prepared.annotation_records                             # the U box is not an skm_a task
    assert (annotation.task, annotation.path, annotation.label_map_version) == (
        "A", f"{h5}#seg", "skm_a6_h5_channels_v1")
    assert annotation.input_visibility == "visible"
    assert prepared.provenance["release_split"] == "val"
    policy = _policy("skm_tea", annotation_modalities={"A": ["qdess_e1", "qdess_e2"]}, index_evidence="channels")
    migrated = migrate_v41_record_v5(prepared, policy, SKM_SCOPE)
    assert (migrated.index_alignment_status, migrated.physical_geometry_status) == ("verified", "unknown")


@pytest.mark.parametrize("channel,expected", [
    (None, "not_checked"),
    (_channel(transform="transpose_xy_flip_z"), "pass"),          # 18/155 real scans: dicom track also z-flipped
    (_channel(transform="identity"), "fail"),
    (_channel(consistent=False), "fail"),
    (_channel(mapping=[2, 1, 3, 4, 5, 6]), "fail"),               # channel order differs from the map
])
def test_skm_geometry_needs_the_verified_channel_transform(channel, expected):
    record = _skm_record()
    prepared = prepare_record_v5(record, SKM_SCOPE, _rows("SKM-TEA_ltr", record["image_paths"]), GEOMETRY, channel)
    assert prepared.geometry_status == expected


def test_checked_geometry_needs_image_geometry():
    record = _isles_record()
    with pytest.raises(V5Error) as error:
        prepare_record_v5(record, ISLES_SCOPE, _rows("ISLES_ltr", record["image_paths"]), None)
    assert (error.value.code, error.value.path) == ("GEOMETRY_MISSING", "$.image_geometry")


def test_unsupported_source_or_scope_is_refused():
    record = _isles_record()
    with pytest.raises(V5Error) as error:
        prepare_record_v5({**record, "source_dataset": "other"}, ISLES_SCOPE, [], GEOMETRY)
    assert (error.value.code, error.value.path) == ("SOURCE_NOT_SUPPORTED", "$.source_dataset")
    with pytest.raises(V5Error) as error:
        prepare_record_v5(record, SKM_SCOPE, [], GEOMETRY)
    assert (error.value.code, error.value.path) == ("SCOPE_MISMATCH", "$.source_dataset")


def test_unknown_label_map_is_refused():
    record = _isles_record()
    record["annotation_records"][0]["label_map_version"] = "unknown_map"
    with pytest.raises(V5Error) as error:
        prepare_record_v5(record, ISLES_SCOPE, [], GEOMETRY)
    assert (error.value.code, error.value.path) == ("LABEL_MAP_UNKNOWN", "$.annotation_records[0].label_map_version")


def test_skm_h5_geometry_swaps_the_first_two_axes():
    affine = np.array([[0.3, 0.01, 0, 10], [0, 0, 0.8, -5], [0.02, -0.31, 0, 3], [0, 0, 0, 1]])
    nifti = {"shape": [4, 6, 5], "spacing_mm": [0.3, 0.31, 0.8], "affine": affine.tolist(),
             "axis_codes": ["R", "I", "A"], "units": "unknown"}
    out = skm_h5_geometry_v5(nifti, units="mm")
    assert (out["shape"], out["spacing_mm"], out["units"]) == ([6, 4, 5], [0.31, 0.3, 0.8], "mm")
    assert set(out) == {"shape", "spacing_mm", "affine", "axis_codes", "units"}
    h5_affine = np.asarray(out["affine"])
    for i, j, k in [(0, 0, 0), (5, 3, 4), (2, 1, 3)]:                     # h5[i, j, k] == nifti[j, i, k]
        assert np.allclose(h5_affine @ [i, j, k, 1], affine @ [j, i, k, 1])
    assert out["axis_codes"] == list(nib.aff2axcodes(h5_affine))


def test_skm_h5_geometry_follows_a_z_flipped_track():
    affine = np.array([[0.3, 0.01, 0, 10], [0, 0, 0.8, -5], [0.02, -0.31, 0, 3], [0, 0, 0, 1]])
    nifti = {"shape": [4, 6, 5], "spacing_mm": [0.3, 0.31, 0.8], "affine": affine.tolist(),
             "axis_codes": ["R", "I", "A"], "units": "unknown"}
    out = skm_h5_geometry_v5(nifti, units="mm", transform="transpose_xy_flip_z")
    assert (out["shape"], out["spacing_mm"]) == ([6, 4, 5], [0.31, 0.3, 0.8])
    h5_affine = np.asarray(out["affine"])
    for i, j, k in [(0, 0, 0), (5, 3, 4), (2, 1, 3)]:                     # h5[i, j, k] == nifti[j, i, 4 - k]
        assert np.allclose(h5_affine @ [i, j, k, 1], affine @ [j, i, 4 - k, 1])
    with pytest.raises(V5Error) as error:
        skm_h5_geometry_v5(nifti, units="mm", transform="identity")
    assert error.value.code == "SKM_TRANSFORM_UNSUPPORTED"


def test_qc_csv_rows_round_trip(tmp_path):
    rows = [{"source": "ISLES_ltr", "case_id": "c1", "image_relpath": "a", "status": "fail",
             "reasons": ["AFFINE_MISMATCH"], "label_shape": [8, 8, 4]}]
    _csv(tmp_path / "g.csv", rows, ["source", "case_id", "image_relpath", "status", "reasons", "label_shape"])
    assert read_qc_csv_v5(tmp_path / "g.csv") == rows


# ---------------------------------------------------------------------------
# Driver on a synthetic tree


def _nifti(path: Path, array: np.ndarray, affine: np.ndarray, units: str = "mm") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array, affine)
    image.header.set_xyzt_units(xyz=units)
    nib.save(image, str(path))


def _isles_tree(root: Path) -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    for name in ("dwi", "adc"):
        _nifti(root / f"isles/c1/c1_{name}.nii.gz", rng.random((8, 8, 4)).astype(np.float32), affine)
    mask = np.zeros((8, 8, 4), dtype=np.uint16)
    mask[2:4, 2:4, 2] = 1
    mask[6, 6, 3] = 1                                                     # vanishes at factor 2
    _nifti(root / "isles/derivatives/c1/c1_msk.nii.gz", mask, affine)


def _skm_tree(root: Path) -> None:
    seg = np.zeros((6, 4, 5, 6), dtype=bool)                              # h5 grid [x, y, z, C]
    for c in range(6):
        seg[c, :, c % 5, c] = True
    labels = np.zeros((4, 6, 5), dtype=np.uint8)                          # nifti = h5 with x/y swapped
    for c in range(6):
        labels[:, c, c % 5] = c + 1
    (root / "image_files").mkdir(parents=True)
    rng = np.random.default_rng(1)
    with h5py.File(root / "image_files/k1.h5", "w") as handle:
        handle["echo1"] = (rng.random((6, 4, 5)) * 100).astype(np.int16) + 1
        handle["echo2"] = (rng.random((6, 4, 5)) * 100).astype(np.int16) + 1
        handle["seg"] = seg
    _nifti(root / "segmentation_masks/dicom-track/k1.nii.gz", labels, np.diag([0.3, 0.3, 0.8, 1.0]), "unknown")


def test_driver_prepares_isles_records_privately(tmp_path):
    raw, private = tmp_path / "raw", tmp_path / "private"
    _isles_tree(raw)
    private.mkdir(mode=0o700)
    report = run_prepare_v5(raw, ISLES_ASSETS, ISLES_SCOPE, _policy("isles_2022"), private / "t07_isles",
                            factor=(2, 2, 2))
    out = private / "t07_isles"
    assert oct(out.stat().st_mode & 0o777) == "0o700"
    for name in ("records_prepared.jsonl", "source_check.md", "loss_reports.json"):
        assert oct((out / name).stat().st_mode & 0o777) == "0o600"
    (prepared,) = [json.loads(line) for line in (out / "records_prepared.jsonl").read_text().splitlines()]
    assert prepared["geometry_status"] == "pass" and prepared["image_geometry"]["units"] == "mm"
    assert report["records"] == 1 and report["failures"] == []
    assert report["geometry_status"] == {"pass": 1}
    md = (out / "source_check.md").read_text()
    assert "c1" not in md and "isles/" not in md and "2×2×2" in md
    assert report["summary"]["downsampling"]["classes"]["ischemic_infarct"]["components_vanished"] == 1
    with pytest.raises(Exception):                                        # outputs are never overwritten
        run_prepare_v5(raw, ISLES_ASSETS, ISLES_SCOPE, _policy("isles_2022"), out, factor=(2, 2, 2))


def _skm_tree_flipped(root: Path) -> None:
    _skm_tree(root)
    labels = np.zeros((4, 6, 5), dtype=np.uint8)
    for c in range(6):
        labels[:, c, 4 - c % 5] = c + 1                                   # nifti = h5 with x/y swapped and z flipped
    (root / "segmentation_masks/dicom-track/k1.nii.gz").unlink()
    _nifti(root / "segmentation_masks/dicom-track/k1.nii.gz", labels, np.diag([0.3, 0.3, 0.8, 1.0]), "unknown")
    raw_track = np.zeros((6, 4, 5), dtype=np.uint8)                       # unregistered track, never consistent
    _nifti(root / "segmentation_masks/raw-data-track/k1.nii.gz", raw_track, np.diag([0.3, 0.3, 0.8, 1.0]), "unknown")


def test_driver_counts_only_the_registered_track_and_can_reuse_qc(tmp_path):
    raw, private = tmp_path / "raw", tmp_path / "private"
    _skm_tree_flipped(raw)
    private.mkdir(mode=0o700)
    assets = SKM_ASSETS + [{"source": "SKM-TEA_ltr", "kind": "label", "case_id": "k1",
                            "relpath": "segmentation_masks/raw-data-track/k1.nii.gz"}]
    first = run_prepare_v5(raw, assets, SKM_SCOPE, _policy("skm_tea"), private / "a", factor=(2, 2, 1),
                           skm_annotations=copy.deepcopy(SKM_META), skm_units="mm")
    assert first["geometry_status"] == {"pass": 1}
    assert first["summary"]["channel"] == {"cases_checked": 1, "consistent": 1, "unevaluated": 0}
    (prepared,) = [json.loads(line) for line in (private / "a/records_prepared.jsonl").read_text().splitlines()]
    affine = np.asarray(prepared["image_geometry"]["affine"])
    assert np.allclose(affine @ [0, 0, 0, 1], np.diag([0.3, 0.3, 0.8, 1.0]) @ [0, 0, 4, 1])
    again = run_prepare_v5(raw, assets, SKM_SCOPE, _policy("skm_tea"), private / "b", factor=(2, 2, 1),
                           skm_annotations=copy.deepcopy(SKM_META), skm_units="mm", reuse_qc=private / "a/geometry_qc")
    assert not (private / "b/geometry_qc").exists()
    assert {k: again[k] for k in ("records", "failures", "geometry_status", "summary")} == \
        {k: first[k] for k in ("records", "failures", "geometry_status", "summary")}
    assert (private / "b/records_prepared.jsonl").read_text() == (private / "a/records_prepared.jsonl").read_text()


def test_driver_prepares_skm_records_from_h5(tmp_path):
    raw, private = tmp_path / "raw", tmp_path / "private"
    _skm_tree(raw)
    private.mkdir(mode=0o700)
    report = run_prepare_v5(raw, SKM_ASSETS, SKM_SCOPE, _policy("skm_tea"), private / "t07_skm",
                            factor=(2, 2, 1), skm_annotations=copy.deepcopy(SKM_META), skm_units="mm")
    (prepared,) = [json.loads(line) for line in (private / "t07_skm/records_prepared.jsonl").read_text().splitlines()]
    assert prepared["geometry_status"] == "pass"
    assert prepared["image_geometry"]["shape"] == [6, 4, 5]
    assert prepared["image_geometry"]["spacing_mm"] == pytest.approx([0.3, 0.3, 0.8])
    assert report["summary"]["channel"] == {"cases_checked": 1, "consistent": 1, "unevaluated": 0}
    assert "k1" not in (private / "t07_skm/source_check.md").read_text()


def test_command_line_prepares_and_migrates_with_counts_only(tmp_path, capsys):
    from anatobind.v41.real_t07_v5 import main
    raw, private = tmp_path / "raw", tmp_path / "private"
    _isles_tree(raw)
    private.mkdir(mode=0o700)
    (tmp_path / "assets.jsonl").write_text("".join(json.dumps(a) + "\n" for a in ISLES_ASSETS))
    policies = yaml.safe_load((CONFIG / "sources_v5.template.yaml").read_text())
    policies["isles_2022"].update(annotation_modalities={"U": ["dwi", "adc", "flair"]}, index_evidence="qc",
                                  physical_units_policy="trust_header", physical_units_evidence="header")
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(policies))
    common = ["--scope", str(CONFIG / "scope_isles_u.yaml"), "--policies", str(tmp_path / "sources.yaml"),
              "--source-id", "isles_2022"]
    assert main(["prepare", "--raw", str(raw), "--assets", str(tmp_path / "assets.jsonl"),
                 "--out", str(private / "t07"), "--factor", "2", "2", "2", *common]) == 0
    printed = capsys.readouterr().out
    assert set(json.loads(printed)) == {"records", "failures", "geometry_status", "source_check_sha256"}
    assert json.loads(printed)["records"] == 1 and "isles/" not in printed
    assert main(["migrate", "--prepared", str(private / "t07/records_prepared.jsonl"),
                 "--out", str(private / "t07/records_v5.jsonl"), *common]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {"records": 1, "failures": [], "index_alignment": {"verified": 1},
                      "physical_geometry": {"verified": 1}}
    assert oct((private / "t07/records_v5.jsonl").stat().st_mode & 0o777) == "0o600"
