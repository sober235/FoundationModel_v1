import json

import pytest

from anatobind.v41.audits import (
    build_registration_records,
    eligibility_rows,
    load_skm_annotations,
    provenance_rows,
    summarize,
    visibility_rows,
    write_scope_audit,
)
from anatobind.v41.schema import validate_sample


def _isles_assets(n=2):
    assets = []
    for i in range(1, n + 1):
        case = f"sub-{i:04d}"
        for role, sub in (("dwi", "dwi"), ("adc", "dwi"), ("flair", "anat")):
            assets.append({"source": "ISLES_ltr", "kind": "image", "case_id": case,
                           "relpath": f"ISLES_ltr/ISLES-2022/{case}/ses-0001/{sub}/{case}_ses-0001_{role}.nii.gz"})
        assets.append({"source": "ISLES_ltr", "kind": "label", "case_id": case,
                       "relpath": f"ISLES_ltr/ISLES-2022/derivatives/{case}/ses-0001/{case}_ses-0001_msk.nii.gz"})
    return assets


def _skm_assets(cases=("CASE_001", "CASE_002")):
    assets = []
    for case in cases:
        assets.append({"source": "SKM-TEA_ltr", "kind": "image", "case_id": case,
                       "relpath": f"SKM-TEA_ltr/image_files/{case}.h5"})
        for track in ("dicom-track", "raw-data-track"):
            assets.append({"source": "SKM-TEA_ltr", "kind": "label", "case_id": case,
                           "relpath": f"SKM-TEA_ltr/segmentation_masks/{track}/{case}.nii.gz"})
    return assets


def _skm_annotations():
    return {
        "CASE_001": {"subject_id": 7, "scan_id": 101, "timepoint": 0, "inspected": True,
                    "split_release": "train",
                    "boxes": [{"supercategory": "Cartilage Lesion", "category_id": 12, "tissue_id": 4, "confidence": 3.0},
                              {"supercategory": "Meniscal Tear", "category_id": 1, "tissue_id": 1, "confidence": 0.0}]},
        "CASE_002": {"subject_id": 7, "scan_id": 102, "timepoint": 1, "inspected": True,
                    "split_release": "train", "boxes": []},
    }


def test_isles_records_validate_and_carry_exposure():
    records = build_registration_records("isles_u", _isles_assets())
    assert len(records) == 2
    for record in records:
        validated = validate_sample(record)
        assert validated.split == "unassigned"
        assert validated.exposure_history[0]["role"] == "unknown"
        assert validated.annotation_records[0].reference_role == "expert_voxel"
        assert validated.annotation_records[0].completeness == "unknown"
        assert validated.image_geometry is None
        assert validated.geometry_status == "not_checked"


def test_skm_records_group_by_subject_and_boxes_are_weak():
    records = build_registration_records("skm_a", _skm_assets(), _skm_annotations())
    assert len(records) == 2
    for record in records:
        validate_sample(record)
    assert records[0]["leakage_group_uid"] == records[1]["leakage_group_uid"] == "skmtea:subject:7"
    anns = records[0]["annotation_records"]
    roles = {(a["task"], a["representation"]): a["reference_role"] for a in anns}
    assert roles[("A", "voxel")] == "expert_voxel"
    assert roles[("U", "bbox")] == "weak_localization"
    rejected = [a for a in anns if a["quality_status"] == "rejected"]
    assert len(rejected) == 1 and rejected[0]["representation"] == "bbox"
    modalities = records[0]["modality_names"]
    assert len(modalities) == len(records[0]["image_paths"]) == len(records[0]["available_modality_mask"])


def test_provenance_visibility_eligibility_rows():
    records = build_registration_records("skm_a", _skm_assets(), _skm_annotations())
    prov = provenance_rows(records)
    assert summarize(prov, "task", "reference_role") == {
        "A, expert_voxel": 4, "U, weak_localization": 2}
    vis = visibility_rows(records)
    assert all(row["image_count"] == 1 for row in vis)
    elig = eligibility_rows(records)
    by_role = summarize(elig, "task", "role")
    assert by_role["A, qualified"] == 4
    assert by_role["U, qualified"] == 1
    assert by_role["U, none"] == 1
    assert all(row["localization_only"] for row in elig if row["task"] == "U" and row["role"] == "qualified")
    neg = [row for row in elig if row["task"] == "A"]
    assert all(len(row["negative_classes"]) == 6 for row in neg)


def test_isles_eligibility_has_no_supervision_until_protocol_verified():
    records = build_registration_records("isles_u", _isles_assets())
    elig = eligibility_rows(records)
    assert {row["role"] for row in elig} == {"none"}
    assert {row["reason"] for row in elig} == {"COMPLETENESS_UNKNOWN"}


def test_write_scope_audit_outputs_and_immutability(tmp_path):
    out = tmp_path / "audit"
    summary = write_scope_audit("isles_u", _isles_assets(), out)
    assert summary["records"] == 2 and summary["patients"] == 2
    for name in ("annotation_provenance_report.csv", "input_visibility_report.csv",
                 "eligibility_coverage.csv", "registration_records.jsonl"):
        assert (out / name).exists()
    with pytest.raises(ValueError, match="PRIVATE_FILE_EXISTS"):
        write_scope_audit("isles_u", _isles_assets(), out)
    with pytest.raises(ValueError, match="UNKNOWN_SCOPE"):
        write_scope_audit("pdgm_u", [], tmp_path / "other")


def test_load_skm_annotations_reads_subject_and_boxes(tmp_path):
    doc = {
        "categories": [{"id": 12, "supercategory": "Cartilage Lesion"}],
        "images": [{"id": 1, "file_name": "CASE_001.h5", "subject_id": 3, "scan_id": 11,
                    "timepoint": 0, "inspected": True}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 12, "tissue_id": 4, "confidence": 2.0}],
    }
    (tmp_path / "train.json").write_text(json.dumps(doc))
    loaded = load_skm_annotations(tmp_path)
    assert loaded["CASE_001"]["subject_id"] == 3
    assert loaded["CASE_001"]["boxes"][0]["supercategory"] == "Cartilage Lesion"
    assert loaded["CASE_001"]["split_release"] == "train"
