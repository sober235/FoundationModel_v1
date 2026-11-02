import copy

import pytest

SECRET_PATH = "/private/zz_patient_root/case_000777"
SECRET_CASE = "CASE_SECRET_000777"


def _annotation():
    return {
        "annotation_uid": "ann-synth-0001",
        "task": "U",
        "class_scope": ["cartilage_lesion"],
        "representation": "voxel",
        "path": "annotations/synth/ann-synth-0001.nii.gz",
        "label_map_version": "synth-u-v1",
        "provenance_type": "manual",
        "creator_or_model": "synthetic_reader",
        "creator_version": "0",
        "parent_annotation_uid": None,
        "completeness": "target_complete",
        "review_region_path": None,
        "input_visibility": "visible",
        "quality_status": "accepted",
        "intended_role": "train",
        "reference_role": "expert_voxel",
        "adjudication_status": "single_reader",
        "reader_ids": ["reader_synth_a"],
        "protocol_hash": "0" * 64,
        "visibility_evidence": "synthetic",
    }


def _sample():
    return {
        "schema_version": "v41.0",
        "sample_uid": "synth-0001",
        "source_dataset": "synthetic",
        "source_release": "unknown",
        "cohort": "synthetic",
        "patient_uid": "synthetic:p0001",
        "study_uid": "synthetic:p0001:s1",
        "session_uid": "synthetic:p0001:s1:ses1",
        "parent_scan_uid": None,
        "derivative_uid": None,
        "leakage_group_uid": "synthetic:p0001",
        "exposure_history": [],
        "split": "unassigned",
        "native_acquisition_dim": 3,
        "observation_mode": "3d",
        "image_paths": ["images/synth/synth-0001_e1.nii.gz"],
        "modality_names": ["qdess_e1"],
        "modality_is_synthetic": [False],
        "available_modality_mask": [True],
        "image_geometry": {
            "shape": [8, 8, 8],
            "spacing_mm": [1.0, 1.0, 1.0],
            "affine": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "axis_codes": ["R", "A", "S"],
            "units": "mm",
        },
        "original_geometry": None,
        "transform_chain": [],
        "geometry_status": "not_checked",
        "label_value_qc": "not_checked",
        "anatomy_ontology_version": "skm_a6_v1",
        "abnormality_ontology_version": "u_v1",
        "relation_semantics_version": "r_v1",
        "annotation_records": [_annotation()],
        "review_regions": [],
        "visibility_records": [],
        "quality_records": [],
        "provenance": {"ingest_commit": "0" * 40, "ingest_time": "2026-11-02T00:00:00Z"},
    }


@pytest.fixture
def annotation_dict():
    return copy.deepcopy(_annotation())


@pytest.fixture
def sample_dict():
    return copy.deepcopy(_sample())


@pytest.fixture
def secret_values():
    return SECRET_PATH, SECRET_CASE
