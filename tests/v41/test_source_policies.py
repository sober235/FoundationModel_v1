"""Source policies decided by the researcher on 2026-11-02.

ISLES: FLAIR is not a model input; release masks are expert voxel references
for acute/sub-acute ischemic infarct (ISLES 2022 published protocol).
SKM-TEA: the H5 seg (== xy-transposed dicom-track NIfTI) is the only A
reference; the raw-data-track masks are not registered.
"""
import h5py
import nibabel as nib
import numpy as np

from anatobind.v41.adapters import isles, skm
from anatobind.v41.audits import build_registration_records, eligibility_rows
from anatobind.v41.geometry_qc import run_geometry_qc
from anatobind.v41.registry import check_reference_qualification
from anatobind.v41.schema import validate_sample


def _isles_assets():
    case = "sub-0001"
    assets = [{"source": "ISLES_ltr", "kind": "label", "case_id": case,
               "relpath": f"ISLES_ltr/ISLES-2022/derivatives/{case}/ses-0001/{case}_ses-0001_msk.nii.gz"}]
    for role, sub in (("dwi", "dwi"), ("adc", "dwi"), ("FLAIR", "anat")):
        assets.append({"source": "ISLES_ltr", "kind": "image", "case_id": case,
                       "relpath": f"ISLES_ltr/ISLES-2022/{case}/ses-0001/{sub}/{case}_ses-0001_{role}.nii.gz"})
    return assets


def test_isles_pairing_excludes_flair():
    pairs = isles.pair_assets(_isles_assets())
    assert set(pairs[0]["images"]) == {"dwi", "adc"}


def test_isles_records_are_dwi_adc_expert_infarct_references():
    record = validate_sample(build_registration_records("isles_u", _isles_assets())[0])
    assert record.modality_names == ["dwi", "adc"]
    assert not any("flair" in p.lower() for p in record.image_paths)
    ann = record.annotation_records[0]
    assert ann.reference_role == "expert_voxel"
    assert ann.completeness == "target_complete"
    assert ann.class_scope == ["ischemic_infarct"]
    check_reference_qualification(record)


def test_isles_masks_supervise_infarct_positives_and_negatives():
    rows = eligibility_rows(build_registration_records("isles_u", _isles_assets()))
    assert [r["role"] for r in rows] == ["qualified"]
    assert rows[0]["positive_classes"] == rows[0]["negative_classes"] == ["ischemic_infarct"]


def _skm_assets(case="CASE_001"):
    return [{"source": "SKM-TEA_ltr", "kind": "image", "case_id": case,
             "relpath": f"SKM-TEA_ltr/image_files/{case}.h5"}] + [
        {"source": "SKM-TEA_ltr", "kind": "label", "case_id": case,
         "relpath": f"SKM-TEA_ltr/segmentation_masks/{track}/{case}.nii.gz"}
        for track in ("dicom-track", "raw-data-track")]


def test_skm_registers_only_dicom_track_as_anatomy_reference():
    record = validate_sample(build_registration_records("skm_a", _skm_assets(), {})[0])
    anatomy = [a for a in record.annotation_records if a.task == "A"]
    assert [a.path for a in anatomy] == ["SKM-TEA_ltr/segmentation_masks/dicom-track/CASE_001.nii.gz"]
    assert anatomy[0].reference_role == "expert_voxel"


def _write_skm(root, case, nifti_shape, seg_shape, raw=True):
    (root / "SKM-TEA_ltr" / "image_files").mkdir(parents=True, exist_ok=True)
    for track in ("dicom-track", "raw-data-track"):
        (root / "SKM-TEA_ltr" / "segmentation_masks" / track).mkdir(parents=True, exist_ok=True)
    with h5py.File(root / "SKM-TEA_ltr" / "image_files" / f"{case}.h5", "w") as handle:
        handle.create_dataset("seg", data=np.zeros((*seg_shape, 6), bool))
    labels = np.zeros(nifti_shape, np.int16)
    labels[0, 0, 0] = 2
    nib.save(nib.Nifti1Image(labels, np.eye(4)),
             str(root / "SKM-TEA_ltr" / "segmentation_masks" / "dicom-track" / f"{case}.nii.gz"))
    if raw:
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), np.int16), np.diag([2, 2, 2, 1])),
                 str(root / "SKM-TEA_ltr" / "segmentation_masks" / "raw-data-track" / f"{case}.nii.gz"))


def test_skm_geometry_qc_checks_dicom_track_against_h5_seg_grid(tmp_path):
    root = tmp_path / "raw"
    _write_skm(root, "CASE_001", (6, 5, 4), (5, 6, 4))
    _write_skm(root, "CASE_002", (6, 5, 4), (6, 5, 4))
    summary = run_geometry_qc(root, _skm_assets("CASE_001") + _skm_assets("CASE_002"), tmp_path / "qc")
    status = summary["geometry_by_source_role_status"]
    assert status == {"SKM-TEA_ltr|dicom_track_vs_h5_seg|pass|-": 1,
                      "SKM-TEA_ltr|dicom_track_vs_h5_seg|fail|SHAPE_MISMATCH": 1}
    assert summary["pairing_by_source_status"] == {"SKM-TEA_ltr|paired": 2}
    assert summary["label_value_qc_by_source"] == {"SKM-TEA_ltr|pass|-|unexpected=False": 2}


def test_isles_geometry_qc_has_no_flair_rows(tmp_path):
    root = tmp_path / "raw"
    base = root / "ISLES_ltr" / "ISLES-2022"
    for sub in ("sub-0001/ses-0001/dwi", "sub-0001/ses-0001/anat", "derivatives/sub-0001/ses-0001"):
        (base / sub).mkdir(parents=True)
    mask = np.zeros((4, 4, 4), np.int16)
    mask[1, 1, 1] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(base / "derivatives/sub-0001/ses-0001/sub-0001_ses-0001_msk.nii.gz"))
    for role, sub, shape in (("dwi", "dwi", (4, 4, 4)), ("adc", "dwi", (4, 4, 4)), ("FLAIR", "anat", (8, 8, 8))):
        nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4)),
                 str(base / f"sub-0001/ses-0001/{sub}/sub-0001_ses-0001_{role}.nii.gz"))
    summary = run_geometry_qc(root, _isles_assets(), tmp_path / "qc")
    assert summary["geometry_by_source_role_status"] == {"ISLES_ltr|adc|pass|-": 1, "ISLES_ltr|dwi|pass|-": 1}
