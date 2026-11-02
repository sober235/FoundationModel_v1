import h5py
import nibabel as nib
import numpy as np

from anatobind.v41.geometry_qc import run_geometry_qc


def test_geometry_qc_writes_three_reports_and_flags_mismatch(tmp_path):
    root = tmp_path / "raw"
    case = root / "ISLES_ltr" / "ISLES-2022"
    (case / "sub-0001" / "ses-0001" / "dwi").mkdir(parents=True)
    (case / "sub-0001" / "ses-0001" / "anat").mkdir(parents=True)
    (case / "derivatives" / "sub-0001" / "ses-0001").mkdir(parents=True)
    mask = np.zeros((4, 4, 4), dtype=np.int16)
    mask[1, 1, 1] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(case / "derivatives" / "sub-0001" / "ses-0001" / "sub-0001_ses-0001_msk.nii.gz"))
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), str(case / "sub-0001" / "ses-0001" / "dwi" / "sub-0001_ses-0001_dwi.nii.gz"))
    shifted = np.eye(4)
    shifted[0, 3] = 2.0
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), shifted), str(case / "sub-0001" / "ses-0001" / "dwi" / "sub-0001_ses-0001_adc.nii.gz"))
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)), str(case / "sub-0001" / "ses-0001" / "anat" / "sub-0001_ses-0001_FLAIR.nii.gz"))
    assets = [
        {"source": "ISLES_ltr", "kind": "label", "case_id": "sub-0001", "relpath": "ISLES_ltr/ISLES-2022/derivatives/sub-0001/ses-0001/sub-0001_ses-0001_msk.nii.gz"},
        {"source": "ISLES_ltr", "kind": "image", "case_id": "sub-0001", "relpath": "ISLES_ltr/ISLES-2022/sub-0001/ses-0001/dwi/sub-0001_ses-0001_dwi.nii.gz"},
        {"source": "ISLES_ltr", "kind": "image", "case_id": "sub-0001", "relpath": "ISLES_ltr/ISLES-2022/sub-0001/ses-0001/dwi/sub-0001_ses-0001_adc.nii.gz"},
        {"source": "ISLES_ltr", "kind": "image", "case_id": "sub-0001", "relpath": "ISLES_ltr/ISLES-2022/sub-0001/ses-0001/anat/sub-0001_ses-0001_FLAIR.nii.gz"},
    ]
    out = tmp_path / "qc"
    summary = run_geometry_qc(root, assets, out)
    assert (out / "geometry_qc.csv").exists() and (out / "label_value_qc.csv").exists() and (out / "image_label_pairing.csv").exists()
    status = summary["geometry_by_source_role_status"]
    assert status["ISLES_ltr|dwi|pass|-"] == 1
    assert status["ISLES_ltr|adc|fail|AFFINE_MISMATCH"] == 1
    assert not any("|flair|" in k for k in status)
    assert summary["label_value_qc_by_source"] == {"ISLES_ltr|pass|-|unexpected=False": 1}
    assert summary["pairing_by_source_status"] == {"ISLES_ltr|paired": 1}
