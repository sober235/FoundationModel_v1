import h5py
import nibabel as nib
import numpy as np

from anatobind.v41.adapters import isles, skm


def test_isles_pairing_groups_mask_with_dwi_and_adc():
    assets = [
        {"source": "ISLES_ltr", "kind": "image", "case_id": "sub-0001", "relpath": "ISLES_ltr/ISLES-2022/sub-0001/ses-0001/dwi/sub-0001_ses-0001_dwi.nii.gz"},
        {"source": "ISLES_ltr", "kind": "image", "case_id": "sub-0001", "relpath": "ISLES_ltr/ISLES-2022/sub-0001/ses-0001/dwi/sub-0001_ses-0001_adc.nii.gz"},
        {"source": "ISLES_ltr", "kind": "image", "case_id": "sub-0001", "relpath": "ISLES_ltr/ISLES-2022/sub-0001/ses-0001/anat/sub-0001_ses-0001_FLAIR.nii.gz"},
        {"source": "ISLES_ltr", "kind": "label", "case_id": "sub-0001", "relpath": "ISLES_ltr/ISLES-2022/derivatives/sub-0001/ses-0001/sub-0001_ses-0001_msk.nii.gz"},
        {"source": "ISLES_ltr", "kind": "archive", "case_id": None, "relpath": "ISLES_ltr/ISLES-2022.zip"},
        {"source": "SKM-TEA_ltr", "kind": "image", "case_id": "MTR_001", "relpath": "SKM-TEA_ltr/image_files/MTR_001.h5"},
    ]
    pairs = isles.pair_assets(assets)
    assert len(pairs) == 1
    assert set(pairs[0]["images"]) == {"dwi", "adc"}
    assert pairs[0]["label_relpath"].endswith("_msk.nii.gz")


def test_skm_pairing_and_channel_check(tmp_path):
    labels = np.zeros((6, 5, 4), dtype=np.int16)
    labels[0, :, :] = 1
    labels[1, :, :] = 2
    labels[2, :, :] = 6
    seg = np.zeros((5, 6, 4, 6), dtype=bool)
    transposed = np.transpose(labels, (1, 0, 2))
    seg[..., 0] = transposed == 1
    seg[..., 1] = transposed == 2
    seg[..., 5] = transposed == 6
    root = tmp_path
    (root / "SKM-TEA_ltr" / "image_files").mkdir(parents=True)
    (root / "SKM-TEA_ltr" / "segmentation_masks" / "dicom-track").mkdir(parents=True)
    with h5py.File(root / "SKM-TEA_ltr" / "image_files" / "MTR_001.h5", "w") as handle:
        handle.create_dataset("seg", data=seg)
    nib.save(nib.Nifti1Image(labels, np.eye(4)), str(root / "SKM-TEA_ltr" / "segmentation_masks" / "dicom-track" / "MTR_001.nii.gz"))
    assets = [
        {"source": "SKM-TEA_ltr", "kind": "image", "case_id": "MTR_001", "relpath": "SKM-TEA_ltr/image_files/MTR_001.h5"},
        {"source": "SKM-TEA_ltr", "kind": "label", "case_id": "MTR_001", "relpath": "SKM-TEA_ltr/segmentation_masks/dicom-track/MTR_001.nii.gz"},
    ]
    pairs = skm.pair_assets(assets)
    assert pairs[0]["raw_track_relpath"] is None
    report = skm.channel_check(root, pairs[0])
    best = report["tracks"]["dicom_track_relpath"]["best"]
    assert best["transform"] == "transpose_xy"
    assert best["channel_to_label"][:2] == [1, 2] and best["channel_to_label"][5] == 6
    assert best["consistent"] is False
    assert report["tracks"]["raw_track_relpath"] is None


def test_skm_nifti_to_h5_is_verified_xy_transpose():
    import numpy as np

    from anatobind.v41.adapters import skm

    labels = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    assert skm.NIFTI_TO_H5_TRANSFORM == "transpose_xy"
    assert np.array_equal(skm.nifti_to_h5(labels), np.transpose(labels, (1, 0, 2)))
