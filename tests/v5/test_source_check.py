from __future__ import annotations

import numpy as np
import pytest

from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.geometry_v5 import downsample_loss_report_v5
from anatobind.v41.source_check_v5 import render_source_check_md_v5, summarize_source_check_v5
from tests.v5.factories import make_source_policy

SECRET_CASE = "zz_fake_case_id"
SECRET_PATH = "/private/zz_patient_root/zz_fake_case_id/img.nii.gz"


def _rows():
    geometry = [
        {"source": "synthetic", "case_id": SECRET_CASE, "image_role": "dwi", "image_relpath": SECRET_PATH,
         "label_relpath": SECRET_PATH, "status": "pass", "reasons": []},
        {"source": "synthetic", "case_id": "c2", "image_role": "dwi", "image_relpath": SECRET_PATH,
         "label_relpath": SECRET_PATH, "status": "fail", "reasons": ["AFFINE_MISMATCH"]},
        {"source": "other", "case_id": "c3", "image_role": "dwi", "image_relpath": SECRET_PATH,
         "label_relpath": SECRET_PATH, "status": "fail", "reasons": ["SHAPE_MISMATCH"]},
        {"source": "synthetic", "case_id": "c4", "image_role": "dwi", "image_relpath": SECRET_PATH,
         "label_relpath": SECRET_PATH, "status": None, "reasons": [SECRET_PATH]},
    ]
    pairing = [
        {"source": "synthetic", "case_id": SECRET_CASE, "status": "paired", "images": ["dwi"], "label_relpath": SECRET_PATH},
        {"source": "synthetic", "case_id": "c2", "status": "missing_label", "images": ["dwi"]},
        {"source": "synthetic", "case_id": "c4", "status": "incomplete", "images": [], "label_relpath": None},
        {"source": "other", "case_id": "c3", "status": "paired", "images": ["dwi"], "label_relpath": SECRET_PATH},
    ]
    channel = [{"case_id": SECRET_CASE, "h5_seg_shape": [4, 4, 4, 6],
                "tracks": {"dicom_track_relpath": {"best": {"transform": "transpose_xy", "consistent": True}},
                           "raw_track_relpath": None}},
               {"case_id": "c2", "h5_seg_shape": [4, 4, 4, 6],
                "tracks": {"dicom_track_relpath": {"best": {"transform": "identity", "consistent": False}}}}]
    small = np.zeros((1, 8, 8, 8), dtype=bool)
    small[0, 3, 3, 3] = True
    loss = [downsample_loss_report_v5(small, (2, 2, 2), ["target_a"]),
            downsample_loss_report_v5(np.zeros((1, 8, 8, 8), dtype=bool), (2, 2, 2), ["target_a"])]
    return geometry, pairing, channel, loss


def test_summary_has_counts_only():
    summary = summarize_source_check_v5(*_rows(), make_source_policy())
    text = repr(summary) + render_source_check_md_v5(summary)
    for secret in (SECRET_CASE, SECRET_PATH, "zz_patient_root", "c2", "c3", "c4"):
        assert secret not in text


def test_summary_counts_match_rows():
    summary = summarize_source_check_v5(*_rows(), make_source_policy())
    assert (summary["source_id"], summary["source_version"]) == ("synthetic_source", "v0")
    assert summary["geometry"] == {"fail": 1, "pass": 1, "unknown": 1}
    assert summary["geometry_reasons"] == {"AFFINE_MISMATCH": 1, "OTHER": 1}
    assert summary["channel"] == {"cases_checked": 2, "consistent": 1, "unevaluated": 0}
    assert summary["downsampling"] == {"factors": [[2, 2, 2]], "scans": 2,
                                       "classes": {"target_a": {"components_vanished": 1, "scans_affected": 1}}}
    assert "| paired | 1 |" in render_source_check_md_v5(summary)


def test_missing_pairs_are_counted_not_dropped():
    summary = summarize_source_check_v5(*_rows(), make_source_policy())
    assert summary["pairing"] == {"incomplete": 1, "missing_label": 1, "paired": 1}


def test_unknown_pairing_status_raises():
    geometry, pairing, channel, loss = _rows()
    pairing.append({"source": "synthetic", "case_id": SECRET_CASE, "status": "orphan"})
    with pytest.raises(V5Error) as err:
        summarize_source_check_v5(geometry, pairing, channel, loss, make_source_policy())
    assert (err.value.code, err.value.path) == ("INVALID_PAIRING_STATUS", "$.pairing_rows[4].status")
    assert SECRET_CASE not in str(err.value)
