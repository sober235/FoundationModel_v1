"""Import and registration: official release splits, pairing, scan and patient counts (Proposal_v5 §3.1, §3.3).

Synthetic files only. Nothing here freezes a formal split, trains or unseals a test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anatobind.v41.adapters import isles, skm
from anatobind.v41.audits import (SKM_RELEASE_SPLITS, build_registration_records, load_skm_annotations,
                                  registration_summary, scan_skm_release)
from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.real_t07_v5 import (main, registration_report_v5, render_registration_md_v5,
                                       run_registration_report_v5)

CONFIG = Path("configs/v5")
SKM_A = validate_scope_v5(yaml.safe_load((CONFIG / "scope_skm_a.yaml").read_text()))
JOINT = validate_scope_v5(yaml.safe_load((CONFIG / "scope_skm_joint_early.yaml").read_text()))
ISLES_U = validate_scope_v5(yaml.safe_load((CONFIG / "scope_isles_u.yaml").read_text()))
CLEAN_PROVENANCE = {"ingest_commit", "ingest_time", "inspected_release_flag", "release_split"}


def _doc(images: list, boxes: list = ()) -> dict:
    """images: (image id, case, subject); boxes: (annotation id, image id)."""
    return {"categories": [{"id": 12, "supercategory": "Cartilage Lesion"}],
            "images": [{"id": i, "file_name": f"{case}.h5", "subject_id": subject, "scan_id": case, "timepoint": 0,
                        "inspected": True} for i, case, subject in images],
            "annotations": [{"id": a, "image_id": i, "category_id": 12, "tissue_id": 4, "confidence": 2.0}
                            for a, i in boxes]}


def _release(root: Path, **docs) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for split, doc in docs.items():
        (root / f"{split}.json").write_text(json.dumps(doc))
    return root


def _clean(root: Path) -> Path:
    """Four scans of three patients: the train patient has a follow-up scan."""
    return _release(root, train=_doc([(1, "k1", 7), (2, "k2", 7)], [(1, 1), (2, 1), (3, 2)]),
                    val=_doc([(1, "k3", 8)], [(1, 1)]), test=_doc([(1, "k4", 9)]))


def _assets(cases) -> list:
    out = []
    for case in cases:
        out += [{"source": skm.SOURCE, "kind": "image", "case_id": case, "relpath": f"image_files/{case}.h5"},
                {"source": skm.SOURCE, "kind": "label", "case_id": case,
                 "relpath": f"segmentation_masks/dicom-track/{case}.nii.gz"},
                {"source": skm.SOURCE, "kind": "label", "case_id": case,
                 "relpath": f"segmentation_masks/raw-data-track/{case}.nii.gz"}]
    return out


def test_all_three_official_files_are_checked_and_read(tmp_path):
    per_case, report, _ = scan_skm_release(_clean(tmp_path))
    assert SKM_RELEASE_SPLITS == ("train", "val", "test") and list(report["files"]) == ["train", "val", "test"]
    assert {split: (f["present"], f["readable"], f["scans"], f["patients"], f["boxes"])
            for split, f in report["files"].items()} == {
        "train": (True, True, 2, 1, 3), "val": (True, True, 1, 1, 1), "test": (True, True, 1, 1, 0)}
    assert report["missing_files"] == [] and report["unreadable_files"] == []
    assert {case: meta["split_release"] for case, meta in per_case.items()} == {
        "k1": "train", "k2": "train", "k3": "val", "k4": "test"}
    assert per_case == load_skm_annotations(tmp_path)


def test_scans_and_confirmable_patients_are_counted_separately(tmp_path):
    _, report, details = scan_skm_release(_release(
        tmp_path, train=_doc([(1, "k1", 7), (2, "k2", 7), (3, "k5", None)]), val=_doc([(1, "k3", 8)]),
        test=_doc([(1, "k4", 9)])))
    assert report["scans"] == 5 and report["patients"] == 3              # a follow-up is a scan, not a patient
    assert report["scans_without_subject"] == 1 and details["scans_without_subject"] == ["k5"]
    assert report["files"]["train"]["scans"] == 3 and report["files"]["train"]["patients"] == 1


def test_a_missing_or_unreadable_file_is_recorded_not_skipped(tmp_path):
    root = _release(tmp_path, train=_doc([(1, "k1", 7)]))
    (root / "val.json").write_text("{ not json")
    per_case, report, _ = scan_skm_release(root)
    assert report["missing_files"] == ["test"] and report["unreadable_files"] == ["val"]
    assert report["files"]["test"] == {"present": False, "readable": False, "image_entries": 0, "scans": 0,
                                       "patients": 0, "scans_without_subject": 0, "boxes": 0}
    assert set(per_case) == {"k1"}


def test_a_duplicate_case_inside_one_file_is_counted_and_its_boxes_are_not_doubled(tmp_path):
    per_case, report, details = scan_skm_release(_release(
        tmp_path, train=_doc([(1, "k1", 7), (2, "k1", 7)], [(1, 1), (1, 2)]), val=_doc([]), test=_doc([])))
    assert report["duplicate_cases_within_file"] == 1 and details["duplicate_cases_within_file"] == ["k1"]
    assert report["scans"] == 1 and report["files"]["train"]["image_entries"] == 2
    assert per_case["k1"]["duplicate_entries"] == 1 and len(per_case["k1"]["boxes"]) == 1


def test_a_case_in_two_official_splits_is_a_recorded_conflict_without_a_release_split(tmp_path):
    per_case, report, details = scan_skm_release(_release(
        tmp_path, train=_doc([(1, "k1", 7)], [(1, 1)]), val=_doc([]), test=_doc([(1, "k1", 7)], [(1, 1), (2, 1)])))
    assert report["cases_in_multiple_splits"] == 1 and details["cases_in_multiple_splits"] == ["k1"]
    assert report["patients_in_multiple_splits"] == 1 and report["scans"] == 1 and report["patients"] == 1
    assert per_case["k1"]["split_release"] is None and per_case["k1"]["release_splits"] == ["train", "test"]
    assert len(per_case["k1"]["boxes"]) == 1                            # the first file's boxes, never the sum
    (record,) = build_registration_records("skm_a", _assets(["k1"]), per_case)
    assert record["provenance"]["release_split"] is None
    assert record["provenance"]["release_split_conflict"] == ["train", "test"] and record["split"] == "unassigned"


def test_a_patient_in_two_official_splits_is_reported_and_blocks_the_official_freeze(tmp_path):
    from anatobind.v41.contracts_v5 import validate_sample_v5
    from anatobind.v41.real_t08_v5 import official_split_from_release_v5
    from tests.v5.factories import make_sample_v5

    per_case, report, details = scan_skm_release(_release(
        tmp_path, train=_doc([(1, "k1", 7)]), val=_doc([]), test=_doc([(1, "k2", 7)])))
    assert report["patients_in_multiple_splits"] == 1 and report["cases_in_multiple_splits"] == 0
    assert details["patients_in_multiple_splits"] == ["7"]
    records = build_registration_records("skm_a", _assets(["k1", "k2"]), per_case)
    assert len({r["leakage_group_uid"] for r in records}) == 1          # one patient, one leakage group
    v5 = [validate_sample_v5(make_sample_v5(sample_uid=r["sample_uid"], patient_uid=r["patient_uid"],
                                            leakage_group_uid=r["leakage_group_uid"], provenance=r["provenance"]))
          for r in records]
    with pytest.raises(V5Error) as error:
        official_split_from_release_v5(v5)
    assert error.value.code == "OFFICIAL_SPLIT_CONFLICT"


def test_registered_records_keep_the_release_split_and_stay_unassigned(tmp_path):
    per_case, _, _ = scan_skm_release(_clean(tmp_path))
    records = build_registration_records("skm_a", _assets(["k1", "k2", "k3", "k4"]), per_case)
    assert [r["provenance"]["release_split"] for r in records] == ["train", "train", "val", "test"]
    assert {r["split"] for r in records} == {"unassigned"}              # the formal split comes from the freeze step
    assert all(set(r["provenance"]) == CLEAN_PROVENANCE for r in records)   # a clean record keeps its earlier form
    assert records[0]["leakage_group_uid"] == records[1]["leakage_group_uid"] == records[0]["patient_uid"]
    assert {e["role"] for r in records for e in r["exposure_history"]} == {"unknown"}
    assert all(len(r["image_paths"]) == 1 for r in records)             # echoes and tracks add no record


def test_records_flag_a_missing_release_entry_and_an_unconfirmed_patient(tmp_path):
    per_case, _, _ = scan_skm_release(_release(tmp_path, train=_doc([(1, "k1", None)]), val=_doc([]), test=_doc([])))
    by_uid = {r["sample_uid"]: r for r in build_registration_records("skm_a", _assets(["k1", "k9"]), per_case)}
    assert by_uid["skmtea:k1"]["provenance"]["patient_identity"] == "unconfirmed_case_fallback"
    assert by_uid["skmtea:k9"]["provenance"]["release_entry"] == "missing"
    assert by_uid["skmtea:k9"]["provenance"]["release_split"] is None
    summary = registration_summary(list(by_uid.values()))
    assert summary["scans"] == 2 and summary["patients_confirmed"] == 0 and summary["scans_patient_unconfirmed"] == 2
    assert summary["by_release_split"]["missing"]["scans"] == 1


@pytest.mark.parametrize("scope", [SKM_A, JOINT])
def test_registration_report_counts_scans_patients_splits_and_exposure(tmp_path, scope):
    assets = _assets(["k1", "k2", "k3", "k4"])
    report, details = registration_report_v5(assets, scope, _clean(tmp_path))
    registration = report["registration"]
    assert registration["scans"] == 4 and registration["patients_confirmed"] == 3
    assert registration["leakage_groups"] == 3 and registration["patients_in_multiple_leakage_groups"] == 0
    assert registration["by_release_split"] == {
        "test": {"scans": 1, "patients": 1, "scans_patient_unconfirmed": 0},
        "train": {"scans": 2, "patients": 1, "scans_patient_unconfirmed": 0},
        "val": {"scans": 1, "patients": 1, "scans_patient_unconfirmed": 0}}
    assert registration["formal_split"] == {"unassigned": 4} and registration["exposure_roles"] == {"unknown": 4}
    assert report["pairing"]["cases"] == 4 and report["pairing"]["paired"] == 4
    assert report["pairing"]["with_raw_track"] == 4                     # a second track adds no scan
    assert report["release"]["release_scans_without_assets"] == 0
    assert report["release"]["registered_scans_without_release_entry"] == 0
    markdown = render_registration_md_v5(report)
    for secret in ("k1", "k4", ".h5", "skmtea:"):
        assert secret not in markdown
    assert "unassigned 4" in markdown and "unknown 4" in markdown and "不构成盲测资格" in markdown
    assert "没有冻结正式划分" in markdown and "| test | 是 | 是 | 1 | 1 | 0 | 0 |" in markdown


def test_registration_report_shows_release_scans_without_local_assets(tmp_path):
    report, details = registration_report_v5(_assets(["k1", "k2", "k3", "k9"]), SKM_A, _clean(tmp_path))
    assert report["release"]["release_scans_without_assets"] == 1 and details["release_scans_without_assets"] == ["k4"]
    assert report["release"]["registered_scans_without_release_entry"] == 1
    assert report["registration"]["by_release_split"]["missing"]["scans"] == 1


def test_skm_registration_report_needs_the_release():
    with pytest.raises(V5Error) as error:
        registration_report_v5(_assets(["k1"]), SKM_A)
    assert error.value.code == "RELEASE_REQUIRED"


def _isles_assets() -> list:
    def image(case, name):
        return {"source": isles.SOURCE, "kind": "image", "case_id": case, "relpath": f"{case}/{case}_{name}.nii.gz"}

    def mask(case, tag=""):
        return {"source": isles.SOURCE, "kind": "label", "case_id": case,
                "relpath": f"derivatives/{case}/{case}_msk{tag}.nii.gz"}

    return [image("c1", "dwi"), image("c1", "adc"), image("c1", "flair"), mask("c1"),       # complete
            image("c2", "dwi"), image("c2", "adc"),                                           # no mask
            image("c3", "dwi"), mask("c3"),                                                   # no ADC
            image("c4", "dwi"), image("c4", "adc"), mask("c4"), mask("c4", "_v2")]            # two masks


def test_isles_pairing_counts_complete_missing_and_duplicate_assets():
    report = isles.pairing_report(_isles_assets())
    assert report == {"cases": 4, "paired": 2, "missing_mask": 1, "missing_dwi": 0, "missing_adc": 1,
                      "duplicate_role_assets": 1, "cases_with_duplicate_assets": 1,
                      "other_images_not_registered": 1}


def test_isles_registration_report_has_no_release_and_every_case_is_one_patient(tmp_path):
    report, details = registration_report_v5(_isles_assets(), ISLES_U)
    assert report["release"] is None and details is None
    registration = report["registration"]
    assert registration["scans"] == 4 and registration["patients_confirmed"] == 4
    assert registration["formal_split"] == {"unassigned": 4} and registration["scans_without_label"] == 1
    assert registration["by_release_split"] == {} and "官方划分文件" not in render_registration_md_v5(report)


def test_registration_report_is_written_privately_and_printed_redacted(tmp_path, capsys):
    release = _clean(tmp_path / "release")
    assets = tmp_path / "assets.jsonl"
    assets.write_text("".join(json.dumps(a) + "\n" for a in _assets(["k1", "k2", "k3", "k4"])))
    out = tmp_path / "out"
    assert main(["registration-report", "--assets", str(assets), "--scope", str(CONFIG / "scope_skm_joint_early.yaml"),
                 "--skm-annotations-dir", str(release), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed == (out / "registration.md").read_text() and "k1" not in printed
    assert oct(out.stat().st_mode & 0o777) == "0o700"
    for name in ("registration.md", "registration_report.json", "registration_details.json"):
        assert oct((out / name).stat().st_mode & 0o777) == "0o600"
    assert not (out / "records_prepared.jsonl").exists() and not (out / "split_registry.json").exists()
    with pytest.raises(FileExistsError):
        run_registration_report_v5(_assets(["k1"]), SKM_A, out, release)


def test_loaders_read_by_the_frozen_role_never_by_the_release_split_or_the_record_field():
    from anatobind.v41.engine.registration_v5 import select_samples_v5
    from tests.v5.factories import make_split_records, make_split_registry

    frozen = make_split_records(["train", "development_val", "locked_test", "unassigned"])
    registry = make_split_registry(frozen)
    relabelled = make_split_records(["locked_test", "train", "train", "train"])     # record fields say otherwise
    assert [r.sample_uid for r in select_samples_v5(relabelled, registry, ["train"])] == ["s-0"]
    assert [r.sample_uid for r in select_samples_v5(relabelled, registry, ["development_val"])] == ["s-1"]
    for role, code in (("locked_test", "LOCKED_TEST_NOT_READABLE"), ("unassigned", "INVALID_ROLE"),
                       ("excluded", "INVALID_ROLE")):
        with pytest.raises(V5Error) as error:
            select_samples_v5(relabelled, registry, [role])
        assert error.value.code == code
