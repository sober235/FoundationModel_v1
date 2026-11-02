import copy
import json

import pytest

from anatobind.v41.registry import (
    REQUIRED_AUDITS,
    evaluate_gate,
    publish_manifest,
)
from anatobind.v41.schema import GateDecision, validate_sample


@pytest.fixture
def good_audits():
    return [{"name": name, "scope": "synthetic", "status": "pass", "sha256": "0" * 64}
            for name in REQUIRED_AUDITS]


@pytest.fixture
def box_fill_record(sample_dict):
    record = copy.deepcopy(sample_dict)
    record["annotation_records"][0]["provenance_type"] = "box_fill"
    record["annotation_records"][0]["reference_role"] = "weak_localization"
    record["annotation_records"][0]["completeness"] = "positive_only"
    return record


def test_box_fill_cannot_enter_expert_test(box_fill_record, good_audits, tmp_path):
    box_fill_record["annotation_records"][0]["intended_role"] = "test"
    box_fill_record["annotation_records"][0]["reference_role"] = "expert_voxel"
    with pytest.raises(ValueError, match="UNQUALIFIED_REFERENCE"):
        publish_manifest([validate_sample(box_fill_record)], good_audits, tmp_path)


def test_automated_label_path_does_not_become_expert(sample_dict, good_audits, tmp_path):
    sample_dict["annotation_records"][0]["provenance_type"] = "automated"
    sample_dict["annotation_records"][0]["reference_role"] = "expert_voxel"
    sample_dict["annotation_records"][0]["path"] = "derived/synthseg/labels/case.nii.gz"
    with pytest.raises(ValueError, match="UNQUALIFIED_REFERENCE"):
        publish_manifest([validate_sample(sample_dict)], good_audits, tmp_path)


def test_manual_expert_and_weak_box_fill_are_admitted(box_fill_record, sample_dict, good_audits, tmp_path):
    box_fill_record["sample_uid"] = "synth-box-0001"
    result = publish_manifest([validate_sample(box_fill_record), validate_sample(sample_dict)],
                              good_audits, tmp_path / "v1")
    assert result["admitted"] == 2
    entries = {e["sample_uid"]: e for e in result["entries"]}
    assert entries["synth-0001"]["status"] == "admitted"
    assert entries["synth-0001"]["roles"] == {"ann-synth-0001": "expert_voxel"}
    assert entries["synth-box-0001"]["roles"] == {"ann-synth-0001": "weak_localization"}


def test_duplicate_sample_uid_is_rejected(sample_dict, good_audits, tmp_path):
    with pytest.raises(ValueError, match="DUPLICATE_SAMPLE_UID"):
        publish_manifest([validate_sample(sample_dict), validate_sample(sample_dict)], good_audits, tmp_path)


def test_missing_geometry_audit_blocks_only_that_source(box_fill_record, sample_dict, good_audits, tmp_path):
    other = dict(sample_dict)
    other["sample_uid"] = "synth-0002"
    other["source_dataset"] = "other_source"
    audits = [a for a in good_audits if a["name"] != "geometry_qc"]
    result = publish_manifest([validate_sample(sample_dict), validate_sample(other)], audits, tmp_path)
    entries = {e["sample_uid"]: e for e in result["entries"]}
    assert entries["synth-0001"]["status"] == "blocked"
    assert "MISSING_AUDIT" in entries["synth-0001"]["reasons"]
    assert entries["synth-0002"]["status"] == "blocked"
    assert result["admitted"] == 0
    assert result["sources"]["synthetic"]["status"] == "blocked"


def test_source_with_complete_audits_passes_while_other_blocked(sample_dict, good_audits, tmp_path):
    other = dict(sample_dict)
    other["sample_uid"] = "synth-0002"
    other["source_dataset"] = "other_source"
    result = publish_manifest([validate_sample(sample_dict), validate_sample(other)], good_audits, tmp_path)
    entries = {e["sample_uid"]: e for e in result["entries"]}
    assert entries["synth-0001"]["status"] == "admitted"
    assert entries["synth-0002"]["status"] == "blocked"
    assert result["sources"]["synthetic"]["status"] == "admitted"
    assert result["sources"]["other_source"]["status"] == "blocked"


def test_manifest_is_immutable_and_locked(box_fill_record, good_audits, tmp_path):
    records = [validate_sample(box_fill_record)]
    result = publish_manifest(records, good_audits, tmp_path / "v1")
    lock = json.loads((tmp_path / "v1" / "manifest.lock.json").read_text())
    assert lock["manifest_sha256"] == result["manifest_sha256"]
    assert len(lock["manifest_sha256"]) == 64
    assert lock["entries"] == 1
    with pytest.raises(ValueError, match="PRIVATE_FILE_EXISTS"):
        publish_manifest(records, good_audits, tmp_path / "v1")


def test_unverified_reference_cannot_enter_test_split(sample_dict, good_audits, tmp_path):
    sample_dict["annotation_records"][0]["reference_role"] = "unverified"
    sample_dict["annotation_records"][0]["intended_role"] = "test"
    sample_dict["split"] = "locked_test"
    result = publish_manifest([validate_sample(sample_dict)], good_audits, tmp_path)
    entry = result["entries"][0]
    assert entry["status"] == "blocked"
    assert "UNVERIFIED_REFERENCE_IN_TEST" in entry["reasons"]


def test_gate_requires_clinical_signoff(good_audits):
    reports = {a["name"]: a["status"] for a in good_audits}
    decision = evaluate_gate("G0-U", reports, {})
    assert isinstance(decision, GateDecision)
    assert decision.engineering_status == "BLOCKED"
    assert decision.scientific_conclusion == "NOT_EVALUATED"
    assert any("target_abnormality_definition" in b for b in decision.blockers)


def test_gate_passes_engineering_with_audits_and_signoffs(good_audits):
    reports = {a["name"]: a["status"] for a in good_audits}
    approvals = {"target_abnormality_definition": {"approved_by": "临床负责人", "approved_at": "2026-11-02"}}
    decision = evaluate_gate("G0-U", reports, approvals)
    assert decision.engineering_status == "PASS"
    assert decision.scientific_conclusion == "NOT_EVALUATED"
    assert decision.approved_by == "临床负责人"


def test_gate_fails_on_failed_audit_and_blocks_on_missing(good_audits):
    reports = {a["name"]: a["status"] for a in good_audits}
    reports["geometry_qc"] = "fail"
    decision = evaluate_gate("G0-A", reports, {"target_abnormality_definition": {"approved_by": "x", "approved_at": "t"}})
    assert decision.engineering_status == "FAIL"
    reports["geometry_qc"] = "missing"
    decision = evaluate_gate("G0-A", reports, {"target_abnormality_definition": {"approved_by": "x", "approved_at": "t"}})
    assert decision.engineering_status == "BLOCKED"


def test_joint_gate_blocked_without_same_case_references():
    reports = {name: "pass" for name in REQUIRED_AUDITS}
    approvals = {"target_abnormality_definition": {"approved_by": "x", "approved_at": "t"}}
    decision = evaluate_gate("G0-joint", reports, approvals, joint_evidence={"same_case_expert_A_and_U": False})
    assert decision.engineering_status == "BLOCKED"
    assert any("same-case" in b for b in decision.blockers)


def test_cli_audit_manifest_gate_roundtrip(tmp_path, capsys):
    import json as _json

    from anatobind.v41 import cli
    from anatobind.v41.audits import build_registration_records

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    assets_path = tmp_path / "assets.jsonl"
    base = "ISLES_ltr/ISLES-2022/sub-0001/ses-0001"
    rows = [{"source": "ISLES_ltr", "kind": "label", "case_id": "sub-0001",
             "relpath": "ISLES_ltr/ISLES-2022/derivatives/sub-0001/ses-0001/sub-0001_msk.nii.gz"}]
    for sub, role in (("dwi", "dwi"), ("dwi", "adc"), ("anat", "flair")):
        rows.append({"source": "ISLES_ltr", "kind": "image", "case_id": "sub-0001",
                     "relpath": f"{base}/{sub}/sub-0001_ses-0001_{role}.nii.gz"})
    assets_path.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    audit_out = private / "audit_isles"
    assert cli.main(["audit", "--scope", "isles_u", "--assets", str(assets_path),
                     "--out", str(audit_out), "--private-root", str(private)]) == 0
    summary = _json.loads(capsys.readouterr().out)
    assert summary["records"] == 1
    report_files = {name: audit_out / name for name in
                    ("annotation_provenance_report.csv", "input_visibility_report.csv",
                     "eligibility_coverage.csv")}
    index = [{"name": name, "scope": "ISLES_ltr", "status": "pass", "path": str(p)}
             for name, p in report_files.items()]
    index_path = tmp_path / "index.json"
    index_path.write_text(_json.dumps(index))
    manifest_out = private / "manifest"
    assert cli.main(["build-manifest", "--records", str(audit_out / "registration_records.jsonl"),
                     "--audit-index", str(index_path), "--out", str(manifest_out),
                     "--private-root", str(private)]) == 0
    printed = _json.loads(capsys.readouterr().out)
    assert printed["blocked"] == 1 and printed["admitted"] == 0
    gate_out = private / "gates"
    reports_path = tmp_path / "reports.json"
    reports_path.write_text(_json.dumps({name: "pass" for name in REQUIRED_AUDITS}))
    assert cli.main(["gate", "--gate", "G0-joint", "--reports", str(reports_path),
                     "--out", str(gate_out), "--private-root", str(private)]) == 0
    decision = _json.loads(capsys.readouterr().out)
    assert decision["engineering_status"] == "BLOCKED"
    body = _json.loads((gate_out / "G0-joint.json").read_text())
    assert body["scientific_conclusion"] == "NOT_EVALUATED"
    assert any("same-case" in b for b in body["blockers"])

def test_audit_index_allows_same_name_per_scope_but_not_duplicates(tmp_path):
    from anatobind.v41.registry import ManifestError, assemble_audit_index

    p = tmp_path / "geometry_qc.csv"
    p.write_text("a\n")
    entries = [{"name": "geometry_qc", "scope": "ISLES_ltr", "status": "pass", "path": str(p)},
               {"name": "geometry_qc", "scope": "SKM-TEA_ltr", "status": "pass", "path": str(p)}]
    assembled = assemble_audit_index(entries)
    assert [e["scope"] for e in assembled] == ["ISLES_ltr", "SKM-TEA_ltr"]
    with pytest.raises(ManifestError, match="DUPLICATE_AUDIT_NAME"):
        assemble_audit_index(entries + [entries[0]])


def test_gate_names_failed_audits_in_blockers():
    reports = {name: "pass" for name in REQUIRED_AUDITS}
    reports["geometry_qc"] = "fail"
    decision = evaluate_gate("G0-U", reports, {"target_abnormality_definition": {"approved_by": "x", "approved_at": "t"}})
    assert decision.engineering_status == "FAIL"
    assert any("failed audits: geometry_qc" in b for b in decision.blockers)

