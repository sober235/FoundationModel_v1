"""Real T11 box import driver and the joint G-DATA rule (plan_v5 T11; Proposal_v5 §5.0). Synthetic files only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from anatobind.v41.annotation_v5 import load_skm_box_policy_v5
from anatobind.v41.contracts_v5 import V5Error, validate_sample_v5, validate_scope_v5
from anatobind.v41.gates_v5 import GATE_REQUIREMENTS, evaluate_gate_v5
from anatobind.v41.real_gdata_v5 import gdata_evidence_v5
from anatobind.v41.real_t11_v5 import CONTACT_ROLE, main, records_sha256_v5, run_skm_import_v5
from anatobind.v41.source_policy_v5 import load_source_policies_v5, source_binding_v5
from tests.v5.factories import make_sample_v5, make_split_registry, write_skm_h5
from tests.v5.test_annotation_v5 import CATEGORIES, SHAPE, _box, _reference_a, _release

CONFIG = Path("configs/v5")
SCOPE = validate_scope_v5(yaml.safe_load((CONFIG / "scope_skm_joint_early.yaml").read_text()))
SKM_A = validate_scope_v5(yaml.safe_load((CONFIG / "scope_skm_a.yaml").read_text()))
POLICIES = load_source_policies_v5(CONFIG / "sources_v5.yaml")
SOURCE = POLICIES["skm_tea_joint_early"]
BOX_POLICY = load_skm_box_policy_v5(CONFIG / "skm_box_policy_v1.yaml")
A6 = list(SCOPE.anatomy_classes)
CASES = {"k1": ("unassigned", "train"), "k2": ("unassigned", "val"), "k3": ("unassigned", "test")}
AT = "2026-11-10T00:00:00Z"


def _record(case: str, **overrides):
    split, release_split = CASES[case]
    fields = {"sample_uid": f"skmtea:{case}", "split": split, "patient_uid": f"skmtea:subject:{case}",
              "leakage_group_uid": f"skmtea:subject:{case}", "active_scope_id": SCOPE.scope_id,
              "reference_policy_id": SCOPE.reference_policy_id, "model_modalities": ["qdess_e1", "qdess_e2"],
              "source_dataset": SOURCE.source_dataset, "source_release": SOURCE.source_release,
              "modality_names": ["qdess_e1", "qdess_e2"],
              "image_paths": [f"image_files/{case}.h5#echo1", f"image_files/{case}.h5#echo2"],
              "modality_is_synthetic": [False, False], "available_modality_mask": [True, True],
              "provenance": {"ingest_commit": "0" * 40, "ingest_time": "2026-11-02T00:00:00Z",
                             "release_split": release_split}}
    annotation = {"task": "A", "class_scope": A6, "path": f"image_files/{case}.h5#seg",
                  "label_map_version": "skm_a6_h5_channels_v1", "annotation_modalities": ["qdess_e1", "qdess_e2"]}
    return validate_sample_v5(make_sample_v5(annotation=annotation, **{**fields, **overrides}))


RECORDS = [_record(case) for case in CASES]


def _tree(root: Path, boxes: dict | None = None) -> Path:
    """Three scans, one per official split (train, val, test)."""
    reference = _reference_a()
    seg = np.stack([reference[name] for name in A6], axis=-1)
    echo = np.ones(SHAPE, dtype=np.int16)
    for case in CASES:
        write_skm_h5(root / f"image_files/{case}.h5", echo, echo, seg)
    boxes = boxes if boxes is not None else {
        "k1": [_box(1, "Cartilage Lesion (2A)", "Femoral Cartilage", [0, 0, 0, 3, 3, 3]),
               _box(2, "Cartilage Lesion (1)", "Patellar Cartilage", [0, 5, 0, 3, 3, 3]),
               _box(3, "Effusion", None, [0, 0, 0, 2, 2, 2])],
        "k2": [_box(4, "Meniscal Tear (Radial)", "Meniscus", [9, 0, 0, 3, 3, 3]),
               _box(5, "Meniscal Tear (Radial)", "Meniscus", [4, 5, 0, 2, 2, 2]),       # no host tissue in the box
               _box(6, "Cartilage Lesion (3)", "Meniscus", [0, 0, 0, 3, 3, 3])],       # category/tissue mismatch
        "k3": [_box(7, "Cartilage Lesion (2B)", "Patellar Cartilage", [0, 5, 0, 3, 3, 3])]}
    directory = root / "annotations/v1.0.0"
    directory.mkdir(parents=True)
    for release_split in ("train", "val", "test"):
        doc = _release()
        cases = [case for case, (_, name) in CASES.items() if name == release_split]
        doc["images"] = [{"id": 100 + i, "file_name": f"{case}.h5"} for i, case in enumerate(cases)]
        doc["annotations"] = [{**box, "image_id": 100 + i} for i, case in enumerate(cases)
                              for box in boxes.get(case, [])]
        (directory / f"{release_split}.json").write_text(json.dumps(doc))
    return directory


def _import(raw, annotations, out, **changes):
    args = {"records": RECORDS, "scope": SCOPE, "source_policy": SOURCE, "box_policy": BOX_POLICY, **changes}
    return run_skm_import_v5(raw, args["records"], args["scope"], args["source_policy"], args["box_policy"],
                             annotations, out, recorded_at=AT)


def _run(tmp_path, **changes):
    raw = tmp_path / "raw"
    annotations = _tree(raw, changes.pop("boxes", None))
    out = tmp_path / "private" / "t11"
    out.parent.mkdir(mode=0o700, exist_ok=True)
    return _import(raw, annotations, out, **changes), out


def _rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_import_covers_all_three_official_splits_and_keeps_the_release_split(tmp_path):
    report, out = _run(tmp_path)
    assert report["release_files"] == {"train": True, "val": True, "test": True}
    assert report["by_release_split"] == {"train": 1, "val": 1, "test": 1}
    assert report["records"] == 3 and report["failures"] == [] and report["summary"]["scans"] == 3
    rows = _rows(out / "box_references.jsonl")
    assert [(row["sample_uid"], row["release_split"]) for row in rows] == [
        ("skmtea:k1", "train"), ("skmtea:k2", "val"), ("skmtea:k3", "test")]
    assert report["scope_hash"] == SCOPE.scope_hash and report["records_sha256"] == records_sha256_v5(RECORDS)
    assert report["source"] == source_binding_v5(SOURCE) and report["policy_sha256"] == BOX_POLICY["policy_sha256"]


def test_import_assigns_no_formal_split(tmp_path):
    report, out = _run(tmp_path)
    assert report["formal_split"] == {"unassigned": 3}                 # the release split is not the formal split
    assert {row["split"] for row in _rows(out / "box_references.jsonl")} == {"unassigned"}
    assert "不是正式划分" in (out / "box_import.md").read_text()


def test_import_contact_is_recorded_apart_from_training_exposure(tmp_path):
    from anatobind.v41.exposure_v5 import lineage_blockers_v5

    report, out = _run(tmp_path)
    ledger = _rows(out / "contact_ledger.jsonl")
    assert [entry["sample_uid"] for entry in ledger] == ["skmtea:k1", "skmtea:k2", "skmtea:k3"]
    assert {entry["role"] for entry in ledger} == {CONTACT_ROLE} == {"administrative_qc"}
    assert {entry["recorded_at"] for entry in ledger} == {AT} and report["contact_role"] == "administrative_qc"
    entries = [{k: v for k, v in entry.items() if k != "sample_uid"} for entry in ledger]
    assert lineage_blockers_v5(entries, "any_lineage", []) == []       # contact alone never blocks a locked test
    assert lineage_blockers_v5(entries + [{"role": "unknown", "lineage_id": None}], "any_lineage", []) == ["unknown"]
    assert "administrative_qc" in (out / "box_import.md").read_text()


def test_import_keeps_boxes_box_level_and_never_records_a_negative(tmp_path):
    report, out = _run(tmp_path)
    rows = {row["sample_uid"]: row for row in _rows(out / "box_references.jsonl")}
    statuses = {box["annotation_id"]: (box["status"], box["reason"]) for row in rows.values() for box in row["boxes"]}
    assert statuses == {1: ("included", None), 2: ("included", None), 3: ("out_of_scope", "out_of_scope_category"),
                        4: ("included", None), 5: ("excluded", "no_host_tissue_in_box"),
                        6: ("excluded", "category_tissue_mismatch"), 7: ("included", None)}
    assert all(box["reference_granularity"] == "box" for row in rows.values() for box in row["boxes"])
    states = [state for row in rows.values() for groups in row["tissue_state"]["states"].values()
              for state in groups.values()]
    assert set(states) == {"positive", "annotation_unknown"}
    summary = report["summary"]
    assert summary["cascade"]["primary"] == 3 and summary["cascade"]["secondary"] == 1
    assert summary["excluded"] == {"no_host_tissue_in_box": 1, "category_tissue_mismatch": 1}
    assert summary["positive_scans_by_split"] == {"train": {"cartilage_lesion:femoral_cartilage": 1},
                                                  "val": {"meniscal_tear:meniscus": 1},
                                                  "test": {"cartilage_lesion:patellar_cartilage": 1}}


def test_outputs_are_private_never_overwritten_and_the_markdown_is_redacted(tmp_path):
    report, out = _run(tmp_path)
    assert oct(out.stat().st_mode & 0o777) == "0o700"
    for name in ("box_references.jsonl", "contact_ledger.jsonl", "box_import.md", "import_report.json"):
        assert oct((out / name).stat().st_mode & 0o777) == "0o600"
    markdown = (out / "box_import.md").read_text()
    assert hashlib.sha256(markdown.encode()).hexdigest() == report["box_import_sha256"]
    for secret in ("skmtea:", "k1", "k2", "k3", ".h5", ".json", str(tmp_path)):
        assert secret not in markdown
    assert "NOT_EVALUATED" in markdown and "annotation_unknown" in markdown and "不作阴性" in markdown
    assert "train 1、val 1、test 1" in markdown and SCOPE.scope_hash in markdown
    with pytest.raises(FileExistsError):
        _import(tmp_path / "raw", tmp_path / "raw/annotations/v1.0.0", out)


def test_a_missing_official_file_is_recorded_and_its_scans_fail(tmp_path):
    raw = tmp_path / "raw"
    annotations = _tree(raw)
    (annotations / "test.json").unlink()
    report = _import(raw, annotations, tmp_path / "out")
    assert report["release_files"] == {"train": True, "val": True, "test": False}
    assert report["failures"] == [{"index": 2, "code": "RELEASE_FILE_MISSING", "path": "$.provenance.release_split"}]
    assert report["summary"]["scans"] == 2 and "test 缺失" in (tmp_path / "out/box_import.md").read_text()


@pytest.mark.parametrize("provenance,bucket", [
    ({"release_split": None, "release_split_conflict": ["train", "test"]}, "conflict"),
    ({"release_split": None, "release_entry": "missing"}, "missing")])
def test_a_scan_with_an_unresolved_official_split_is_recorded_not_imported(tmp_path, provenance, bucket):
    raw = tmp_path / "raw"
    annotations = _tree(raw)
    base = {"ingest_commit": "0" * 40, "ingest_time": "2026-11-02T00:00:00Z"}
    records = [RECORDS[0], RECORDS[1], _record("k3", provenance={**base, **provenance})]
    report = _import(raw, annotations, tmp_path / "out", records=records)
    assert report["by_release_split"] == {"train": 1, "val": 1, "test": 0, bucket: 1}
    assert report["failures"] == [{"index": 2, "code": "RELEASE_SPLIT_UNRESOLVED",
                                   "path": "$.provenance.release_split"}]
    assert [row["sample_uid"] for row in _rows(tmp_path / "out/contact_ledger.jsonl")] == ["skmtea:k1", "skmtea:k2"]


def test_a_scan_that_cannot_be_imported_is_a_counted_failure_without_identifiers(tmp_path):
    raw = tmp_path / "raw"
    annotations = _tree(raw)
    (raw / "image_files/k2.h5").unlink()
    report = _import(raw, annotations, tmp_path / "out")
    assert report["failures"] == [{"index": 1, "code": "VOLUME_MISSING", "path": "$.ref"}]
    assert report["summary"]["scans"] == 2
    markdown = (tmp_path / "out/box_import.md").read_text()
    assert "VOLUME_MISSING" in markdown and "k2" not in markdown


def test_a_scan_missing_from_its_release_file_fails_instead_of_becoming_boxless(tmp_path):
    raw = tmp_path / "raw"
    annotations = _tree(raw)
    doc = json.loads((annotations / "val.json").read_text())
    doc["images"] = []
    (annotations / "val.json").write_text(json.dumps(doc))
    report = _import(raw, annotations, tmp_path / "out")
    assert report["failures"] == [{"index": 1, "code": "RELEASE_ENTRY_MISSING", "path": "$.sample_uid"}]


def test_an_unknown_release_category_is_refused_not_guessed(tmp_path):
    raw = tmp_path / "raw"
    annotations = _tree(raw)
    doc = json.loads((annotations / "train.json").read_text())
    doc["categories"].append({"id": 99, "name": "Bone Marrow Edema", "supercategory": "Cartilage Lesion"})
    (annotations / "train.json").write_text(json.dumps(doc))
    report = _import(raw, annotations, tmp_path / "out")
    assert [failure["code"] for failure in report["failures"]] == ["SKM_CATEGORY_UNKNOWN"]


@pytest.mark.parametrize("changes,code", [
    ({"scope": SKM_A}, "SCOPE_TASK_MISMATCH"),                                    # a module scope has no boxes
    ({"source_policy": POLICIES["skm_tea"]}, "SCOPE_POLICY_MISMATCH"),            # skm_a's binding is not this scope's
    ({"box_policy": {**BOX_POLICY, "policy_id": "another_policy"}}, "SCOPE_POLICY_MISMATCH"),
    ({"records": [*RECORDS, RECORDS[0]]}, "DUPLICATE_SAMPLE"),
    ({"records": [_record("k1", active_scope_id="skm_a"), *RECORDS[1:]]}, "RECORD_SCOPE_MISMATCH"),
    ({"records": []}, "INVALID_TYPE"),
])
def test_inputs_from_another_chain_are_refused_before_any_file_is_read(tmp_path, changes, code):
    with pytest.raises(V5Error) as error:                         # raw and annotations do not exist: nothing is read
        _import(tmp_path / "absent", tmp_path / "absent", tmp_path / "out", **changes)
    assert error.value.code == code and not (tmp_path / "out").exists()


def test_command_line_prints_the_redacted_markdown_only(tmp_path, capsys):
    import dataclasses

    raw = tmp_path / "raw"
    annotations = _tree(raw)
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in RECORDS))
    assert main(["import-skm", "--raw", str(raw), "--records", str(records),
                 "--scope", str(CONFIG / "scope_skm_joint_early.yaml"), "--policies", str(CONFIG / "sources_v5.yaml"),
                 "--source-id", "skm_tea_joint_early", "--box-policy", str(CONFIG / "skm_box_policy_v1.yaml"),
                 "--annotations-dir", str(annotations), "--recorded-at", AT, "--out", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out
    assert printed == (tmp_path / "out/box_import.md").read_text() and "skmtea:" not in printed


# ---------------------------------------------------------------------------
# Joint G-DATA rule

def _frozen(records):
    """The formal split, frozen after import by the official rule (test -> locked_test when not exposure-blocked)."""
    official = {r.leakage_group_uid: r.provenance["release_split"] for r in records}
    return make_split_registry(records, lineage_id="v51_skm_joint_early", official_split=official)


def _gdata_inputs(tmp_path):
    import dataclasses

    report, _ = _run(tmp_path)
    registry = _frozen(RECORDS)
    policy = dataclasses.replace(SOURCE, index_evidence=f"joint source check (sha256 {'a' * 64})")
    prepare = {"records": 3, "failures": [], "geometry_status": {"pass": 3}, "source_check_sha256": "a" * 64,
               "summary": {"pairing": {"incomplete": 0, "missing_label": 0, "paired": 3},
                           "geometry": {"fail": 0, "pass": 3, "unknown": 0},
                           "channel": {"cases_checked": 3, "consistent": 3, "unevaluated": 0}}}
    return {"scope": SCOPE, "policy": policy, "prepare": prepare,
            "label_qc": {"SKM-TEA_ltr|pass|-|unexpected=False": 3},
            "migrate": {"records": 3, "failures": [], "index_alignment": {"verified": 3},
                        "physical_geometry": {"unknown": 3}},
            "split": {"groups": 3, "samples": 3, "split_sha256": registry["split_sha256"]},
            "records": RECORDS, "registry": registry, "box_import": report}


def test_the_formal_split_comes_from_the_freeze_step_not_from_import(tmp_path):
    registry = _frozen(RECORDS)
    assert registry["assignment"] == {"skmtea:k1": "train", "skmtea:k2": "development_val",
                                      "skmtea:k3": "locked_test"}
    exposed = [*RECORDS[:2], _record("k3", exposure_history=[
        {"experiment_id": "legacy", "role": "unknown", "recorded_at": AT, "lineage_id": None, "evidence": "legacy"}])]
    assert _frozen(exposed)["assignment"]["skmtea:k3"] == "excluded"   # the name "test" is not blind-test status


def test_joint_gdata_evidence_passes_with_the_box_import_of_the_same_chain(tmp_path):
    inputs = _gdata_inputs(tmp_path)
    evidence, markdown = gdata_evidence_v5(**inputs)
    assert set(evidence) == set(GATE_REQUIREMENTS["G-DATA"]["evidence"])
    assert all(item["status"] == "PASS" and item["task"] == "joint" for item in evidence.values())
    assert inputs["box_import"]["box_import_sha256"] in markdown and BOX_POLICY["policy_sha256"] in markdown
    assert "train 1、val 1、test 1" in markdown and "不是正式划分" in markdown and "skmtea:" not in markdown
    source = source_binding_v5(inputs["policy"])
    blocked = evaluate_gate_v5("G-DATA", SCOPE, evidence, {}, source)
    assert blocked["blockers"] == ["APPROVAL_MISSING:data_use"]                 # the researcher signs data_use
    skm_a_approval = {"data_use": {"approved_by": "researcher", "approved_at": "2026-11-09",
                                   "scope_hash": SKM_A.scope_hash, **source_binding_v5(POLICIES["skm_tea"])}}
    assert evaluate_gate_v5("G-DATA", SCOPE, evidence, skm_a_approval, source)["engineering_status"] == "BLOCKED"
    joint = evaluate_gate_v5("G-JOINT", SCOPE, {}, {}, source)
    assert "CORE_RELATION_REQUIRED" in joint["blockers"]                        # G-DATA does not open G-JOINT


def test_joint_gdata_needs_the_box_import(tmp_path):
    inputs = _gdata_inputs(tmp_path)
    with pytest.raises(V5Error) as error:
        gdata_evidence_v5(**{**inputs, "box_import": None})
    assert error.value.code == "BOX_IMPORT_MISSING"


@pytest.mark.parametrize("field,value", [
    ("scope_hash", SKM_A.scope_hash), ("source", source_binding_v5(POLICIES["skm_tea"])),
    ("policy_id", "another_policy"), ("records", 2), ("records_sha256", "0" * 64), ("release_files", None),
    ("box_import_sha256", None)])
def test_a_box_import_of_another_chain_is_refused(tmp_path, field, value):
    inputs = _gdata_inputs(tmp_path)
    with pytest.raises(V5Error) as error:
        gdata_evidence_v5(**{**inputs, "box_import": {**inputs["box_import"], field: value}})
    assert error.value.code == "BOX_IMPORT_MISMATCH" and error.value.path == f"$.box_import.{field}"


def test_import_failures_fail_the_joint_evidence(tmp_path):
    inputs = _gdata_inputs(tmp_path)
    report = copy.deepcopy(inputs["box_import"])
    report["failures"] = [{"index": 0, "code": "VOLUME_MISSING", "path": "$.ref"}]
    report["summary"]["scans"] = 2
    evidence, _ = gdata_evidence_v5(**{**inputs, "box_import": report})
    assert evidence["semantics"]["status"] == "FAIL" and evidence["pairing"]["status"] == "FAIL"
    assert evidence["readability"]["status"] == "PASS"


@pytest.mark.parametrize("changes", [{"release_files": {"train": True, "val": True, "test": False}},
                                     {"by_release_split": {"train": 1, "val": 1, "test": 0, "conflict": 1}}])
def test_a_missing_official_file_or_an_unresolved_split_fails_provenance(tmp_path, changes):
    inputs = _gdata_inputs(tmp_path)
    evidence, _ = gdata_evidence_v5(**{**inputs, "box_import": {**inputs["box_import"], **changes}})
    assert evidence["provenance"]["status"] == "FAIL"


def test_joint_policy_must_register_both_a_and_u(tmp_path):
    import dataclasses

    inputs = _gdata_inputs(tmp_path)
    policy = dataclasses.replace(inputs["policy"], annotation_modalities={"A": ("qdess_e1", "qdess_e2")})
    with pytest.raises(V5Error) as error:
        gdata_evidence_v5(**{**inputs, "policy": policy})
    assert (error.value.code, error.value.path) == ("SCOPE_POLICY_MISMATCH", "$.policy.annotation_modalities.U")


def test_a_module_scope_refuses_a_box_import(tmp_path):
    from tests.v5 import test_real_gdata as module

    report, _ = _run(tmp_path)
    with pytest.raises(V5Error) as error:
        module._evidence(box_import=report)
    assert error.value.code == "BOX_IMPORT_NOT_EXPECTED"


def test_prepare_accepts_the_joint_scope_and_carries_no_box_annotation():
    from anatobind.v41.audits import build_registration_records
    from anatobind.v41.real_t07_v5 import prepare_record_v5
    from tests.v5.test_real_t07 import GEOMETRY, SKM_ASSETS, SKM_META, _channel, _rows

    record = build_registration_records("skm_a", SKM_ASSETS, SKM_META)[0]
    assert [a["task"] for a in record["annotation_records"]] == ["A", "U"]      # the v41 record carries the box
    rows = _rows("SKM-TEA_ltr", record["image_paths"])
    prepared = prepare_record_v5(record, SCOPE, rows, GEOMETRY, _channel())
    (annotation,) = prepared.annotation_records
    assert (annotation.task, annotation.label_map_version) == ("A", "skm_a6_h5_channels_v1")
    assert prepared.geometry_status == "pass"
    assert CATEGORIES                                                            # shared synthetic release names


def test_the_joint_chain_runs_end_to_end_from_prepare_to_gdata_evidence(tmp_path):
    """prepare -> register the source check -> migrate -> import -> freeze -> G-DATA evidence, as the real run will."""
    import dataclasses

    from anatobind.v41.real_t07_v5 import run_migrate_v5, run_prepare_v5
    from anatobind.v41.real_t08_v5 import run_freeze_v5
    from tests.v5.test_real_t07 import SKM_ASSETS, SKM_META, _skm_tree

    raw, private = tmp_path / "raw", tmp_path / "private"
    _skm_tree(raw)                                                    # one scan, release split val, grid (6, 4, 5)
    private.mkdir(mode=0o700)
    prepare = run_prepare_v5(raw, SKM_ASSETS, SCOPE, SOURCE, private / "t07", factor=(2, 2, 1),
                             skm_annotations=copy.deepcopy(SKM_META), skm_units="mm")
    assert prepare["records"] == 1 and prepare["failures"] == [] and prepare["geometry_status"] == {"pass": 1}
    policy = dataclasses.replace(SOURCE, index_evidence=f"joint source check (sha256 {prepare['source_check_sha256']})")
    migrate = run_migrate_v5(private / "t07/records_prepared.jsonl", policy, SCOPE, private / "records_v5.jsonl")
    assert migrate["records"] == 1 and migrate["index_alignment"] == {"verified": 1}
    records = [validate_sample_v5(json.loads(line))
               for line in (private / "records_v5.jsonl").read_text().splitlines()]
    assert [a.task for a in records[0].annotation_records] == ["A"] and records[0].active_scope_id == SCOPE.scope_id
    doc = _release()
    doc["images"] = [{"id": 1, "file_name": "k1.h5"}]
    meniscus = next(t["id"] for t in doc["tissues"] if t["name"] == "Meniscus")
    doc["annotations"] = [{"id": 1, "image_id": 1, "category_id": CATEGORIES.index("Meniscal Tear (Radial)") + 1,
                           "tissue_id": meniscus, "bbox": [4.0, 0.0, 4.0, 1.0, 4.0, 1.0], "confidence": 3,
                           "labeler": 1}]
    (raw / "annotations").mkdir()
    (raw / "annotations/val.json").write_text(json.dumps(doc))
    for name in ("train", "test"):                                    # the other official files exist, without scans
        (raw / f"annotations/{name}.json").write_text(json.dumps({**_release(), "images": [], "annotations": []}))
    assert records[0].split == "unassigned" and records[0].provenance["release_split"] == "val"
    report = run_skm_import_v5(raw, records, SCOPE, policy, BOX_POLICY, raw / "annotations", private / "t11",
                               recorded_at=AT)
    assert report["failures"] == [] and report["summary"]["cascade"]["primary"] == 1
    assert report["formal_split"] == {"unassigned": 1} and report["by_release_split"] == {"train": 0, "val": 1, "test": 0}
    split = run_freeze_v5(records, yaml.safe_load((CONFIG / "split_rule_skm_a.yaml").read_text()),
                          "v51_skm_joint_early", private / "t08")
    registry = json.loads((private / "t08/split_registry.json").read_text())
    assert list(registry["assignment"].values()) == ["development_val"]
    evidence, markdown = gdata_evidence_v5(
        scope=SCOPE, policy=policy, prepare=prepare, label_qc={"SKM-TEA_ltr|pass|-|unexpected=False": 1},
        migrate=migrate, split=split, records=records, registry=registry, box_import=report)
    assert {item["status"] for item in evidence.values()} == {"PASS"}
    assert "k1" not in markdown and report["box_import_sha256"] in markdown
