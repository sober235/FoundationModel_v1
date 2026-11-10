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
from anatobind.v41.real_t11_v5 import READABLE_SPLITS, main, run_skm_import_v5
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
CASES = {"k1": ("train", "train"), "k2": ("development_val", "val"), "k3": ("locked_test", "test")}


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
REGISTRY = make_split_registry(RECORDS, lineage_id="v51_skm_joint_early")


def _tree(root: Path, boxes: dict | None = None, test_file: str = "{ this file must never be opened") -> Path:
    """Three scans (train, val, test). The test release file is not JSON: opening it would fail the run."""
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
               _box(6, "Cartilage Lesion (3)", "Meniscus", [0, 0, 0, 3, 3, 3])]}       # category/tissue mismatch
    directory = root / "annotations/v1.0.0"
    directory.mkdir(parents=True)
    for release_split in ("train", "val"):
        doc = _release()
        cases = [case for case, (_, name) in CASES.items() if name == release_split]
        doc["images"] = [{"id": 100 + i, "file_name": f"{case}.h5"} for i, case in enumerate(cases)]
        doc["annotations"] = [{**box, "image_id": 100 + i} for i, case in enumerate(cases)
                              for box in boxes.get(case, [])]
        (directory / f"{release_split}.json").write_text(json.dumps(doc))
    (directory / "test.json").write_text(test_file)
    return directory


def _run(tmp_path, **changes):
    raw = tmp_path / "raw"
    annotations = _tree(raw, changes.pop("boxes", None))
    args = {"records": RECORDS, "registry": REGISTRY, "scope": SCOPE, "source_policy": SOURCE,
            "box_policy": BOX_POLICY}
    args.update(changes)
    out = tmp_path / "private" / "t11"
    out.parent.mkdir(mode=0o700, exist_ok=True)
    return run_skm_import_v5(raw, args["records"], args["registry"], args["scope"], args["source_policy"],
                             args["box_policy"], annotations, out), out


def test_import_reads_only_train_and_development_val_and_never_opens_the_test_release(tmp_path):
    report, out = _run(tmp_path)
    assert READABLE_SPLITS == ("train", "development_val")
    assert report["selected"] == {"train": 1, "development_val": 1} and report["not_read"] == {"locked_test": 1}
    assert report["failures"] == [] and report["summary"]["scans"] == 2 and report["summary"]["boxes"] == 6
    rows = [json.loads(line) for line in (out / "box_references.jsonl").read_text().splitlines()]
    assert [(row["sample_uid"], row["split"]) for row in rows] == [("skmtea:k1", "train"),
                                                                    ("skmtea:k2", "development_val")]
    assert report["scope_hash"] == SCOPE.scope_hash and report["split_sha256"] == REGISTRY["split_sha256"]
    assert report["source"] == source_binding_v5(SOURCE) and report["policy_sha256"] == BOX_POLICY["policy_sha256"]


def test_import_keeps_boxes_box_level_and_never_records_a_negative(tmp_path):
    report, out = _run(tmp_path)
    rows = {row["sample_uid"]: row for row in map(json.loads, (out / "box_references.jsonl").read_text().splitlines())}
    statuses = {box["annotation_id"]: (box["status"], box["reason"]) for row in rows.values() for box in row["boxes"]}
    assert statuses == {1: ("included", None), 2: ("included", None), 3: ("out_of_scope", "out_of_scope_category"),
                        4: ("included", None), 5: ("excluded", "no_host_tissue_in_box"),
                        6: ("excluded", "category_tissue_mismatch")}
    assert all(box["reference_granularity"] == "box" for row in rows.values() for box in row["boxes"])
    states = [state for row in rows.values() for groups in row["tissue_state"]["states"].values()
              for state in groups.values()]
    assert set(states) == {"positive", "annotation_unknown"}
    summary = report["summary"]
    assert summary["cascade"]["primary"] == 2 and summary["cascade"]["secondary"] == 1
    assert summary["excluded"] == {"no_host_tissue_in_box": 1, "category_tissue_mismatch": 1}
    assert summary["positive_scans_by_split"] == {"train": {"cartilage_lesion:femoral_cartilage": 1},
                                                  "development_val": {"meniscal_tear:meniscus": 1}}


def test_outputs_are_private_never_overwritten_and_the_markdown_is_redacted(tmp_path):
    report, out = _run(tmp_path)
    assert oct(out.stat().st_mode & 0o777) == "0o700"
    for name in ("box_references.jsonl", "box_import.md", "import_report.json"):
        assert oct((out / name).stat().st_mode & 0o777) == "0o600"
    markdown = (out / "box_import.md").read_text()
    assert hashlib.sha256(markdown.encode()).hexdigest() == report["box_import_sha256"]
    for secret in ("skmtea:", "k1", "k2", "k3", ".h5", ".json", str(tmp_path)):
        assert secret not in markdown
    assert "NOT_EVALUATED" in markdown and "annotation_unknown" in markdown and "不作阴性" in markdown
    assert "train 1" in markdown and "locked_test 1" in markdown and SCOPE.scope_hash in markdown
    with pytest.raises(FileExistsError):
        run_skm_import_v5(tmp_path / "raw", RECORDS, REGISTRY, SCOPE, SOURCE, BOX_POLICY,
                          tmp_path / "raw/annotations/v1.0.0", out)


def test_a_scan_that_cannot_be_imported_is_a_counted_failure_without_identifiers(tmp_path):
    raw = tmp_path / "raw"
    annotations = _tree(raw)
    (raw / "image_files/k2.h5").unlink()
    report = run_skm_import_v5(raw, RECORDS, REGISTRY, SCOPE, SOURCE, BOX_POLICY, annotations, tmp_path / "out")
    assert report["failures"] == [{"index": 1, "code": "VOLUME_MISSING", "path": "$.ref"}]
    assert report["summary"]["scans"] == 1 and report["selected"]["development_val"] == 1
    markdown = (tmp_path / "out/box_import.md").read_text()
    assert "VOLUME_MISSING" in markdown and "k2" not in markdown


def test_a_scan_missing_from_its_release_file_fails_instead_of_becoming_boxless(tmp_path):
    raw = tmp_path / "raw"
    annotations = _tree(raw)
    doc = json.loads((annotations / "val.json").read_text())
    doc["images"] = []
    (annotations / "val.json").write_text(json.dumps(doc))
    report = run_skm_import_v5(raw, RECORDS, REGISTRY, SCOPE, SOURCE, BOX_POLICY, annotations, tmp_path / "out")
    assert report["failures"] == [{"index": 1, "code": "RELEASE_ENTRY_MISSING", "path": "$.sample_uid"}]


def test_an_unknown_release_category_is_refused_not_guessed(tmp_path):
    raw = tmp_path / "raw"
    annotations = _tree(raw)
    doc = json.loads((annotations / "train.json").read_text())
    doc["categories"].append({"id": 99, "name": "Bone Marrow Edema", "supercategory": "Cartilage Lesion"})
    (annotations / "train.json").write_text(json.dumps(doc))
    report = run_skm_import_v5(raw, RECORDS, REGISTRY, SCOPE, SOURCE, BOX_POLICY, annotations, tmp_path / "out")
    assert [failure["code"] for failure in report["failures"]] == ["SKM_CATEGORY_UNKNOWN"]


@pytest.mark.parametrize("changes,code", [
    ({"scope": SKM_A}, "SCOPE_TASK_MISMATCH"),                                    # a module scope has no boxes
    ({"source_policy": POLICIES["skm_tea"]}, "SCOPE_POLICY_MISMATCH"),            # skm_a's binding is not this scope's
    ({"box_policy": {**BOX_POLICY, "policy_id": "another_policy"}}, "SCOPE_POLICY_MISMATCH"),
    ({"records": RECORDS[:2]}, "SPLIT_RECORDS_MISMATCH"),                         # registry of another record set
    ({"records": [_record("k1", active_scope_id="skm_a"), *RECORDS[1:]]}, "RECORD_SCOPE_MISMATCH"),
    ({"registry": {**REGISTRY, "split_sha256": "0" * 64}}, "SPLIT_HASH_MISMATCH"),
])
def test_inputs_from_another_chain_are_refused_before_any_file_is_read(tmp_path, changes, code):
    args = {"records": RECORDS, "registry": REGISTRY, "scope": SCOPE, "source_policy": SOURCE,
            "box_policy": BOX_POLICY, **changes}
    with pytest.raises(V5Error) as error:                         # raw and annotations do not exist: nothing is read
        run_skm_import_v5(tmp_path / "absent", args["records"], args["registry"], args["scope"],
                          args["source_policy"], args["box_policy"], tmp_path / "absent", tmp_path / "out")
    assert error.value.code == code and not (tmp_path / "out").exists()


@pytest.mark.parametrize("splits", [("train", "locked_test"), ("excluded",), ()])
def test_sealed_splits_cannot_be_requested(tmp_path, splits):
    with pytest.raises(V5Error) as error:
        run_skm_import_v5(tmp_path / "absent", RECORDS, REGISTRY, SCOPE, SOURCE, BOX_POLICY, tmp_path / "absent",
                          tmp_path / "out", splits=splits)
    assert error.value.code == "SPLIT_SEALED"


def test_command_line_prints_the_redacted_markdown_only(tmp_path, capsys):
    raw = tmp_path / "raw"
    annotations = _tree(raw)
    records = tmp_path / "records.jsonl"
    import dataclasses
    records.write_text("".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in RECORDS))
    registry = tmp_path / "split_registry.json"
    registry.write_text(json.dumps(REGISTRY))
    assert main(["import-skm", "--raw", str(raw), "--records", str(records), "--split-registry", str(registry),
                 "--scope", str(CONFIG / "scope_skm_joint_early.yaml"), "--policies", str(CONFIG / "sources_v5.yaml"),
                 "--source-id", "skm_tea_joint_early", "--box-policy", str(CONFIG / "skm_box_policy_v1.yaml"),
                 "--annotations-dir", str(annotations), "--out", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out
    assert printed == (tmp_path / "out/box_import.md").read_text() and "skmtea:" not in printed


# ---------------------------------------------------------------------------
# Joint G-DATA rule

def _gdata_inputs(tmp_path):
    import dataclasses

    report, _ = _run(tmp_path)
    policy = dataclasses.replace(SOURCE, index_evidence=f"joint source check (sha256 {'a' * 64})")
    prepare = {"records": 3, "failures": [], "geometry_status": {"pass": 3}, "source_check_sha256": "a" * 64,
               "summary": {"pairing": {"incomplete": 0, "missing_label": 0, "paired": 3},
                           "geometry": {"fail": 0, "pass": 3, "unknown": 0},
                           "channel": {"cases_checked": 3, "consistent": 3, "unevaluated": 0}}}
    return {"scope": SCOPE, "policy": policy, "prepare": prepare,
            "label_qc": {"SKM-TEA_ltr|pass|-|unexpected=False": 3},
            "migrate": {"records": 3, "failures": [], "index_alignment": {"verified": 3},
                        "physical_geometry": {"unknown": 3}},
            "split": {"groups": 3, "samples": 3, "split_sha256": REGISTRY["split_sha256"]},
            "records": RECORDS, "registry": REGISTRY, "box_import": report}


def test_joint_gdata_evidence_passes_with_the_box_import_of_the_same_chain(tmp_path):
    inputs = _gdata_inputs(tmp_path)
    evidence, markdown = gdata_evidence_v5(**inputs)
    assert set(evidence) == set(GATE_REQUIREMENTS["G-DATA"]["evidence"])
    assert all(item["status"] == "PASS" and item["task"] == "joint" for item in evidence.values())
    assert inputs["box_import"]["box_import_sha256"] in markdown and BOX_POLICY["policy_sha256"] in markdown
    assert "locked_test 1" in markdown and "不在本证据内" in markdown and "skmtea:" not in markdown
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
    ("policy_id", "another_policy"), ("split_sha256", "0" * 64), ("selected", {"train": 1, "development_val": 0}),
    ("splits", []), ("box_import_sha256", None)])
def test_a_box_import_of_another_chain_is_refused(tmp_path, field, value):
    inputs = _gdata_inputs(tmp_path)
    with pytest.raises(V5Error) as error:
        gdata_evidence_v5(**{**inputs, "box_import": {**inputs["box_import"], field: value}})
    assert error.value.code == "BOX_IMPORT_MISMATCH" and error.value.path == f"$.box_import.{field}"


def test_import_failures_fail_the_joint_evidence(tmp_path):
    inputs = _gdata_inputs(tmp_path)
    report = copy.deepcopy(inputs["box_import"])
    report["failures"] = [{"index": 0, "code": "VOLUME_MISSING", "path": "$.ref"}]
    report["summary"]["scans"] = 1
    evidence, _ = gdata_evidence_v5(**{**inputs, "box_import": report})
    assert evidence["semantics"]["status"] == "FAIL" and evidence["pairing"]["status"] == "FAIL"
    assert evidence["readability"]["status"] == "PASS"


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
    """prepare -> register the source check -> migrate -> freeze -> import -> G-DATA evidence, as the real run will."""
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
    split = run_freeze_v5(records, yaml.safe_load((CONFIG / "split_rule_skm_a.yaml").read_text()),
                          "v51_skm_joint_early", private / "t08")
    registry = json.loads((private / "t08/split_registry.json").read_text())
    assert list(registry["assignment"].values()) == ["development_val"]
    doc = _release()
    doc["images"] = [{"id": 1, "file_name": "k1.h5"}]
    meniscus = next(t["id"] for t in doc["tissues"] if t["name"] == "Meniscus")
    doc["annotations"] = [{"id": 1, "image_id": 1, "category_id": CATEGORIES.index("Meniscal Tear (Radial)") + 1,
                           "tissue_id": meniscus, "bbox": [4.0, 0.0, 4.0, 1.0, 4.0, 1.0], "confidence": 3,
                           "labeler": 1}]
    (raw / "annotations").mkdir()
    (raw / "annotations/val.json").write_text(json.dumps(doc))
    report = run_skm_import_v5(raw, records, registry, SCOPE, policy, BOX_POLICY, raw / "annotations", private / "t11")
    assert report["failures"] == [] and report["summary"]["cascade"]["primary"] == 1
    evidence, markdown = gdata_evidence_v5(
        scope=SCOPE, policy=policy, prepare=prepare, label_qc={"SKM-TEA_ltr|pass|-|unexpected=False": 1},
        migrate=migrate, split=split, records=records, registry=registry, box_import=report)
    assert {item["status"] for item in evidence.values()} == {"PASS"}
    assert "k1" not in markdown and report["box_import_sha256"] in markdown
