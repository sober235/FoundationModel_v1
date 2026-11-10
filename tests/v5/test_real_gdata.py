"""G-DATA evidence from the real T07/T08 outputs (plan_v5 §2C T10 real-run prerequisite 2). Synthetic summaries."""
from __future__ import annotations

import copy
import hashlib

import pytest
import yaml
from pathlib import Path

from anatobind.v41.contracts_v5 import V5Error, validate_sample_v5, validate_scope_v5
from anatobind.v41.gates_v5 import GATE_REQUIREMENTS, evaluate_gate_v5
from anatobind.v41.real_gdata_v5 import gdata_evidence_v5
from anatobind.v41.source_policy_v5 import source_binding_v5, validate_source_policy_v5
from tests.v5.factories import make_sample_v5, make_split_registry

SCOPE = validate_scope_v5(yaml.safe_load(Path("configs/v5/scope_isles_u.yaml").read_text()))
SKM_SCOPE = validate_scope_v5(yaml.safe_load(Path("configs/v5/scope_skm_a.yaml").read_text()))
_TEMPLATE = yaml.safe_load(Path("configs/v5/sources_v5.template.yaml").read_text())
# The registered policy cites the source-check summary its prepare run wrote (sha256 "a" * 64 here).
POLICY = validate_source_policy_v5("isles_2022", {**_TEMPLATE["isles_2022"], "annotation_modalities": {"U": ["dwi"]},
                                                  "index_evidence": f"source check summary (sha256 {'a' * 64})"})


def _record(i: int, split: str, **overrides):
    fields = {"sample_uid": f"s-{i}", "split": split, "patient_uid": f"synthetic:p{i}",
              "leakage_group_uid": f"synthetic:p{i}", "active_scope_id": "isles_u",
              "reference_policy_id": "isles_released_reference_v1", "model_modalities": ["dwi", "adc"],
              "source_dataset": "ISLES_ltr", "source_release": "ISLES-2022", "modality_names": ["dwi", "adc"],
              "image_paths": ["dwi.nii.gz", "adc.nii.gz"], "modality_is_synthetic": [False, False],
              "available_modality_mask": [True, True]}
    return validate_sample_v5(make_sample_v5(**{**fields, **overrides}))


RECORDS = [_record(0, "train"), _record(1, "train"), _record(2, "train"), _record(3, "development_val")]
REGISTRY = make_split_registry(RECORDS, lineage_id="v51_isles_u_b2")
REGISTRY_3 = make_split_registry(RECORDS[:3], lineage_id="v51_isles_u_b2")    # migration kept three of four
PREPARE = {"records": 4, "failures": [], "geometry_status": {"pass": 4}, "source_check_sha256": "a" * 64,
           "summary": {"pairing": {"incomplete": 0, "missing_label": 0, "paired": 4},
                       "geometry": {"fail": 0, "pass": 8, "unknown": 0},
                       "channel": {"cases_checked": 0, "consistent": 0, "unevaluated": 0}}}
LABEL_QC = {"ISLES_ltr|pass|-|unexpected=False": 3, "ISLES_ltr|needs_review|ALL_ZERO|unexpected=False": 1}
MIGRATE = {"records": 4, "failures": [], "index_alignment": {"verified": 4}, "physical_geometry": {"verified": 4}}
SPLIT = {"groups": 4, "samples": 4, "split_sha256": REGISTRY["split_sha256"],
         "counts": {"train": 3, "development_val": 1}}
APPROVAL = {"data_use": {"approved_by": "researcher", "approved_at": "2026-11-08", "scope_hash": SCOPE.scope_hash,
                         **source_binding_v5(POLICY)}}


def _evidence(scope=SCOPE, policy=POLICY, **changes):
    args = {"prepare": PREPARE, "label_qc": LABEL_QC, "migrate": MIGRATE, "split": SPLIT}
    args.update(changes)
    chain = {"records": args.pop("records", RECORDS), "registry": args.pop("registry", REGISTRY)}
    return gdata_evidence_v5(scope=scope, policy=policy, **copy.deepcopy(args), **chain)


def test_complete_outputs_give_passing_bound_evidence():
    evidence, markdown = _evidence()
    assert set(evidence) == set(GATE_REQUIREMENTS["G-DATA"]["evidence"])
    digest = hashlib.sha256(markdown.encode()).hexdigest()
    for item in evidence.values():
        assert item == {"status": "PASS", "sha256": digest, "scope_hash": SCOPE.scope_hash, "task": "U",
                        **source_binding_v5(POLICY)}
    decision = evaluate_gate_v5("G-DATA", SCOPE, evidence, APPROVAL, source_binding_v5(POLICY))
    assert decision["engineering_status"] == "PASS" and decision["evidence_hashes"] == [digest] * 6
    assert "a" * 64 in markdown and REGISTRY["split_sha256"] in markdown and "ALL_ZERO" in markdown
    rows = [line for line in markdown.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) == 7 and all(line.count("|") == 4 for line in rows)      # no cell breaks the table


@pytest.mark.parametrize("changes,failed", [
    ({"prepare": {**PREPARE, "failures": [{"index": 0, "code": "X", "path": "$"}]}}, "readability"),
    ({"label_qc": {**LABEL_QC, "ISLES_ltr|fail|BAD|unexpected=True": 1}}, "semantics"),
    ({"prepare": {**PREPARE, "summary": {**PREPARE["summary"],
                                         "pairing": {"incomplete": 0, "missing_label": 1, "paired": 3}}}}, "pairing"),
    ({"prepare": {**PREPARE, "geometry_status": {"pass": 3, "fail": 1}}}, "task_geometry"),
    ({"migrate": {**MIGRATE, "index_alignment": {"verified": 3, "unknown": 1}}}, "task_geometry"),
    ({"migrate": {**MIGRATE, "records": 3, "failures": [{"index": 0, "code": "SOURCE_VERSION_MISMATCH", "path": "$"}],
                  "index_alignment": {"verified": 3}, "physical_geometry": {"verified": 3}},
      "split": {**SPLIT, "samples": 3, "split_sha256": REGISTRY_3["split_sha256"]}, "records": RECORDS[:3],
      "registry": REGISTRY_3}, "provenance"),
    ({"split": {**SPLIT, "samples": 3}}, "identity"),
])
def test_each_shortfall_fails_its_own_item(changes, failed):
    evidence, markdown = _evidence(**changes)
    assert evidence[failed]["status"] == "FAIL"
    assert [k for k, v in evidence.items() if v["status"] != "PASS"] == [failed]
    decision = evaluate_gate_v5("G-DATA", SCOPE, evidence, APPROVAL, source_binding_v5(POLICY))
    assert decision["blockers"] == [f"EVIDENCE_MISSING:{failed}"]


def test_skm_semantics_need_every_channel_check_consistent():
    summary = {**PREPARE["summary"], "channel": {"cases_checked": 4, "consistent": 3, "unevaluated": 1}}
    evidence, _ = _evidence(prepare={**PREPARE, "summary": summary})
    assert evidence["semantics"]["status"] == "FAIL"


def test_command_line_writes_private_evidence(tmp_path, capsys):
    import json

    from anatobind.v41.real_gdata_v5 import main
    import dataclasses

    paths = {}
    for name, value in (("prepare", PREPARE), ("qc", {"label_value_qc_by_source": LABEL_QC}), ("migrate", MIGRATE),
                        ("split", SPLIT), ("registry", REGISTRY)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    paths["records"] = tmp_path / "records_v5.jsonl"
    paths["records"].write_text("".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in RECORDS))
    policies = tmp_path / "sources.yaml"
    policies.write_text(yaml.safe_dump({"isles_2022": {**_TEMPLATE["isles_2022"], "annotation_modalities": {"U": ["dwi"]},
                                                       "index_evidence": POLICY.index_evidence}}))
    out = tmp_path / "gdata"
    assert main(["--prepare-report", str(paths["prepare"]), "--qc-summary", str(paths["qc"]),
                 "--migrate-report", str(paths["migrate"]), "--split-summary", str(paths["split"]),
                 "--records", str(paths["records"]), "--split-registry", str(paths["registry"]),
                 "--scope", "configs/v5/scope_isles_u.yaml", "--policies", str(policies),
                 "--source-id", "isles_2022", "--out", str(out)]) == 0
    evidence = json.loads((out / "gdata_evidence.json").read_text())
    markdown = (out / "gdata_evidence.md").read_text()
    assert evidence == _evidence()[0] and markdown == _evidence()[1] and capsys.readouterr().out == markdown
    assert oct(out.stat().st_mode & 0o777) == "0o700"
    assert oct((out / "gdata_evidence.json").stat().st_mode & 0o777) == "0o600"


def test_scope_must_be_the_policy_scope():
    """Reproduces the reported defect: skm_a with the ISLES policy and ISLES counts gave six PASS items."""
    with pytest.raises(V5Error) as error:
        _evidence(scope=SKM_SCOPE)
    assert (error.value.code, error.value.path) == ("SCOPE_POLICY_MISMATCH", "$.policy.scope_id")


@pytest.mark.parametrize("field,value", [("model_modalities", ["adc", "dwi"]),
                                         ("reference_policy_id", "other_reference_v1"),
                                         ("annotation_modalities", {"A": ["dwi"]})])
def test_policy_must_match_the_scope_inputs_reference_and_task(field, value):
    entry = {**_TEMPLATE["isles_2022"], "annotation_modalities": {"U": ["dwi"]}, "index_evidence": POLICY.index_evidence,
             field: value}
    with pytest.raises(V5Error) as error:
        _evidence(policy=validate_source_policy_v5("isles_2022", entry))
    path = "$.policy.annotation_modalities.U" if field == "annotation_modalities" else f"$.policy.{field}"
    assert (error.value.code, error.value.path) == ("SCOPE_POLICY_MISMATCH", path)   # the missing task is named


def test_source_check_must_be_the_one_the_policy_registered():
    with pytest.raises(V5Error) as error:
        _evidence(prepare={**PREPARE, "source_check_sha256": "c" * 64})
    assert (error.value.code, error.value.path) == ("SOURCE_CHECK_NOT_REGISTERED", "$.prepare.source_check_sha256")
    unregistered = validate_source_policy_v5("isles_2022", {**_TEMPLATE["isles_2022"], "annotation_modalities": {"U": ["dwi"]}})
    with pytest.raises(V5Error) as error:
        _evidence(policy=unregistered)
    assert error.value.code == "SOURCE_CHECK_NOT_REGISTERED"


@pytest.mark.parametrize("changes,code,path", [
    ({"split": {**SPLIT, "split_sha256": "b" * 64}}, "SPLIT_HASH_MISMATCH", "$.split.split_sha256"),
    ({"records": RECORDS[:3]}, "RECORDS_MISMATCH", "$.records"),
    ({"records": RECORDS[:3] + [_record(9, "development_val")]}, "SPLIT_RECORDS_MISMATCH", "$.registry.assignment"),
    ({"records": RECORDS[:3] + [_record(3, "development_val", active_scope_id="skm_a")]}, "RECORD_SCOPE_MISMATCH",
     "$.records[3].active_scope_id"),
    ({"records": RECORDS[:3] + [_record(3, "development_val", model_modalities=["adc", "dwi"])]}, "RECORD_SCOPE_MISMATCH",
     "$.records[3].model_modalities"),
    ({"records": RECORDS[:3] + [_record(3, "development_val", source_release="ISLES-2015")]}, "RECORD_SOURCE_MISMATCH",
     "$.records[3].source_release"),
])
def test_records_and_registry_must_be_the_same_data_chain(changes, code, path):
    with pytest.raises(V5Error) as error:
        _evidence(**changes)
    assert (error.value.code, error.value.path) == (code, path)
