"""G-DATA evidence from the real T07/T08 outputs (plan_v5 §2C T10 real-run prerequisite 2). Synthetic summaries."""
from __future__ import annotations

import copy
import hashlib

import pytest
import yaml
from pathlib import Path

from anatobind.v41.contracts_v5 import validate_scope_v5
from anatobind.v41.gates_v5 import GATE_REQUIREMENTS, evaluate_gate_v5
from anatobind.v41.real_gdata_v5 import gdata_evidence_v5
from anatobind.v41.source_policy_v5 import load_source_policies_v5, source_binding_v5

SCOPE = validate_scope_v5(yaml.safe_load(Path("configs/v5/scope_isles_u.yaml").read_text()))
POLICY = load_source_policies_v5(Path("configs/v5/sources_v5.template.yaml"))["isles_2022"]
PREPARE = {"records": 4, "failures": [], "geometry_status": {"pass": 4}, "source_check_sha256": "a" * 64,
           "summary": {"pairing": {"incomplete": 0, "missing_label": 0, "paired": 4},
                       "geometry": {"fail": 0, "pass": 8, "unknown": 0},
                       "channel": {"cases_checked": 0, "consistent": 0, "unevaluated": 0}}}
LABEL_QC = {"ISLES_ltr|pass|-|unexpected=False": 3, "ISLES_ltr|needs_review|ALL_ZERO|unexpected=False": 1}
MIGRATE = {"records": 4, "failures": [], "index_alignment": {"verified": 4}, "physical_geometry": {"verified": 4}}
SPLIT = {"groups": 4, "samples": 4, "split_sha256": "b" * 64, "counts": {"train": 3, "development_val": 1}}
APPROVAL = {"data_use": {"approved_by": "researcher", "approved_at": "2026-11-08", "scope_hash": SCOPE.scope_hash,
                         **source_binding_v5(POLICY)}}


def _evidence(**changes):
    args = {"prepare": PREPARE, "label_qc": LABEL_QC, "migrate": MIGRATE, "split": SPLIT}
    args.update(changes)
    return gdata_evidence_v5(scope=SCOPE, policy=POLICY, **copy.deepcopy(args))


def test_complete_outputs_give_passing_bound_evidence():
    evidence, markdown = _evidence()
    assert set(evidence) == set(GATE_REQUIREMENTS["G-DATA"]["evidence"])
    digest = hashlib.sha256(markdown.encode()).hexdigest()
    for item in evidence.values():
        assert item == {"status": "PASS", "sha256": digest, "scope_hash": SCOPE.scope_hash, "task": "U",
                        **source_binding_v5(POLICY)}
    decision = evaluate_gate_v5("G-DATA", SCOPE, evidence, APPROVAL, source_binding_v5(POLICY))
    assert decision["engineering_status"] == "PASS" and decision["evidence_hashes"] == [digest] * 6
    assert "a" * 64 in markdown and "b" * 64 in markdown and "ALL_ZERO" in markdown


@pytest.mark.parametrize("changes,failed", [
    ({"prepare": {**PREPARE, "failures": [{"index": 0, "code": "X", "path": "$"}]}}, "readability"),
    ({"label_qc": {**LABEL_QC, "ISLES_ltr|fail|BAD|unexpected=True": 1}}, "semantics"),
    ({"prepare": {**PREPARE, "summary": {**PREPARE["summary"],
                                         "pairing": {"incomplete": 0, "missing_label": 1, "paired": 3}}}}, "pairing"),
    ({"prepare": {**PREPARE, "geometry_status": {"pass": 3, "fail": 1}}}, "task_geometry"),
    ({"migrate": {**MIGRATE, "index_alignment": {"verified": 3, "unknown": 1}}}, "task_geometry"),
    ({"migrate": {**MIGRATE, "records": 3, "failures": [{"index": 0, "code": "SOURCE_VERSION_MISMATCH", "path": "$"}],
                  "index_alignment": {"verified": 3}, "physical_geometry": {"verified": 3}},
      "split": {**SPLIT, "samples": 3}}, "provenance"),
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
    paths = {}
    for name, value in (("prepare", PREPARE), ("qc", {"label_value_qc_by_source": LABEL_QC}), ("migrate", MIGRATE),
                        ("split", SPLIT)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    out = tmp_path / "gdata"
    assert main(["--prepare-report", str(paths["prepare"]), "--qc-summary", str(paths["qc"]),
                 "--migrate-report", str(paths["migrate"]), "--split-summary", str(paths["split"]),
                 "--scope", "configs/v5/scope_isles_u.yaml", "--policies", "configs/v5/sources_v5.template.yaml",
                 "--source-id", "isles_2022", "--out", str(out)]) == 0
    evidence = json.loads((out / "gdata_evidence.json").read_text())
    markdown = (out / "gdata_evidence.md").read_text()
    assert evidence == _evidence()[0] and markdown == _evidence()[1] and capsys.readouterr().out == markdown
    assert oct(out.stat().st_mode & 0o777) == "0o700"
    assert oct((out / "gdata_evidence.json").stat().st_mode & 0o777) == "0o600"
