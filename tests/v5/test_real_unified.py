"""Unified driver: smoke, training and the development report (design 2026-11-10 §5.3–§5.6). Synthetic volumes, CPU."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from anatobind.v41 import real_unified_v5 as real_unified_module
from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.engine.unified_registration_v5 import unified_plan_sha256_v5
from anatobind.v41.gates_v5 import gate_task_v5
from anatobind.v41.real_t10_v5 import training_smoke_evidence_v5
from anatobind.v41.real_unified_v5 import (
    check_unified_approval_v5, main, render_unified_md_v5, render_unified_smoke_md_v5, run_unified_smoke_v5,
    run_unified_training_v5)
from tests.v5.factories import (
    SYNTH_CODE, SYNTH_ENVIRONMENT, SYNTH_LABEL_MAPS, SYNTH_SOURCE, make_gate_decisions, make_module_scope, make_scope,
    make_split_registry, make_unified_plan, write_volume_case)

REPO = Path(__file__).resolve().parents[2]
SCOPES = {"knee": validate_scope_v5(make_scope()),
          "brain": validate_scope_v5(make_module_scope("U", input_modalities=["synthetic_dwi", "synthetic_adc"]))}
BOX_POLICY = {"policy_sha256": "p" * 64, "host_tissues": {"TA": {"group": "g_a", "anatomy": ["tissue_a"]},
                                                          "TB": {"group": "g_b", "anatomy": ["tissue_b"]}}}
PROTOCOLS = {
    "knee": {"scope": SCOPES["knee"], "source_id": SYNTH_SOURCE["source_id"], "description": "synthetic knee",
             "box_policy": BOX_POLICY, "groups": {"g_a": ["tissue_a"], "g_b": ["tissue_b"]},
             "groups_by_class": {"target_a": ["g_a", "g_b"], "target_b": ["g_b"]},
             "display": {"target_a": "异常甲", "target_b": "异常乙", "g_a": "组织甲", "g_b": "组织乙"}},
    "brain": {"scope": SCOPES["brain"], "source_id": SYNTH_SOURCE["source_id"], "description": "synthetic brain",
              "box_policy": None, "groups": None, "groups_by_class": None, "display": {"target_a": "异常甲"}}}
SPLITS = ["train", "train", "development_val"]


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _box(start, size):
    return {"status": "included", "analysis": "primary", "abnormality_class": "target_a",
            "box": {"start": start, "size": size}, "host": {"tissue_group": "g_a"}}


def _approval(scope, plan_sha256):
    return {"approved_by": "synthetic-fixture", "approved_at": "2026-11-10", "scope_hash": scope.scope_hash,
            "unified_plan_sha256": plan_sha256}


@pytest.fixture
def env(tmp_path):
    raw = tmp_path / "raw"
    plan = make_unified_plan(SCOPES)
    digest = unified_plan_sha256_v5(plan)
    boxes = tmp_path / "box_references.jsonl"
    boxes.write_text("".join(json.dumps({"sample_uid": uid, "policy_sha256": "p" * 64, "boxes": items}) + "\n"
                             for uid, items in (("knee-0", [_box([1, 1, 1], [2, 2, 2])]), ("knee-1", []),
                                                ("knee-2", [_box([0, 0, 0], [3, 3, 3])]))))
    members = {}
    for pid, scope in SCOPES.items():
        records = [write_volume_case(raw, scope, f"{pid}-{i}", split=split) for i, split in enumerate(SPLITS)]
        members[pid] = {
            "records": records, "registry": make_split_registry(records, lineage_id=f"lin-{pid}"),
            "source": dict(SYNTH_SOURCE), "gdata_decision": make_gate_decisions(scope, "smoke")["G-DATA"],
            "gbase_decisions": make_gate_decisions(scope, "baseline"),
            "smoke_evidence": {"status": "PASS", "sha256": "e" * 64, "scope_hash": scope.scope_hash,
                               "task": gate_task_v5(scope), **SYNTH_SOURCE},
            "box_references": boxes if pid == "knee" else None}
    common = {"plan": plan, "plan_sha256": digest, "protocols": PROTOCOLS, "members": members,
              "approvals": {pid: _approval(scope, digest) for pid, scope in SCOPES.items()}, "raw": raw,
              "label_maps": SYNTH_LABEL_MAPS, "repo": REPO, "device": "cpu", "created_at": "2026-11-10T00:00:00Z",
              "environment": dict(SYNTH_ENVIRONMENT), "code": dict(SYNTH_CODE), "fixture_mode": True}
    return SimpleNamespace(
        raw=raw, plan=plan, digest=digest,
        smoke_kwargs=dict(common, private_root=tmp_path / "smoke", run_id="smoke-u"),
        train_kwargs=dict(common, private_root=tmp_path / "train", run_id="train-u"))


def test_smoke_passes_and_gives_evidence_bound_to_each_member(env):
    summary = run_unified_smoke_v5(**env.smoke_kwargs)
    assert summary["status"] == "PASS", summary
    assert summary["resume"]["passed"] and summary["resume"]["bitwise_parameters"]
    assert {pid: member["samples"] for pid, member in summary["members"].items()} == {"brain": 1, "knee": 1}
    markdown = render_unified_smoke_md_v5(summary)
    assert str(env.raw) not in markdown and "knee-0" not in markdown and "NOT_EVALUATED" in markdown
    for pid, scope in SCOPES.items():
        evidence = training_smoke_evidence_v5(summary, markdown, scope, dict(SYNTH_SOURCE))
        assert evidence["status"] == "PASS" and evidence["scope_hash"] == scope.scope_hash
    root = env.smoke_kwargs["private_root"]
    assert json.loads((root / "summary.json").read_text())["status"] == "PASS"
    exposure = json.loads((root / "continuous" / "runs" / "smoke-u.knee" / "exposure.json").read_text())
    assert len(exposure["entries"]) == 1                                # the registered subset only


def test_training_runs_end_to_end_and_reports_both_members(env):
    summary = run_unified_training_v5(**env.train_kwargs)
    assert summary["status"] == "COMPLETED" and summary["stop_code"] is None, summary
    assert summary["training"]["step"] == 4 and summary["training"]["samples"] == {"brain": 2, "knee": 2}
    assert set(summary["members"]) == {"brain", "knee"}
    assert summary["members"]["brain"]["module"]["n_scans"] == 1
    knee = summary["members"]["knee"]
    assert set(knee) >= {"anatomy_dice", "findings", "sentences"}
    assert knee["findings"]["overall"]["n_patients"] == 1 and knee["findings"]["overall"]["n_boxes"] == 1
    assert set(knee["anatomy_dice"]) == {"tissue_a", "tissue_b"}
    assert sum(knee["sentences"].values()) >= 1
    text = render_unified_md_v5(summary)
    assert "NOT_EVALUATED" in text and "不是误报" in text and "医生没有标注不等于正常" in text and "未提供" in text
    assert str(env.raw) not in text and "knee-2" not in text and "brain-2" not in text
    root = env.train_kwargs["private_root"]
    exposure = json.loads((root / "runs" / "train-u.knee" / "exposure.json").read_text())
    roles = sorted(entry[0]["role"] for entry in exposure["entries"].values())
    assert roles == ["model_selection", "training", "training"]


def test_the_report_puts_baseline_values_next_to_the_unified_ones(env):
    summary = run_unified_training_v5(**env.train_kwargs)
    brain = copy.deepcopy(summary["members"]["brain"]["module"])
    text = render_unified_md_v5(summary, baseline_summaries={"brain": {"module": brain}})
    assert "未提供" in text and "| brain |" in text and "基线" in text


def test_nothing_is_read_before_the_plan_is_approved(env, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("data was read before the approval check")

    monkeypatch.setattr(real_unified_module, "load_member_samples_v5", _fail_if_called)
    approvals = copy.deepcopy(env.train_kwargs["approvals"])
    approvals["knee"]["unified_plan_sha256"] = "0" * 64
    _raises("UNIFIED_PLAN_NOT_APPROVED", "$.approvals.resources.unified_plan_sha256", run_unified_training_v5,
            **dict(env.train_kwargs, approvals=approvals))
    members = copy.copy(env.train_kwargs["members"])
    members["brain"] = dict(members["brain"], smoke_evidence=dict(members["brain"]["smoke_evidence"], status="FAIL"))
    _raises("SMOKE_EVIDENCE_MISMATCH", "$.members.brain.smoke_evidence", run_unified_training_v5,
            **dict(env.train_kwargs, members=members))
    assert not env.train_kwargs["private_root"].exists()
    scope = SCOPES["knee"]
    _raises("GATE_FIXTURE_NOT_ALLOWED", "$.approvals.resources", check_unified_approval_v5,
            _approval(scope, env.digest), env.digest, scope, False)


def test_training_never_holds_development_boxes(env, monkeypatch):
    seen = []
    original = real_unified_module.load_box_references_v5

    def recording(path, registry, roles):
        seen.append(list(roles))
        return original(path, registry, roles)

    monkeypatch.setattr(real_unified_module, "load_box_references_v5", recording)
    run_unified_training_v5(**env.train_kwargs)
    assert seen == [["train"], ["development_val"]]


def test_plan_hash_command_prints_the_hash(capsys):
    assert main(["plan-hash", "--plan", "configs/v5/proposals/unified_knee_brain.yaml", "--protocols",
                 "configs/v5/protocols_v5.yaml", "--policies", "configs/v5/sources_v5.yaml"]) == 0
    assert re.fullmatch(r"[0-9a-f]{64}\n", capsys.readouterr().out)
