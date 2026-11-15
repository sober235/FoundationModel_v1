"""Real T10 baseline driver (plan_v5 §2C T10 real-run step 3): synthetic volumes only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.real_baseline_v5 import (
    baseline_plan_sha256_v5, check_baseline_approval_v5, render_baseline_md_v5, run_baseline_v5,
    validate_baseline_plan_v5)
from anatobind.v41.real_t10_v5 import recipe_for_candidate_v5, smoke_plan_sha256_v5
from tests.v5.factories import (
    SYNTH_CODE, SYNTH_ENVIRONMENT, SYNTH_LABEL_MAPS, SYNTH_SOURCE, make_gate_decisions, make_metric_policy,
    make_split_registry, write_volume_case)
from tests.v5.test_real_t10 import CANDIDATES, REPO, SCOPE, _plan

SMOKE_SHA = "e" * 64


def _baseline(smoke=None, **changes) -> dict:
    smoke = smoke or _plan()
    chosen = copy.deepcopy(CANDIDATES[0])
    recipe = recipe_for_candidate_v5(smoke, chosen)
    recipe.update(recipe_id="synthetic_baseline_v1", readable_roles=["train", "development_val"],
                  budget={"max_steps": 6, "max_wall_hours": 0.2})
    plan = {"plan_id": "baseline_synthetic_u_v1", "scope_id": SCOPE.scope_id, "scope_hash": SCOPE.scope_hash,
            "source": dict(SYNTH_SOURCE), "lineage_id": smoke["lineage_id"],
            "smoke": {"plan_sha256": smoke_plan_sha256_v5(smoke), "training_smoke_sha256": SMOKE_SHA,
                      "chosen": chosen},
            "recipe": recipe, "supervision": copy.deepcopy(smoke["supervision"]),
            "acceptance_policy": copy.deepcopy(smoke["acceptance_policy"]),
            "metric_policy": make_metric_policy(), "budget": {"max_wall_hours": 0.5}}
    plan.update(changes)
    return plan


def _approval(plan, **changes) -> dict:
    return {"approved_by": "synthetic-fixture", "approved_at": "2026-11-09", "scope_hash": SCOPE.scope_hash,
            "baseline_plan_sha256": baseline_plan_sha256_v5(plan), **changes}


def _smoke_evidence(**changes) -> dict:
    return {"status": "PASS", "sha256": SMOKE_SHA, "scope_hash": SCOPE.scope_hash, "task": "U", **SYNTH_SOURCE,
            **changes}


def _raises(code, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert err.value.code == code, err.value
    return err.value


def test_baseline_plan_must_match_the_passed_smoke():
    smoke = _plan()
    plan = validate_baseline_plan_v5(_baseline(smoke), SCOPE, dict(SYNTH_SOURCE), smoke)
    assert sorted(plan["recipe"]["readable_roles"]) == ["development_val", "train"]
    bad = _baseline(smoke)
    bad["recipe"]["optimizer"] = dict(bad["recipe"]["optimizer"], lr=0.5)
    assert _raises("BASELINE_RECIPE_MISMATCH", validate_baseline_plan_v5, bad, SCOPE, dict(SYNTH_SOURCE),
                   smoke).path == "$.recipe.optimizer"
    bad = _baseline(smoke)
    bad["recipe"]["precision"] = "amp_bf16"                            # not the frozen candidate
    _raises("BASELINE_RECIPE_MISMATCH", validate_baseline_plan_v5, bad, SCOPE, dict(SYNTH_SOURCE), smoke)
    bad = _baseline(smoke)
    bad["smoke"]["chosen"] = {"patch_size": [64, 64, 64], "batch_size": 1, "precision": "fp32"}
    _raises("BASELINE_CANDIDATE_UNKNOWN", validate_baseline_plan_v5, bad, SCOPE, dict(SYNTH_SOURCE), smoke)
    bad = _baseline(smoke)
    bad["smoke"]["plan_sha256"] = "0" * 64
    _raises("SMOKE_PLAN_MISMATCH", validate_baseline_plan_v5, bad, SCOPE, dict(SYNTH_SOURCE), smoke)
    bad = _baseline(smoke)
    bad["supervision"]["tasks"]["U"]["dice_weight"] = 2.0
    _raises("BASELINE_POLICY_MISMATCH", validate_baseline_plan_v5, bad, SCOPE, dict(SYNTH_SOURCE), smoke)
    bad = _baseline(smoke)
    bad["recipe"]["readable_roles"] = ["train"]
    _raises("BASELINE_ROLES_INVALID", validate_baseline_plan_v5, bad, SCOPE, dict(SYNTH_SOURCE), smoke)
    _raises("METRIC_POLICY_INCOMPLETE", validate_baseline_plan_v5,
            _baseline(smoke, metric_policy=make_metric_policy(iou_match_threshold=None)), SCOPE,
            dict(SYNTH_SOURCE), smoke)
    _raises("INVALID_BASELINE_PLAN", validate_baseline_plan_v5, _baseline(smoke, budget={"max_wall_hours": 0.1}),
            SCOPE, dict(SYNTH_SOURCE), smoke)
    _raises("SMOKE_PLAN_SCOPE_MISMATCH", validate_baseline_plan_v5, _baseline(smoke, scope_hash="0" * 64), SCOPE,
            dict(SYNTH_SOURCE), smoke)


def test_baseline_needs_the_approval_naming_its_plan():
    plan = _baseline()
    digest = baseline_plan_sha256_v5(plan)
    assert check_baseline_approval_v5(_approval(plan), digest, SCOPE, fixture_mode=True)
    _raises("APPROVAL_MISSING", check_baseline_approval_v5, None, digest, SCOPE, fixture_mode=True)
    _raises("BASELINE_PLAN_NOT_APPROVED", check_baseline_approval_v5,
            {k: v for k, v in _approval(plan).items() if k != "baseline_plan_sha256"}, digest, SCOPE,
            fixture_mode=True)
    _raises("GATE_FIXTURE_NOT_ALLOWED", check_baseline_approval_v5, _approval(plan), digest, SCOPE,
            fixture_mode=False)


def _world(tmp_path):
    raw = tmp_path / "raw"
    splits = ["train", "train", "train", "development_val", "development_val", "locked_test"]
    records = [write_volume_case(raw, SCOPE, f"case-{i}", split=split) for i, split in enumerate(splits)]
    return raw, records, make_split_registry(records, lineage_id="lin-a")


def _run(tmp_path, plan, **changes):
    raw, records, registry = _world(tmp_path)
    args = {"plan": plan, "plan_sha256": baseline_plan_sha256_v5(plan), "scope": SCOPE, "source": dict(SYNTH_SOURCE),
            "records": records, "registry": registry, "raw": raw, "label_maps": SYNTH_LABEL_MAPS,
            "gate_decision": make_gate_decisions(SCOPE, "baseline")["G-BASE-U"], "approval": _approval(plan),
            "smoke_evidence": _smoke_evidence(), "private_root": tmp_path / "private", "repo": REPO,
            "device": "cpu", "run_id": "baseline-synthetic-u", "created_at": "2026-11-09T00:00:00Z",
            "environment": dict(SYNTH_ENVIRONMENT), "code": dict(SYNTH_CODE), "fixture_mode": True}
    args.update(changes)
    return run_baseline_v5(**args), records, registry


def test_end_to_end_baseline_trains_on_train_and_scores_every_development_scan(tmp_path):
    summary, records, registry = _run(tmp_path, _baseline())
    assert summary["status"] == "COMPLETED" and summary["stop_code"] is None, summary
    root = tmp_path / "private"
    run = json.loads((root / "runs" / "baseline-synthetic-u" / "run.json").read_text())
    assert run["run_kind"] == "baseline" and run["split"]["readable_roles"] == ["development_val", "train"]
    assert run["parents"] == [] and run["approvals"]["resources"]["baseline_plan_sha256"] == summary["plan_sha256"]
    exposure = json.loads((root / "runs" / "baseline-synthetic-u" / "exposure.json").read_text())["entries"]
    roles = {uid: {e["role"] for e in entries} for uid, entries in exposure.items()}
    by_split = {r.sample_uid: registry["assignment"][r.sample_uid] for r in records}
    assert {uid for uid, split in by_split.items() if split in ("train", "development_val")} == set(roles)
    assert all(roles[uid] == {"training" if by_split[uid] == "train" else "model_selection"} for uid in roles)
    assert summary["training"]["status"] == "COMPLETED" and summary["training"]["step"] == 6
    module = summary["module"]
    assert module["n_scans"] == 2 and module["split"] == "development_val"
    assert module["scientific_conclusion"] == "NOT_EVALUATED"
    assert len(list((root / "inference").glob("*.record.json"))) == 2       # development_val only
    markdown = render_baseline_md_v5(summary)
    assert not any(r.sample_uid in markdown for r in records) and str(tmp_path) not in markdown
    assert summary["plan_sha256"] in markdown and "development_val" in markdown and "NOT_EVALUATED" in markdown
    assert json.loads((root / "summary.json").read_text()) == json.loads(json.dumps(summary))


def test_baseline_refuses_unapproved_or_unbound_inputs_before_reading_data(tmp_path):
    plan = _baseline()
    _raises("BASELINE_PLAN_NOT_APPROVED", _run, tmp_path / "a", plan, approval=_approval(plan, baseline_plan_sha256="0" * 64))
    _raises("SMOKE_EVIDENCE_MISMATCH", _run, tmp_path / "b", plan, smoke_evidence=_smoke_evidence(sha256="1" * 64))
    _raises("SMOKE_EVIDENCE_MISMATCH", _run, tmp_path / "c", plan, smoke_evidence=_smoke_evidence(status="FAIL"))
    blocked = dict(make_gate_decisions(SCOPE, "baseline")["G-BASE-U"], engineering_status="BLOCKED")
    _raises("GATE_NOT_PASSED", _run, tmp_path / "d", plan, gate_decision=blocked)
    for name in ("a", "b", "c", "d"):
        assert not (tmp_path / name / "private" / "runs").exists()


def test_baseline_stops_and_records_a_training_failure(tmp_path, monkeypatch):
    import anatobind.v41.real_baseline_v5 as driver

    def failing(*args, **kwargs):
        return {"status": "FAILED", "step": 2, "failure_code": "CUDA_OOM", "last_checkpoint": None,
                "last_checkpoint_sha256": None}

    monkeypatch.setattr(driver, "train_v5", failing)
    summary, _, _ = _run(tmp_path, _baseline())
    assert summary["status"] == "FAILED" and summary["stop_code"] == "CUDA_OOM" and summary["stop_phase"] == "training"
    assert summary["module"] is None
    assert json.loads((tmp_path / "private" / "summary.json").read_text())["stop_code"] == "CUDA_OOM"


@pytest.mark.parametrize("scope_id, source_id", [("isles_u", "isles_2022"), ("skm_a", "skm_tea")])
def test_registered_baseline_proposals_follow_their_smokes(scope_id, source_id):
    from anatobind.v41.source_policy_v5 import load_source_policies_v5, source_binding_v5

    scope = validate_scope_v5(yaml.safe_load(Path(f"configs/v5/scope_{scope_id}.yaml").read_text()))
    source = source_binding_v5(load_source_policies_v5(Path("configs/v5/sources_v5.yaml"))[source_id])
    smoke = yaml.safe_load(Path(f"configs/v5/proposals/smoke_{scope_id}.yaml").read_text())
    path = Path(f"configs/v5/proposals/baseline_{scope_id}.yaml")
    assert path.read_text().startswith("# PROPOSAL — NOT APPROVED")
    plan = validate_baseline_plan_v5(yaml.safe_load(path.read_text()), scope, source, smoke)
    assert plan["smoke"]["chosen"] == smoke["candidates"][0]                     # the rule's choice in the smoke
    assert plan["recipe"]["downsample_factor"] == [1, 1, 1] and plan["recipe"]["selection_rule"] == "last"


def _cli_args(tmp_path, scope_id="isles_u", source_id="isles_2022"):
    return ["--plan", f"configs/v5/proposals/baseline_{scope_id}.yaml",
            "--smoke-plan", f"configs/v5/proposals/smoke_{scope_id}.yaml", "--scope", f"configs/v5/scope_{scope_id}.yaml",
            "--policies", "configs/v5/sources_v5.yaml", "--source-id", source_id]


@pytest.mark.parametrize("scope_id, source_id", [("isles_u", "isles_2022"), ("skm_a", "skm_tea")])
def test_plan_hash_command_prints_the_registered_plan_hash(scope_id, source_id, capsys):
    from anatobind.v41.real_baseline_v5 import main

    assert main(["plan-hash"] + _cli_args(None, scope_id, source_id)) == 0
    plan = yaml.safe_load(Path(f"configs/v5/proposals/baseline_{scope_id}.yaml").read_text())
    assert capsys.readouterr().out.strip() == baseline_plan_sha256_v5(plan)


def test_baseline_command_refuses_without_approval_c_before_reading_data(tmp_path):
    from anatobind.v41.real_baseline_v5 import main

    plan = yaml.safe_load(Path("configs/v5/proposals/baseline_isles_u.yaml").read_text())
    isles = validate_scope_v5(yaml.safe_load(Path("configs/v5/scope_isles_u.yaml").read_text()))
    signed = {"approved_by": "researcher", "approved_at": "2026-11-09", "scope_hash": isles.scope_hash}
    approvals = tmp_path / "approvals.yaml"
    missing = tmp_path / "missing"                                          # records, registry, decision: never read
    args = ["baseline"] + _cli_args(tmp_path) + [
        "--raw", str(missing), "--records", str(missing / "r.jsonl"), "--split-registry", str(missing / "s.json"),
        "--gbase-decision", str(missing / "d.json"), "--smoke-evidence", str(missing / "e.json"),
        "--approvals", str(approvals), "--run-id", "baseline-isles-u", "--device", "cpu", "--out", str(tmp_path / "out")]
    cases = [({"isles_u": {"resources": dict(signed, baseline_plan_sha256=baseline_plan_sha256_v5(plan))}},
              "APPROVAL_MISSING"),                                          # a scope-level approval is not G-BASE's
             ({"isles_u": {"G-BASE-U": {"resources": signed}}}, "BASELINE_PLAN_NOT_APPROVED"),
             ({"isles_u": {"G-BASE-U": {"resources": dict(signed, approved_by="synthetic-fixture",
                                                          baseline_plan_sha256=baseline_plan_sha256_v5(plan))}}},
              "GATE_FIXTURE_NOT_ALLOWED")]
    for content, code in cases:
        approvals.write_text(yaml.safe_dump(content))
        _raises(code, main, args)
        assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("ticks, phase", [([0.0], "training"), ([0.0, 0.0], "evaluation")])
def test_baseline_stops_when_its_plan_budget_runs_out(tmp_path, monkeypatch, ticks, phase):
    import anatobind.v41.real_baseline_v5 as driver

    class Clock:                                       # the driver's clock: ticks, then far past the plan budget
        def __init__(self):
            self.values = list(ticks)

        def monotonic(self):
            return self.values.pop(0) if self.values else 1.0e9

    monkeypatch.setattr(driver, "time", Clock())
    summary, _, _ = _run(tmp_path, _baseline())
    assert summary["status"] == "FAILED" and summary["stop_code"] == "BASELINE_BUDGET_EXCEEDED"
    assert summary["stop_phase"] == phase and summary["module"] is None
    assert summary["training"]["status"] == "COMPLETED"
    assert not list((tmp_path / "private").glob("inference/*.record.json"))
