"""Real T10 smoke driver (plan_v5 §2C T10, real-run step 2). CPU, synthetic NIfTI volumes under tmp_path."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.engine.registration_v5 import run_sha256_v5
from anatobind.v41.real_t10_v5 import (
    check_smoke_approval_v5, choose_candidate_v5, render_smoke_md_v5, run_smoke_v5, smoke_plan_sha256_v5,
    synthetic_make_batch_v5, training_smoke_evidence_v5, validate_smoke_plan_v5)
from tests.v5.factories import (
    SYNTH_CODE, SYNTH_ENVIRONMENT, SYNTH_LABEL_MAPS, SYNTH_SOURCE, make_acceptance_policy, make_gate_decisions,
    make_module_scope, make_recipe, make_split_registry, make_supervision, write_volume_case)

REPO = Path(__file__).resolve().parents[2]
SCOPE = validate_scope_v5(make_module_scope("U"))
CANDIDATES = [{"patch_size": [8, 8, 8], "batch_size": 2, "precision": "fp32"},
              {"patch_size": [4, 4, 4], "batch_size": 2, "precision": "fp32"}]


def _raises(code, path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (code, path)


def _plan(**changes) -> dict:
    recipe = make_recipe(model={"tasks": ["U"], "widths": [4, 8], "strides": [[1, 1, 1], [2, 2, 2]],
                                "blocks_per_stage": 1},
                         optimizer={"name": "adamw", "lr": 0.01, "weight_decay": 0.0},
                         inference={"window": [8, 8, 8], "stride": [4, 4, 4]}, checkpoint_every=2,
                         budget={"max_steps": 6, "max_wall_hours": 0.2})
    for key in ("patch_size", "batch_size", "precision"):
        recipe.pop(key)
    plan = {"plan_id": "smoke_synthetic_u_v1", "scope_id": SCOPE.scope_id, "scope_hash": SCOPE.scope_hash,
            "source": dict(SYNTH_SOURCE), "lineage_id": "lin-a", "recipe": recipe,
            "supervision": make_supervision(SCOPE), "acceptance_policy": make_acceptance_policy(SCOPE),
            "candidates": copy.deepcopy(CANDIDATES),
            "candidate_rule": {"name": "first_ok_within_memory", "max_memory_fraction": 0.75},
            "subset": {"salt": "synthetic-salt", "count": 2, "strata": None},
            "overfit": {"criteria": {"min_loss_drop": 0.0, "min_dice": 0.0, "loss_window": 1}},
            "resume": {"pause_at": 4, "tolerances": {"loss_atol": 1e-6, "loss_rtol": 1e-6, "param_rtol": 1e-6},
                       "deterministic": False},
            "budget": {"max_wall_hours": 0.5}}
    plan.update(changes)
    return plan


def _approval(plan, **changes) -> dict:
    return {"approved_by": "synthetic-fixture", "approved_at": "2026-11-09", "scope_hash": SCOPE.scope_hash,
            "smoke_plan_sha256": smoke_plan_sha256_v5(plan), **changes}


def test_plan_binds_scope_source_candidates_and_pause():
    plan = validate_smoke_plan_v5(_plan(), SCOPE, dict(SYNTH_SOURCE))
    assert smoke_plan_sha256_v5(plan) == smoke_plan_sha256_v5(copy.deepcopy(plan))
    _raises("SMOKE_PLAN_SCOPE_MISMATCH", "$.scope_hash", validate_smoke_plan_v5, _plan(scope_hash="0" * 64), SCOPE,
            dict(SYNTH_SOURCE))
    _raises("SMOKE_PLAN_SOURCE_MISMATCH", "$.source", validate_smoke_plan_v5, _plan(), SCOPE,
            {"source_id": "other", "source_version": "v0"})
    with_patch = _plan()
    with_patch["recipe"]["patch_size"] = [8, 8, 8]          # patch, batch and precision come from candidates only
    _raises("INVALID_SMOKE_PLAN", "$.recipe", validate_smoke_plan_v5, with_patch, SCOPE, dict(SYNTH_SOURCE))
    _raises("PATCH_NOT_DIVISIBLE", "$.patch_size", validate_smoke_plan_v5,
            _plan(candidates=[dict(CANDIDATES[0], patch_size=[7, 8, 8])]), SCOPE, dict(SYNTH_SOURCE))
    for pause in (3, 6, 0):
        _raises("INVALID_PAUSE", "$.resume.pause_at", validate_smoke_plan_v5,
                _plan(resume=dict(_plan()["resume"], pause_at=pause)), SCOPE, dict(SYNTH_SOURCE))
    over = _plan(budget={"max_wall_hours": 0.1})            # one training run may not outlast the whole smoke
    _raises("INVALID_SMOKE_PLAN", "$.recipe.budget.max_wall_hours", validate_smoke_plan_v5, over, SCOPE,
            dict(SYNTH_SOURCE))
    dev = _plan()
    dev["recipe"]["readable_roles"] = ["train", "development_val"]
    _raises("SMOKE_ROLES_INVALID", "$.recipe.readable_roles", validate_smoke_plan_v5, dev, SCOPE, dict(SYNTH_SOURCE))


def test_resources_approval_must_name_the_exact_plan():
    plan = _plan()
    digest = smoke_plan_sha256_v5(plan)
    assert check_smoke_approval_v5(_approval(plan), digest, SCOPE, fixture_mode=True)
    _raises("APPROVAL_MISSING", "$.approvals.resources", check_smoke_approval_v5, None, digest, SCOPE, True)
    _raises("SMOKE_PLAN_NOT_APPROVED", "$.approvals.resources.smoke_plan_sha256", check_smoke_approval_v5,
            _approval(plan, smoke_plan_sha256="0" * 64), digest, SCOPE, True)
    _raises("GATE_BINDING_MISMATCH", "$.approvals.resources.scope_hash", check_smoke_approval_v5,
            _approval(plan, scope_hash="0" * 64), digest, SCOPE, True)
    _raises("GATE_FIXTURE_NOT_ALLOWED", "$.approvals.resources", check_smoke_approval_v5, _approval(plan), digest,
            SCOPE, False)


def test_candidate_rule_takes_the_first_full_step_within_memory():
    row = {"patch_size": [8, 8, 8], "batch_size": 2, "precision": "fp32", "status": "OK", "peak_memory_bytes": 10}
    rows = [dict(row, status="OOM", peak_memory_bytes=None), dict(row, batch_size=4, peak_memory_bytes=90),
            dict(row, batch_size=3, peak_memory_bytes=70), dict(row, batch_size=1)]
    rule = {"name": "first_ok_within_memory", "max_memory_fraction": 0.75}
    assert choose_candidate_v5(rows, rule, 100) == {"patch_size": [8, 8, 8], "batch_size": 3, "precision": "fp32"}
    assert choose_candidate_v5(rows, rule, None)["batch_size"] == 4      # no cap off CUDA
    assert choose_candidate_v5(rows[:2], rule, 100) is None


def test_synthetic_batches_measure_without_reading_data():
    from anatobind.v41.engine.smoke_v5 import measure_candidates_v5
    from anatobind.v41.models.resunet_v5 import build_model_v5
    from anatobind.v41.real_t10_v5 import recipe_for_candidate_v5

    plan = _plan()
    rows = measure_candidates_v5(lambda c: build_model_v5(recipe_for_candidate_v5(plan, c), SCOPE),
                                 synthetic_make_batch_v5(SCOPE, ("U",), seed=0), copy.deepcopy(CANDIDATES), "cpu",
                                 recipe=recipe_for_candidate_v5(plan, CANDIDATES[0]), supervision=plan["supervision"])
    assert [row["status"] for row in rows] == ["OK", "OK"]
    assert all(row["optimizer_state_included"] for row in rows)


def _world(tmp_path):
    raw = tmp_path / "raw"
    splits = ["train", "train", "train", "train", "development_val", "locked_test"]
    records = [write_volume_case(raw, SCOPE, f"case-{i}", split=split) for i, split in enumerate(splits)]
    registry = make_split_registry(records, lineage_id="lin-a")
    decision = make_gate_decisions(SCOPE, "smoke")["G-DATA"]
    return raw, records, registry, decision


def _smoke(tmp_path, plan, **changes):
    raw, records, registry, decision = _world(tmp_path)
    args = {"plan": plan, "plan_sha256": smoke_plan_sha256_v5(plan), "scope": SCOPE, "source": dict(SYNTH_SOURCE),
            "records": records, "registry": registry, "raw": raw, "label_maps": SYNTH_LABEL_MAPS,
            "gate_decision": decision, "approval": _approval(plan), "private_root": tmp_path / "private",
            "repo": REPO, "device": "cpu", "run_id": "smoke-synthetic-u", "created_at": "2026-11-09T00:00:00Z",
            "environment": dict(SYNTH_ENVIRONMENT), "code": dict(SYNTH_CODE), "fixture_mode": True}
    args.update(changes)
    return run_smoke_v5(**args), records, registry


def test_end_to_end_smoke_on_synthetic_volumes(tmp_path):
    plan = _plan()
    summary, records, registry = _smoke(tmp_path, plan)
    assert summary["status"] == "PASS" and summary["stop_code"] is None, summary
    root = tmp_path / "private"
    # one registered run, executed twice (continuous; paused at 4 and resumed by a fresh model)
    runs = {name: json.loads((root / name / "runs" / "smoke-synthetic-u" / "run.json").read_text())
            for name in ("continuous", "resumed")}
    assert runs["continuous"] == runs["resumed"] and runs["continuous"]["run_kind"] == "smoke"
    assert runs["continuous"]["split"]["readable_roles"] == ["train"]
    subset = runs["continuous"]["sample_subset"]["sample_uids"]
    assert len(subset) == 2 and all(registry["assignment"][uid] == "train" for uid in subset)
    exposure = json.loads((root / "continuous" / "runs" / "smoke-synthetic-u" / "exposure.json").read_text())
    assert sorted(exposure["entries"]) == subset
    assert summary["run"]["run_sha256"] == run_sha256_v5(runs["continuous"])
    for name in ("continuous", "resumed"):
        assert not (root / name / "runs" / "smoke-synthetic-u" / "failure.json").exists()
    # checks
    assert summary["batch_check"]["channels_in_scope_order"] and summary["batch_check"]["model_grid_is_original"]
    assert summary["candidates"][0]["status"] == "OK" and summary["chosen"] == CANDIDATES[0]
    assert summary["continuous"]["status"] == "COMPLETED" and summary["continuous"]["step"] == 6
    assert summary["resumed"]["pause_status"] == "PAUSED" and summary["resumed"]["pause_step"] == 4
    assert summary["resume_check"]["passed"] and summary["resume_check"]["bitwise_parameters"]
    assert summary["overfit"]["steps_completed"] == 6 and summary["overfit"]["passed"]
    assert summary["inference"]["validated"] and summary["inference"]["reloaded_by_hash"]
    assert summary["inference"]["original_shape_matches"]
    assert json.loads((root / "summary.json").read_text()) == json.loads(json.dumps(summary))
    # the report and the evidence
    markdown = render_smoke_md_v5(summary)
    assert not any(uid in markdown for uid in (r.sample_uid for r in records)) and str(tmp_path) not in markdown
    assert "PASS" in markdown and summary["plan_sha256"] in markdown and "optimizer.step" in markdown
    evidence = training_smoke_evidence_v5(summary, markdown, SCOPE, dict(SYNTH_SOURCE))
    assert evidence["status"] == "PASS" and evidence["scope_hash"] == SCOPE.scope_hash and evidence["task"] == "U"


def test_smoke_stops_and_records_when_no_candidate_fits(tmp_path, monkeypatch):
    import anatobind.v41.real_t10_v5 as driver

    def all_oom(build, make, candidates, device, *, recipe, supervision):
        return [dict(c, status="OOM", failure_phase="training_step", fb_peak_memory_bytes=None,
                     fb_seconds_per_step=0.1, peak_memory_bytes=None, seconds_per_step=None, samples_per_second=None,
                     optimizer_state_included=True, scaler_enabled=False, scheduler="poly") for c in candidates]

    monkeypatch.setattr(driver, "measure_candidates_v5", all_oom)
    summary, _, _ = _smoke(tmp_path, _plan())
    assert summary["status"] == "FAIL" and (summary["stop_code"], summary["stop_phase"]) == \
        ("NO_USABLE_CANDIDATE", "measurement")
    assert summary["run"] is None and not (tmp_path / "private" / "continuous").exists()   # nothing registered
    assert "NO_USABLE_CANDIDATE" in render_smoke_md_v5(summary)


def test_smoke_refuses_an_unapproved_plan_before_reading_anything(tmp_path):
    plan = _plan()
    with pytest.raises(V5Error) as err:
        _smoke(tmp_path, plan, approval=_approval(plan, smoke_plan_sha256="0" * 64))
    assert err.value.code == "SMOKE_PLAN_NOT_APPROVED" and not (tmp_path / "private").exists()
    with pytest.raises(V5Error) as err:
        _smoke(tmp_path / "b", plan, plan_sha256="0" * 64)
    assert err.value.code == "SMOKE_PLAN_HASH_MISMATCH"
    torch.use_deterministic_algorithms(False)


def test_end_to_end_smoke_runs_an_anatomy_scope_too(tmp_path):
    scope = validate_scope_v5(make_module_scope("A"))
    plan = _plan(scope_id=scope.scope_id, scope_hash=scope.scope_hash, supervision=make_supervision(scope),
                 acceptance_policy=make_acceptance_policy(scope))
    plan["recipe"]["model"]["tasks"] = ["A"]
    raw = tmp_path / "raw"
    records = [write_volume_case(raw, scope, f"case-{i}", split=split)
               for i, split in enumerate(["train", "train", "train", "development_val"])]
    registry = make_split_registry(records, lineage_id="lin-a")
    approval = dict(_approval(plan), scope_hash=scope.scope_hash)
    summary = run_smoke_v5(plan=plan, plan_sha256=smoke_plan_sha256_v5(plan), scope=scope, source=dict(SYNTH_SOURCE),
                           records=records, registry=registry, raw=raw, label_maps=SYNTH_LABEL_MAPS,
                           gate_decision=make_gate_decisions(scope, "smoke")["G-DATA"], approval=approval,
                           private_root=tmp_path / "private", repo=REPO, device="cpu", run_id="smoke-synthetic-a",
                           created_at="2026-11-09T00:00:00Z", environment=dict(SYNTH_ENVIRONMENT),
                           code=dict(SYNTH_CODE), fixture_mode=True)
    assert summary["resume_check"]["passed"] and summary["inference"]["original_shape_matches"]
    assert summary["inference"]["dice_classes"] >= 1 and summary["task"] == "A"
    assert "A" in summary["batch_check"]["support"]


def test_batch_check_reads_only_the_subset_and_reports_counts(tmp_path):
    from anatobind.v41.real_t10_v5 import batch_check_v5

    raw, records, registry, _ = _world(tmp_path)
    report = batch_check_v5(_plan(), SCOPE, records, registry, raw, SYNTH_LABEL_MAPS)
    assert report["samples"] == 2 and report["channels_in_scope_order"] and report["model_grid_is_original"]
    assert report["batch"]["image_shape"] == [2, len(SCOPE.input_modalities), 8, 8, 8] and report["batch"]["finite"]
    assert not any(r.sample_uid in json.dumps(report) for r in records)


@pytest.mark.parametrize("scope_id,source_id", [("isles_u", "isles_2022"), ("skm_a", "skm_tea")])
def test_registered_smoke_proposals_validate_and_say_they_are_proposals(scope_id, source_id):
    import yaml

    from anatobind.v41.source_policy_v5 import load_source_policies_v5, source_binding_v5

    path = Path(f"configs/v5/proposals/smoke_{scope_id}.yaml")
    assert path.read_text().startswith("# PROPOSAL — NOT APPROVED")
    scope = validate_scope_v5(yaml.safe_load(Path(f"configs/v5/scope_{scope_id}.yaml").read_text()))
    source = source_binding_v5(load_source_policies_v5(Path("configs/v5/sources_v5.yaml"))[source_id])
    plan = validate_smoke_plan_v5(yaml.safe_load(path.read_text()), scope, source)
    assert plan["recipe"]["downsample_factor"] == [1, 1, 1] and plan["recipe"]["readable_roles"] == ["train"]
    assert plan["supervision"]["track"] == "released_reference"
    assert plan["supervision"]["allow_unknown_visibility"] is (scope_id == "isles_u")
    assert plan["acceptance_policy"]["policy_role"] == "engineering"


def test_gbase_evidence_binds_the_five_items_and_needs_a_passing_smoke():
    from anatobind.v41.contracts_v5 import canonical_sha256
    from anatobind.v41.gates_v5 import GATE_REQUIREMENTS, evaluate_gate_v5
    from anatobind.v41.real_t10_v5 import gbase_evidence_v5

    plan = _plan()
    decision = make_gate_decisions(SCOPE, "smoke")["G-DATA"]
    split_md = "# 冻结划分\n| development_val | 3 |\n"
    smoke = {"status": "PASS", "sha256": "e" * 64, "scope_hash": SCOPE.scope_hash, "task": "U", **SYNTH_SOURCE}
    evidence = gbase_evidence_v5(scope=SCOPE, source=dict(SYNTH_SOURCE), plan=plan, gdata_decision=decision,
                                 split_markdown=split_md, development_groups=3, smoke_evidence=smoke)
    assert set(evidence) == set(GATE_REQUIREMENTS["G-BASE-U"]["evidence"])
    assert all(item["status"] == "PASS" and item["scope_hash"] == SCOPE.scope_hash for item in evidence.values())
    assert evidence["supervision_policy"]["sha256"] == canonical_sha256(plan["supervision"])
    assert evidence["g_data"]["sha256"] == canonical_sha256(decision)
    approvals = {"resources": {"approved_by": "synthetic-fixture", "approved_at": "2026-11-09",
                               "scope_hash": SCOPE.scope_hash}}
    gate = evaluate_gate_v5("G-BASE-U", SCOPE, evidence, approvals, dict(SYNTH_SOURCE), fixture_mode=True)
    assert gate["engineering_status"] == "PASS"
    failed = gbase_evidence_v5(scope=SCOPE, source=dict(SYNTH_SOURCE), plan=plan, gdata_decision=decision,
                               split_markdown=split_md, development_groups=0, smoke_evidence=dict(smoke, status="FAIL"))
    assert failed["training_smoke"]["status"] == "FAIL" and failed["frozen_dev_split"]["status"] == "FAIL"
    with pytest.raises(V5Error) as err:
        gbase_evidence_v5(scope=SCOPE, source=dict(SYNTH_SOURCE), plan=plan, gdata_decision=decision,
                          split_markdown=split_md, development_groups=3,
                          smoke_evidence=dict(smoke, scope_hash="0" * 64))
    assert err.value.code == "EVIDENCE_BINDING_MISMATCH"


def test_gdata_gate_command_evaluates_the_signed_approval(tmp_path):
    import yaml

    from anatobind.v41.real_t10_v5 import main
    from tests.v5.test_real_gdata import APPROVAL, POLICY, SCOPE as ISLES, _TEMPLATE, _evidence

    evidence, _ = _evidence()
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    policies = tmp_path / "sources.yaml"
    policies.write_text(yaml.safe_dump({"isles_2022": {**_TEMPLATE["isles_2022"], "annotation_modalities": {"U": ["dwi"]},
                                                       "index_evidence": POLICY.index_evidence}}))
    approvals = tmp_path / "approvals.yaml"
    approvals.write_text(yaml.safe_dump({"isles_u": {"G-DATA": dict(APPROVAL)}}))
    args = ["gdata-gate", "--evidence", str(tmp_path / "evidence.json"), "--approvals", str(approvals), "--scope",
            "configs/v5/scope_isles_u.yaml", "--policies", str(policies), "--source-id", "isles_2022"]
    assert main(args + ["--out", str(tmp_path / "gate")]) == 0
    decision = json.loads((tmp_path / "gate" / "gdata_decision.json").read_text())
    assert decision["engineering_status"] == "PASS" and decision["scope_hash"] == ISLES.scope_hash
    approvals.write_text(yaml.safe_dump({"isles_u": {}}))
    assert main(args + ["--out", str(tmp_path / "gate2")]) == 0
    blocked = json.loads((tmp_path / "gate2" / "gdata_decision.json").read_text())
    assert blocked["engineering_status"] == "BLOCKED" and blocked["blockers"] == ["APPROVAL_MISSING:data_use"]


def test_gbase_gate_command_binds_split_smoke_and_resources(tmp_path):
    import yaml

    from anatobind.v41.gates_v5 import GATE_REQUIREMENTS, evaluate_gate_v5
    from anatobind.v41.real_t10_v5 import main
    from anatobind.v41.source_policy_v5 import load_source_policies_v5, source_binding_v5
    from tests.v5.factories import make_evidence

    isles = validate_scope_v5(yaml.safe_load(Path("configs/v5/scope_isles_u.yaml").read_text()))
    source = source_binding_v5(load_source_policies_v5(Path("configs/v5/sources_v5.yaml"))["isles_2022"])
    signed = {"approved_by": "researcher", "approved_at": "2026-11-09", "scope_hash": isles.scope_hash}
    decision = evaluate_gate_v5("G-DATA", isles, make_evidence(GATE_REQUIREMENTS["G-DATA"]["evidence"], isles, source),
                                {"data_use": dict(signed, **source)}, source)
    smoke = {"status": "PASS", "sha256": "e" * 64, "scope_hash": isles.scope_hash, "task": "U", **source}
    summary = {"lineage_id": "v51_isles_u_b2", "kind": "stratified", "groups": 4, "samples": 4,
               "counts": {"development_val": 1, "excluded": 0, "locked_test": 1, "train": 2, "unassigned": 0},
               "exposed_groups": 0, "exposed_target": "excluded", "strata": None, "volume_thresholds_ml": None,
               "split_sha256": "f" * 64, "rule": {"seed": 1}}
    files = {"gdata": decision, "smoke": smoke, "split": summary}
    for name, value in files.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    approvals = tmp_path / "approvals.yaml"
    approvals.write_text(yaml.safe_dump({"isles_u": {"resources": dict(signed, smoke_plan_sha256="0" * 64)}}))
    args = ["gbase-gate", "--plan", "configs/v5/proposals/smoke_isles_u.yaml", "--scope",
            "configs/v5/scope_isles_u.yaml", "--policies", "configs/v5/sources_v5.yaml", "--source-id", "isles_2022",
            "--gdata-decision", str(tmp_path / "gdata.json"), "--split-summary", str(tmp_path / "split.json"),
            "--smoke-evidence", str(tmp_path / "smoke.json"), "--approvals", str(approvals)]
    assert main(args + ["--out", str(tmp_path / "smoke_only")]) == 0     # a smoke approval never opens G-BASE
    assert json.loads((tmp_path / "smoke_only" / "gbase_decision.json").read_text())["blockers"] == [
        "APPROVAL_MISSING:resources"]
    approvals.write_text(yaml.safe_dump({"isles_u": {"G-BASE-U": {"resources": signed}}}))
    assert main(args + ["--out", str(tmp_path / "base")]) == 0
    gate = json.loads((tmp_path / "base" / "gbase_decision.json").read_text())
    assert gate["gate_id"] == "G-BASE-U" and gate["engineering_status"] == "PASS" and gate["fixture_mode"] is False
    assert set(json.loads((tmp_path / "base" / "gbase_evidence.json").read_text())) == set(
        GATE_REQUIREMENTS["G-BASE-U"]["evidence"])

    (tmp_path / "smoke.json").write_text(json.dumps(dict(smoke, status="FAIL")))
    approvals.write_text(yaml.safe_dump({"isles_u": {}}))
    assert main(args + ["--out", str(tmp_path / "base2")]) == 0
    blocked = json.loads((tmp_path / "base2" / "gbase_decision.json").read_text())
    assert blocked["blockers"] == ["APPROVAL_MISSING:resources", "EVIDENCE_MISSING:training_smoke"]

    (tmp_path / "split.json").write_text(json.dumps(dict(summary, lineage_id="other")))
    with pytest.raises(V5Error) as err:
        main(args + ["--out", str(tmp_path / "base3")])
    assert err.value.code == "SPLIT_LINEAGE_MISMATCH"
