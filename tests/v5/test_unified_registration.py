"""Unified plan, member recipes and the unified run record (Proposal_v5 §0.6; design 2026-11-10 §5.3)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from anatobind.v41.annotation_v5 import load_skm_box_policy_v5
from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.engine.registration_v5 import run_sha256_v5, validate_recipe_v5
from anatobind.v41.engine.unified_registration_v5 import (
    member_recipe_v5, register_unified_run_v5, unified_plan_sha256_v5, validate_unified_plan_v5, write_unified_run_v5)
from anatobind.v41.models.unified_v5 import build_unified_model_v5
from anatobind.v41.paths import PrivatePathError
from anatobind.v41.source_policy_v5 import load_source_policies_v5, source_binding_v5
from tests.v5.factories import (
    SYNTH_CODE, SYNTH_CREATED_AT, SYNTH_ENVIRONMENT, SYNTH_SOURCE, make_module_scope, make_scope,
    make_unified_members, make_unified_plan)

REPO = Path(__file__).resolve().parents[2]
SCOPES = {"knee": validate_scope_v5(make_scope()),
          "brain": validate_scope_v5(make_module_scope("U", input_modalities=["synthetic_dwi", "synthetic_adc"]))}
SOURCES = {pid: dict(SYNTH_SOURCE) for pid in SCOPES}


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _scope(name):
    return validate_scope_v5(yaml.safe_load(Path(f"configs/v5/scope_{name}.yaml").read_text()))


def test_the_proposed_plan_file_is_valid_and_bound_to_the_real_scopes():
    plan = yaml.safe_load(Path("configs/v5/proposals/unified_knee_brain.yaml").read_text())
    scopes = {"knee_qdess": _scope("skm_joint_early"), "brain_dwi_adc": _scope("isles_u")}
    policies = load_source_policies_v5(Path("configs/v5/sources_v5.yaml"))
    sources = {"knee_qdess": source_binding_v5(policies["skm_tea_joint_early"]),
               "brain_dwi_adc": source_binding_v5(policies["isles_2022"])}
    assert validate_unified_plan_v5(plan, scopes, sources) is plan
    baseline = yaml.safe_load(Path("configs/v5/proposals/baseline_isles_u.yaml").read_text())
    brain = plan["members"]["brain_dwi_adc"]
    for key in ("supervision", "acceptance_policy", "metric_policy"):
        assert brain[key] == baseline[key]                     # brain metrics stay comparable with the baseline
    box_policy = load_skm_box_policy_v5(Path("configs/v5/skm_box_policy_v1.yaml"))
    assert plan["members"]["knee_qdess"]["box_supervision"]["box_policy_sha256"] == box_policy["policy_sha256"]
    build_unified_model_v5(plan, scopes)                       # patch and window divide the shared strides


def test_member_recipes_are_valid_recipes_of_both_kinds():
    plan = make_unified_plan(SCOPES)
    recipe = member_recipe_v5(plan, "knee", "baseline")
    assert validate_recipe_v5(recipe) is recipe and recipe["recipe_id"] == f"{plan['plan_id']}.knee"
    assert recipe["model"]["tasks"] == ["A", "U"] and "stem_channels" not in recipe["model"]
    smoke = member_recipe_v5(plan, "knee", "smoke")
    assert smoke["readable_roles"] == ["train"] and smoke["budget"]["max_steps"] == plan["smoke"]["steps"]


@pytest.mark.parametrize("edit,code,path", [
    (lambda p: p["members"]["knee"].update(scope_hash="0" * 64), "UNIFIED_SCOPE_MISMATCH", "$.members.knee.scope_hash"),
    (lambda p: p["members"]["knee"].update(source={"source_id": "x", "source_version": "y"}),
     "UNIFIED_SOURCE_MISMATCH", "$.members.knee.source"),
    (lambda p: p["members"]["brain"].update(box_supervision=dict(p["members"]["knee"]["box_supervision"]),
                                             downsample_factor=[2, 2, 1]),
     "UNIFIED_PLAN_INVALID", "$.members.brain.box_supervision"),
    (lambda p: p["members"]["brain"]["finding_policy"].update(host="majority_accepted_anatomy"),
     "UNIFIED_PLAN_INVALID", "$.members.brain.finding_policy.host"),
    (lambda p: p["smoke"].update(pause_at=3), "UNIFIED_PLAN_INVALID", "$.smoke.pause_at"),
    (lambda p: p.pop("budget"), "UNIFIED_PLAN_INVALID", "$")])
def test_invalid_plans_are_refused(edit, code, path):
    plan = make_unified_plan(SCOPES)
    assert validate_unified_plan_v5(copy.deepcopy(plan), SCOPES, SOURCES)
    edit(plan)
    _raises(code, path, validate_unified_plan_v5, plan, SCOPES, SOURCES)


def test_a_unified_run_binds_its_member_runs_and_writes_their_exposure(tmp_path):
    plan = make_unified_plan(SCOPES)
    member_runs, registries, _ = make_unified_members(plan, SCOPES, "run-u")
    run = register_unified_run_v5(run_id="run-u", run_kind="baseline", plan=plan,
                                  plan_sha256=unified_plan_sha256_v5(plan), member_runs=member_runs,
                                  environment=dict(SYNTH_ENVIRONMENT), code=dict(SYNTH_CODE),
                                  created_at=SYNTH_CREATED_AT, fixture_mode=True)
    assert run["run_kind"] == "unified_baseline" and sorted(run["members"]) == ["brain", "knee"]
    assert run["members"]["knee"]["run_sha256"] == run_sha256_v5(member_runs["knee"])
    run_dir = write_unified_run_v5(run, member_runs, registries, tmp_path, REPO)
    assert json.loads((run_dir / "run.json").read_text()) == run
    for pid in ("knee", "brain"):
        exposure = json.loads((tmp_path / "runs" / f"run-u.{pid}" / "exposure.json").read_text())
        assert exposure["run_id"] == f"run-u.{pid}" and exposure["entries"]
    with pytest.raises(PrivatePathError):
        write_unified_run_v5(run, member_runs, registries, tmp_path, REPO)


def test_a_member_run_of_another_recipe_is_refused():
    plan = make_unified_plan(SCOPES)
    member_runs, _, _ = make_unified_members(plan, SCOPES, "run-u")
    other = make_unified_plan(SCOPES, shared={"optimizer": {"lr": 0.5}})
    _raises("UNIFIED_MEMBER_MISMATCH", "$.member_runs.brain", register_unified_run_v5, run_id="run-u",
            run_kind="baseline", plan=other, plan_sha256=unified_plan_sha256_v5(other), member_runs=member_runs,
            environment=dict(SYNTH_ENVIRONMENT), code=dict(SYNTH_CODE), created_at=SYNTH_CREATED_AT, fixture_mode=True)


# ---------------------------------------------------------------------------
# Pilot of 2026-11-12: an optional hard-negative fraction and an optional stop step, both absent from earlier plans

ORIGINAL_PLAN = Path("configs/v5/proposals/unified_knee_brain.yaml")
PILOT_PLAN = Path("configs/v5/proposals/unified_knee_brain_negtopk_pilot.yaml")
ORIGINAL_PLAN_SHA256 = "2f43a97185292644b2ab2f1c91a8a9eeb7dba4dbec274200758bac910d3aa71c"


def _flat(node, prefix=""):
    if isinstance(node, dict):
        return {key: value for name, child in node.items() for key, value in _flat(child, f"{prefix}.{name}").items()}
    return {prefix.lstrip("."): node}


def test_the_plan_of_the_finished_run_keeps_its_hash():
    plan = yaml.safe_load(ORIGINAL_PLAN.read_text())
    assert unified_plan_sha256_v5(plan) == ORIGINAL_PLAN_SHA256
    assert "negative_topk_fraction" not in plan["members"]["knee_qdess"]["box_supervision"]
    assert "stop_at_step" not in plan["shared"]


def test_a_plan_may_keep_only_the_top_fraction_of_the_negatives():
    plan = make_unified_plan(SCOPES)
    plan["members"]["knee"]["box_supervision"]["negative_topk_fraction"] = 0.01
    assert validate_unified_plan_v5(plan, SCOPES, SOURCES) is plan
    for value in (0, 0.0, 1.5, -0.01, True, "0.01", None):
        plan["members"]["knee"]["box_supervision"]["negative_topk_fraction"] = value
        _raises("UNIFIED_PLAN_INVALID", "$.members.knee.box_supervision", validate_unified_plan_v5, plan, SCOPES,
                SOURCES)
    plan["members"]["knee"]["box_supervision"]["negative_topk_fraction"] = 1.0
    plan["members"]["knee"]["box_supervision"]["another_key"] = 1
    _raises("UNIFIED_PLAN_INVALID", "$.members.knee.box_supervision", validate_unified_plan_v5, plan, SCOPES, SOURCES)


def test_a_plan_may_stop_before_the_end_of_its_schedule_without_changing_a_member_recipe():
    plan = make_unified_plan(SCOPES)
    recipes = {pid: member_recipe_v5(plan, pid, kind) for pid in SCOPES for kind in ("baseline", "smoke")}
    plan["shared"]["stop_at_step"] = 2                              # the schedule (budget.max_steps) stays 4
    assert validate_unified_plan_v5(plan, SCOPES, SOURCES) is plan
    assert {pid: member_recipe_v5(plan, pid, kind) for pid in SCOPES for kind in ("baseline", "smoke")} == recipes
    assert plan["shared"]["budget"]["max_steps"] == 4
    for value in (0, 4, 5, -2, 2.0, True, "2", None):               # a stop at the end of the schedule is no stop
        plan["shared"]["stop_at_step"] = value
        _raises("UNIFIED_PLAN_INVALID", "$.shared.stop_at_step", validate_unified_plan_v5, plan, SCOPES, SOURCES)
    plan["shared"]["stop_at_step"] = 3
    plan["shared"]["another_key"] = 1
    _raises("UNIFIED_PLAN_INVALID", "$.shared", validate_unified_plan_v5, plan, SCOPES, SOURCES)


def test_the_pilot_plan_differs_from_the_finished_run_only_where_the_researcher_approved():
    """Approval of 2026-11-12: top 1 % of the negatives, stop at step 6000 of the 30000-step schedule, 16 h in all.
    Seeds, model, optimiser, schedule, splits (scope, source, lineage), sampling and every other value are equal."""
    original, pilot = (yaml.safe_load(path.read_text()) for path in (ORIGINAL_PLAN, PILOT_PLAN))
    before, after = _flat(original), _flat(pilot)
    changed = {key: (before.get(key), after.get(key)) for key in sorted(set(before) | set(after))
               if before.get(key) != after.get(key)}
    assert changed == {
        "plan_id": ("unified_knee_brain_v1", "unified_knee_brain_negtopk_pilot_v1"),
        "shared.stop_at_step": (None, 6000),
        "members.knee_qdess.box_supervision.negative_topk_fraction": (None, 0.01),
        "budget.max_wall_hours": (20.0, 16.0)}
    assert pilot["shared"]["budget"] == {"max_steps": 30000, "max_wall_hours": 16.0}
    assert pilot["shared"]["checkpoint_every"] == 2000              # checkpoints at 2000, 4000 and 6000
    box = pilot["members"]["knee_qdess"]["box_supervision"]
    assert (box["topk_fraction"], box["positive_weight"], box["negative_weight"]) == (0.1, 1.0, 1.0)
    scopes = {"knee_qdess": _scope("skm_joint_early"), "brain_dwi_adc": _scope("isles_u")}
    policies = load_source_policies_v5(Path("configs/v5/sources_v5.yaml"))
    sources = {"knee_qdess": source_binding_v5(policies["skm_tea_joint_early"]),
               "brain_dwi_adc": source_binding_v5(policies["isles_2022"])}
    assert validate_unified_plan_v5(pilot, scopes, sources) is pilot
    assert unified_plan_sha256_v5(pilot) != ORIGINAL_PLAN_SHA256
