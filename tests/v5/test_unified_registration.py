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
