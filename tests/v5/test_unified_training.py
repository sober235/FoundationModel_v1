"""Unified training loop: one batch per protocol each step, one optimizer step (design 2026-11-10 §5.3). CPU, fp32."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.engine import train_unified_v5 as train_unified_module
from anatobind.v41.engine.data_v5 import TrainingSampleV5
from anatobind.v41.engine.train_unified_v5 import train_unified_v5, unified_loss_v5
from anatobind.v41.engine.train_v5 import load_checkpoint_v5
from anatobind.v41.engine.unified_registration_v5 import (
    member_recipe_v5, register_unified_run_v5, unified_plan_sha256_v5, write_unified_run_v5)
from anatobind.v41.losses import box_mil_objective_v5, masked_objective_v5
from anatobind.v41.models.interface_v5 import ModelOutputV5
from anatobind.v41.models.unified_v5 import build_unified_model_v5
from tests.v5.factories import (
    SYNTH_BOX_SPEC, SYNTH_CODE, SYNTH_CREATED_AT, SYNTH_ENVIRONMENT, make_module_scope, make_scope,
    make_training_sample_v5, make_unified_members, make_unified_plan)

REPO = Path(__file__).resolve().parents[2]
SCOPES = {"knee": validate_scope_v5(make_scope()),
          "brain": validate_scope_v5(make_module_scope("U", input_modalities=["synthetic_dwi", "synthetic_adc"]))}
SHAPE = (4, 8, 8)
TRAIN_UIDS = ["s-0", "s-1"]


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _box_sample(scope, uid: str, seed: int, empty: bool):
    """A box-supervised sample: U targets are bag ids (even ids on purpose); `empty` is a scan without boxes."""
    base = make_training_sample_v5(scope, uid, shape=SHAPE, seed=seed)
    bags = np.zeros_like(base.targets["U"], dtype=np.int32) if empty else base.targets["U"].astype(np.int32) * 2
    return TrainingSampleV5(observation=base.observation, targets={**base.targets, "U": bags},
                            e_pos={**base.e_pos, "U": bags > 0}, e_neg={**base.e_neg, "U": bags == 0})


def _fixture(root: Path, run_id: str = "run-u"):
    plan = make_unified_plan(SCOPES)
    member_runs, registries, _ = make_unified_members(plan, SCOPES, run_id)
    run = register_unified_run_v5(run_id=run_id, run_kind="baseline", plan=plan,
                                  plan_sha256=unified_plan_sha256_v5(plan), member_runs=member_runs,
                                  environment=dict(SYNTH_ENVIRONMENT), code=dict(SYNTH_CODE),
                                  created_at=SYNTH_CREATED_AT, fixture_mode=True)
    run_dir = write_unified_run_v5(run, member_runs, registries, root, REPO)
    members = {}
    for pid, scope in SCOPES.items():
        if pid == "knee":
            samples = {uid: _box_sample(scope, uid, i, empty=i == 1) for i, uid in enumerate(TRAIN_UIDS)}
        else:
            samples = {uid: make_training_sample_v5(scope, uid, shape=SHAPE, seed=10 + i)
                       for i, uid in enumerate(TRAIN_UIDS)}
        members[pid] = {"run": member_runs[pid], "samples": samples, "recipe": member_recipe_v5(plan, pid, "baseline"),
                        "supervision": plan["members"][pid]["supervision"],
                        "box_supervision": plan["members"][pid]["box_supervision"], "loss_weight": 1.0}
    return SimpleNamespace(plan=plan, run=run, run_dir=run_dir, root=root, members=members,
                           model=lambda: build_unified_model_v5(plan, SCOPES))


def _train(fixture, resume=None, **kwargs):
    return train_unified_v5(fixture.run, fixture.model(), fixture.members, fixture.plan, fixture.run_dir,
                            fixture.root, REPO, resume, **kwargs)


def test_training_completes_and_logs_every_member(tmp_path):
    fixture = _fixture(tmp_path)
    result = _train(fixture)
    assert (result["status"], result["step"], result["failure_code"]) == ("COMPLETED", 4, None)
    lines = [json.loads(x) for x in (fixture.run_dir / "metrics.jsonl").read_text().splitlines()]
    steps = [x for x in lines if x["event"] == "step"]
    assert [x["step"] for x in steps] == [1, 2, 3, 4]
    assert set(steps[0]["members"]) == {"brain", "knee"} and set(steps[0]["members"]["knee"]["tasks"]) == {"A", "U"}
    assert sorted(p.name for p in (fixture.run_dir / "checkpoints").glob("*.pt")) == [
        "step_00000002.pt", "step_00000004.pt"]


def test_pause_and_resume_equals_the_continuous_run_bitwise_on_cpu(tmp_path):
    first, second = _fixture(tmp_path / "a"), _fixture(tmp_path / "b")
    full = _train(first)
    paused = _train(second, pause_at=2)
    assert (paused["status"], paused["step"]) == ("PAUSED", 2)
    resumed = _train(second, (paused["last_checkpoint"], paused["last_checkpoint_sha256"]))
    assert resumed["status"] == "COMPLETED"
    a = load_checkpoint_v5(full["last_checkpoint"], first.run, full["last_checkpoint_sha256"], "cpu")
    b = load_checkpoint_v5(resumed["last_checkpoint"], second.run, resumed["last_checkpoint_sha256"], "cpu")
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["sampler"] == b["sampler"] and set(a["sampler"]) == {"brain", "knee"}
    assert a["data_sha256"] == b["data_sha256"]


def test_binding_and_init_are_checked_before_anything_is_written(tmp_path):
    fixture = _fixture(tmp_path)
    members = dict(fixture.members, knee=dict(fixture.members["knee"]))
    members["knee"]["samples"] = dict(members["knee"]["samples"],
                                      other=_box_sample(SCOPES["knee"], "other", 5, empty=False))
    _raises("SAMPLE_NOT_REGISTERED", "$.members.knee.samples", train_unified_v5, fixture.run, fixture.model(),
            members, fixture.plan, fixture.run_dir, fixture.root, REPO, None)
    model = fixture.model()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    _raises("MODEL_INIT_MISMATCH", "$.model", train_unified_v5, fixture.run, model, fixture.members, fixture.plan,
            fixture.run_dir, fixture.root, REPO, None)
    other = copy.deepcopy(fixture.run)
    other["created_at"] = "2026-01-01T00:00:00Z"
    _raises("RUN_NOT_REGISTERED", "$.run_dir", train_unified_v5, other, fixture.model(), fixture.members,
            fixture.plan, fixture.run_dir, fixture.root, REPO, None)
    assert not (fixture.run_dir / "data.json").exists() and not (fixture.run_dir / "checkpoints").exists()


def test_a_non_finite_loss_stops_the_run_with_a_failure_record(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(train_unified_module, "box_mil_objective_v5",
                        lambda *a, **k: torch.tensor(float("nan"), requires_grad=True))
    result = _train(fixture)
    assert (result["status"], result["step"], result["failure_code"]) == ("FAILED", 0, "NON_FINITE_LOSS")
    assert json.loads((fixture.run_dir / "failure.json").read_text())["code"] == "NON_FINITE_LOSS"
    _raises("RUN_ALREADY_STARTED", "$.run_dir", _train, fixture)


def test_unified_loss_uses_the_box_objective_only_where_box_supervision_is_given():
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 2, 2, 2)
    bags = torch.zeros(1, 1, 2, 2, 2, dtype=torch.int32)
    bags[0, 0, 0, 0, 0] = 2
    eligibility = {"U": {"e_pos": bags > 0, "e_neg": bags == 0}}
    supervision = {"tasks": {"U": {"kind": "sigmoid", "ce_weight": 1.0, "dice_weight": 1.0, "task_weight": 3.0}}}
    output = ModelOutputV5(anatomy_logits=None, abnormality_logits=logits)
    boxed, parts = unified_loss_v5(output, {"U": bags}, eligibility, supervision, SYNTH_BOX_SPEC)
    expected = box_mil_objective_v5(logits, bags, bags > 0, bags == 0, SYNTH_BOX_SPEC)
    assert float(boxed) == pytest.approx(3.0 * float(expected))
    assert parts["U"]["support"] == 1 + 7                                  # one bag, seven negative voxels
    plain, _ = unified_loss_v5(output, {"U": bags > 0}, eligibility, supervision, None)
    expected = masked_objective_v5(logits, bags > 0, bags > 0, bags == 0, supervision["tasks"]["U"])
    assert float(plain) == pytest.approx(3.0 * float(expected))
