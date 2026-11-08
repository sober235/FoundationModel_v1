"""v5.1 resumable training core (plan_v5 T09b; Proposal §4.3, §6.3). CPU only, fp32."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import stat
from pathlib import Path

import pytest
import torch

from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.engine import train_v5 as train_module
from anatobind.v41.engine.registration_v5 import run_sha256_v5, write_run_v5
from anatobind.v41.engine.train_v5 import (
    build_optimizer_v5, build_scaler_v5, build_scheduler_v5, load_checkpoint_v5, save_checkpoint_v5, train_v5,
    training_loss_v5)
from anatobind.v41.losses import masked_objective_v5
from anatobind.v41.models.interface_v5 import ModelOutputV5
from anatobind.v41.paths import PrivatePathError, write_private_json
from tests.v5.factories import (
    TinyPerceptionModel, make_module_scope, make_recipe, make_run, make_split_records, make_split_registry,
    make_supervision, make_training_sample_v5)

REPO = Path(__file__).resolve().parents[2]
SPLITS = ["train", "train", "train", "development_val", "locked_test"]
TRAIN_UIDS = ["s-0", "s-1", "s-2"]
STATE_KEYS = {"step", "model", "optimizer", "scheduler", "scaler", "sampler", "rng", "wall_seconds", "data_sha256"}


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _scope(task="A", **overrides):
    return validate_scope_v5(make_module_scope(task, **overrides))


def _au_scope():
    return _scope("A", scope_id="synthetic_au", abnormality_classes=["target_a"])


def _tasks(scope):
    return [task for task, classes in (("A", scope.anatomy_classes), ("U", scope.abnormality_classes)) if classes]


def _setup(tmp_path, name="private", scope=None, **recipe_overrides):
    """A registered fixture run on disk (run.json + exposure.json) with in-memory training samples."""
    scope = scope or _scope("A")
    values = {"model": {"tasks": _tasks(scope)}, "patch_size": [4, 4, 4], "batch_size": 2, "checkpoint_every": 3,
              "budget": {"max_steps": 6, "max_wall_hours": 1.0}}
    values.update(recipe_overrides)
    recipe = make_recipe(**values)
    records = make_split_records(SPLITS)
    registry = make_split_registry(records)
    supervision = make_supervision(scope)
    run = make_run(scope=scope, records=records, split_registry=registry, recipe=recipe, supervision=supervision)
    root = tmp_path / name
    run_dir = write_run_v5(run, registry, root, REPO)
    samples = {uid: make_training_sample_v5(scope, uid, shape=(6, 6, 6), seed=i) for i, uid in enumerate(TRAIN_UIDS)}
    return {"run": run, "recipe": recipe, "supervision": supervision, "scope": scope, "samples": samples,
            "root": root, "run_dir": run_dir}


def _model(scope, seed=0, cls=TinyPerceptionModel, tasks=None, **kwargs):
    torch.manual_seed(seed)
    return cls(scope, tuple(tasks or _tasks(scope)), **kwargs)


def _train(ctx, model, resume=None, **replace):
    args = {"run": ctx["run"], "model": model, "samples": ctx["samples"], "recipe": ctx["recipe"],
            "supervision": ctx["supervision"], "run_dir": ctx["run_dir"], "private_root": ctx["root"],
            "repo": REPO, "resume": resume}
    args.update(replace)
    return train_v5(**args)


def _metrics(ctx):
    return [json.loads(line) for line in (ctx["run_dir"] / "metrics.jsonl").read_text().splitlines()]


def _assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    else:
        assert a == b


def _au_batch(seed=0):
    torch.manual_seed(seed)
    a_logits = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
    u_logits = torch.randn(2, 1, 4, 4, 4, requires_grad=True)
    targets = {"A": torch.randint(0, 3, (2, 4, 4, 4)), "U": torch.rand(2, 1, 4, 4, 4) < 0.3}
    return a_logits, u_logits, targets


def test_loss_has_no_gradient_outside_eligibility():
    scope = _au_scope()
    supervision = make_supervision(scope)
    a_logits, u_logits, targets = _au_batch()
    inside = torch.zeros(2, 4, 4, 4, dtype=torch.bool)
    inside[:, :2] = True
    a_e = inside.unsqueeze(1).expand(2, 2, 4, 4, 4).clone()
    eligibility = {"A": {"e_pos": a_e, "e_neg": a_e.clone()},
                   "U": {"e_pos": inside.unsqueeze(1).clone(), "e_neg": inside.unsqueeze(1).clone()}}
    # A is a softmax over all classes: one ineligible class removes the whole voxel from A's loss.
    eligibility["A"]["e_neg"][:, 1, 0, 0, 0] = False
    a_used = inside.clone()
    a_used[:, 0, 0, 0] = False
    loss, parts = training_loss_v5(ModelOutputV5(a_logits, u_logits), targets, eligibility, supervision)
    loss.backward()
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    a_grad = a_logits.grad.abs().sum(dim=1)
    assert (a_grad[~a_used] == 0).all() and (a_grad[a_used] > 0).all()
    u_grad = u_logits.grad[:, 0]
    assert (u_grad[~inside] == 0).all() and (u_grad[inside] != 0).all()
    assert parts["A"]["support"] == int(a_used.sum()) and parts["U"]["support"] == int(inside.sum())
    assert set(parts) == {"A", "U"} and all(isinstance(p["loss"], float) for p in parts.values())


def test_task_without_support_does_not_change_other_task():
    scope = _au_scope()
    supervision = make_supervision(scope, tasks={"A": {"task_weight": 2.0}})
    a_logits, u_logits, targets = _au_batch()
    a_e = torch.ones(2, 2, 4, 4, 4, dtype=torch.bool)
    none = torch.zeros(2, 1, 4, 4, 4, dtype=torch.bool)
    eligibility = {"A": {"e_pos": a_e, "e_neg": a_e}, "U": {"e_pos": none, "e_neg": none}}
    loss, parts = training_loss_v5(ModelOutputV5(a_logits, u_logits), targets, eligibility, supervision)
    expected = masked_objective_v5(a_logits, targets["A"], a_e, a_e, supervision["tasks"]["A"]) * 2.0
    assert torch.equal(loss, expected)      # Σ task_weight·L_t: no averaging over tasks that have support
    assert parts["U"] == {"loss": 0.0, "support": 0}
    loss.backward()
    assert u_logits.grad is not None and (u_logits.grad == 0).all()   # a differentiable, finite zero


def test_non_finite_loss_is_refused(monkeypatch):
    scope = _scope("A")
    a_logits, _, targets = _au_batch()
    a_e = torch.ones(2, 2, 4, 4, 4, dtype=torch.bool)
    monkeypatch.setattr(train_module, "masked_objective_v5", lambda *args: torch.tensor(math.nan, requires_grad=True))
    _raises("NON_FINITE_LOSS", "$.tasks.A", training_loss_v5, ModelOutputV5(a_logits, None), {"A": targets["A"]},
            {"A": {"e_pos": a_e, "e_neg": a_e}}, make_supervision(scope))


def test_optimizer_scheduler_and_scaler_follow_recipe():
    scope = _scope("A")
    recipe = make_recipe(optimizer={"lr": 0.01, "weight_decay": 0.5}, scheduler={"exponent": 2.0},
                         budget={"max_steps": 4})
    model = _model(scope)
    optimizer = build_optimizer_v5(model, recipe)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.01 and optimizer.param_groups[0]["weight_decay"] == 0.5
    scheduler = build_scheduler_v5(optimizer, recipe)
    rates = []
    for _ in range(3):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates == pytest.approx([0.01 * (1 - k / 4) ** 2.0 for k in range(3)])
    assert build_scaler_v5(recipe, "cpu").is_enabled() is False
    assert build_scaler_v5(make_recipe(precision="amp_bf16"), "cpu").is_enabled() is False
    assert build_scaler_v5(make_recipe(precision="amp_fp16"), "cpu").is_enabled() is True


def test_resume_is_bitwise_identical_on_cpu(tmp_path):
    first = _setup(tmp_path, "first")
    model = _model(first["scope"], seed=0)
    result = _train(first, model)
    assert result["status"] == "COMPLETED" and result["step"] == 6 and result["failure_code"] is None
    middle = first["run_dir"] / "checkpoints" / "step_00000003.pt"
    middle_sha = json.loads(middle.with_name(middle.name + ".json").read_text())["sha256"]
    # The same registered run under a second private root, a model with other weights, resumed at step 3.
    second = _setup(tmp_path, "second")
    assert second["run"] == first["run"]
    resumed = _model(second["scope"], seed=99)
    again = _train(second, resumed, resume=(middle, middle_sha))
    assert again["status"] == "COMPLETED" and again["step"] == 6 and resumed.calls == 3
    _assert_same(dict(model.state_dict()), dict(resumed.state_dict()))
    final_a = load_checkpoint_v5(result["last_checkpoint"], first["run"], result["last_checkpoint_sha256"], "cpu")
    final_b = load_checkpoint_v5(again["last_checkpoint"], second["run"], again["last_checkpoint_sha256"], "cpu")
    for key in STATE_KEYS - {"wall_seconds"}:
        _assert_same(final_a[key], final_b[key])
    # The same patches were drawn: steps 4..6 record bit-identical losses and learning rates.
    steps_a = _metrics(first)
    lines_b = _metrics(second)
    assert lines_b[0] == {"event": "resume", "step": 3, "checkpoint_sha256": middle_sha}
    assert lines_b[1:] == steps_a[3:] and [line["step"] for line in steps_a] == [1, 2, 3, 4, 5, 6]


def test_checkpoint_contains_all_states(tmp_path):
    ctx = _setup(tmp_path)
    model = _model(ctx["scope"])
    result = _train(ctx, model)
    path = result["last_checkpoint"]
    assert path == ctx["run_dir"] / "checkpoints" / "step_00000006.pt"
    assert result["last_checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert json.loads(path.with_name(path.name + ".json").read_text()) == {
        "sha256": result["last_checkpoint_sha256"], "step": 6, "run_sha256": run_sha256_v5(ctx["run"])}
    assert sorted(p.name for p in path.parent.glob("*.pt")) == ["step_00000003.pt", "step_00000006.pt"]
    state = load_checkpoint_v5(path, ctx["run"], result["last_checkpoint_sha256"], "cpu")
    assert set(state) == STATE_KEYS | {"format", "run_sha256"}
    assert state["format"] == "anatobind_ckpt_v5" and state["run_sha256"] == run_sha256_v5(ctx["run"])
    assert state["step"] == 6 and set(state["rng"]) == {"python", "torch", "cuda"}
    _assert_same(state["model"], dict(model.state_dict()))
    assert state["optimizer"]["state"] and state["scheduler"]["last_epoch"] == 6
    assert state["sampler"]["bit_generator"]["bit_generator"] == "PCG64" and state["scaler"] == {}
    assert isinstance(state["wall_seconds"], float) and state["wall_seconds"] >= 0
    lines = _metrics(ctx)
    assert [line["step"] for line in lines] == [1, 2, 3, 4, 5, 6]
    assert set(lines[0]) == {"event", "step", "loss", "lr", "tasks"} and set(lines[0]["tasks"]) == {"A"}
    assert stat.S_IMODE((ctx["run_dir"] / "metrics.jsonl").stat().st_mode) == 0o600


def test_resume_refuses_hash_or_run_mismatch(tmp_path):
    ctx = _setup(tmp_path)
    result = _train(ctx, _model(ctx["scope"]))
    path, digest = result["last_checkpoint"], result["last_checkpoint_sha256"]
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    tampered = tmp_path / "tampered.pt"
    tampered.write_bytes(bytes(data))
    _raises("CHECKPOINT_HASH_MISMATCH", "$.checkpoint", load_checkpoint_v5, tampered, ctx["run"], digest, "cpu")
    fresh = _setup(tmp_path, "fresh")
    model = _model(fresh["scope"])
    _raises("CHECKPOINT_HASH_MISMATCH", "$.checkpoint", _train, fresh, model, resume=(tampered, digest))
    # One recipe value changed makes a different run; its checkpoints are not this run's.
    other = _setup(tmp_path, "other", checkpoint_every=2)
    assert other["run"]["recipe_sha256"] != ctx["run"]["recipe_sha256"]
    _raises("RUN_MISMATCH", "$.run_sha256", load_checkpoint_v5, path, other["run"], digest, "cpu")
    _raises("RUN_MISMATCH", "$.run_sha256", _train, other, model, resume=(path, digest))
    _raises("INVALID_SHA256", "$.expected_sha256", load_checkpoint_v5, path, ctx["run"], "x" * 64, "cpu")
    assert model.calls == 0
    assert not (fresh["run_dir"] / "metrics.jsonl").exists() and not (other["run_dir"] / "metrics.jsonl").exists()


def test_resume_refuses_changed_data(tmp_path):
    """Review Focus 5 for data: the same uids with one changed target voxel (a re-migrated record or an edited
    label map) cannot resume a run trained on the old content."""
    ctx = _setup(tmp_path)
    result = _train(ctx, _model(ctx["scope"]))
    middle = ctx["run_dir"] / "checkpoints" / "step_00000003.pt"
    middle_sha = json.loads(middle.with_name(middle.name + ".json").read_text())["sha256"]
    second = _setup(tmp_path, "second")
    sample = second["samples"]["s-1"]
    targets = {task: value.copy() for task, value in sample.targets.items()}
    targets["A"][0, 0, 0] = (targets["A"][0, 0, 0] + 1) % 3
    changed = dict(second["samples"], **{"s-1": dataclasses.replace(sample, targets=targets)})
    model = _model(second["scope"])
    _raises("DATA_MISMATCH", "$.samples", _train, second, model, samples=changed, resume=(middle, middle_sha))
    assert model.calls == 0 and not (second["run_dir"] / "metrics.jsonl").exists()
    assert _train(second, model, resume=(middle, middle_sha))["status"] == "COMPLETED"
    # The fresh start recorded the digest once (private, never overwritten); every checkpoint carries it.
    state = load_checkpoint_v5(result["last_checkpoint"], ctx["run"], result["last_checkpoint_sha256"], "cpu")
    record = ctx["run_dir"] / "data.json"
    assert json.loads(record.read_text()) == {"data_sha256": state["data_sha256"]}
    assert stat.S_IMODE(record.stat().st_mode) == 0o600 and len(state["data_sha256"]) == 64


def test_checkpoints_never_overwrite(tmp_path):
    ctx = _setup(tmp_path)
    result = _train(ctx, _model(ctx["scope"]))
    path = result["last_checkpoint"]
    before = path.read_bytes()
    state = load_checkpoint_v5(path, ctx["run"], result["last_checkpoint_sha256"], "cpu")
    body = {key: state[key] for key in STATE_KEYS}
    with pytest.raises(PrivatePathError) as err:
        save_checkpoint_v5(path, body, ctx["run"], ctx["root"], REPO)
    assert err.value.code == "PRIVATE_FILE_EXISTS" and path.read_bytes() == before
    assert not [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")]
    with pytest.raises(PrivatePathError) as err:
        save_checkpoint_v5(REPO / "step.pt", body, ctx["run"], ctx["root"], REPO)
    assert err.value.code == "PRIVATE_PATH_IN_GIT"
    _raises("INVALID_CHECKPOINT_STATE", "$.state", save_checkpoint_v5, path.with_name("x.pt"),
            {k: v for k, v in body.items() if k != "rng"}, ctx["run"], ctx["root"], REPO)
    # A second fresh start in a run directory that already trained is refused before any step,
    # and so is resuming from an older checkpoint while a newer one exists.
    again = _model(ctx["scope"])
    _raises("RUN_ALREADY_STARTED", "$.run_dir", _train, ctx, again)
    middle = path.with_name("step_00000003.pt")
    middle_sha = json.loads(middle.with_name(middle.name + ".json").read_text())["sha256"]
    _raises("CHECKPOINT_NOT_LATEST", "$.resume", _train, ctx, again, resume=(middle, middle_sha))
    assert again.calls == 0 and path.read_bytes() == before


class _FailingModel(TinyPerceptionModel):
    """Behaves like TinyPerceptionModel until forward call `fail_on_call`, which returns inf or raises OOM."""

    def __init__(self, scope, tasks, fail_on_call, failure):
        super().__init__(scope, tasks)
        self.fail_on_call, self.failure = fail_on_call, failure

    def _forward_v5(self, batch):
        output = super()._forward_v5(batch)
        if self.calls < self.fail_on_call:
            return output
        if self.failure == "oom":
            raise torch.cuda.OutOfMemoryError("synthetic")
        return ModelOutputV5(anatomy_logits=output.anatomy_logits + math.inf, abnormality_logits=None)


@pytest.mark.parametrize("failure, code", [("inf", "NON_FINITE_LOGITS"), ("oom", "OUT_OF_MEMORY")])
def test_nonfinite_or_oom_writes_failure_without_retry(tmp_path, failure, code):
    ctx = _setup(tmp_path)
    model = _model(ctx["scope"], cls=_FailingModel, fail_on_call=5, failure=failure)
    result = _train(ctx, model)
    kept = ctx["run_dir"] / "checkpoints" / "step_00000003.pt"
    assert result == {"status": "FAILED", "step": 4, "last_checkpoint": kept,
                      "last_checkpoint_sha256": hashlib.sha256(kept.read_bytes()).hexdigest(), "failure_code": code}
    assert model.calls == 5     # the failing forward ran once; nothing was retried
    record = json.loads((ctx["run_dir"] / "failure.json").read_text())
    assert set(record) == {"code", "step", "recorded_at"} and (record["code"], record["step"]) == (code, 4)
    assert sorted(p.name for p in kept.parent.glob("*.pt")) == ["step_00000003.pt"]
    assert [line["step"] for line in _metrics(ctx)] == [1, 2, 3, 4]
    # A failed run is final; a new attempt is a new run.
    _raises("RUN_FAILED", "$.run_dir", _train, ctx, _model(ctx["scope"]),
            resume=(kept, result["last_checkpoint_sha256"]))


def test_wall_budget_is_cumulative_and_fails_without_retry(tmp_path, monkeypatch):
    ctx = _setup(tmp_path)
    done = _train(ctx, _model(ctx["scope"]))
    state = load_checkpoint_v5(done["last_checkpoint"].with_name("step_00000003.pt"), ctx["run"],
                               json.loads(done["last_checkpoint"].with_name("step_00000003.pt.json").read_text())
                               ["sha256"], "cpu")
    # A checkpoint whose recorded wall time already exhausts the budget cannot buy a fresh hour by resuming.
    spent = _setup(tmp_path, "spent")
    body = dict({key: state[key] for key in STATE_KEYS}, wall_seconds=3600.5)
    path = spent["root"] / "carried" / "step_00000003.pt"
    digest = save_checkpoint_v5(path, body, spent["run"], spent["root"], REPO)
    model = _model(spent["scope"])
    result = _train(spent, model, resume=(path, digest))
    assert (result["status"], result["step"], result["failure_code"]) == ("FAILED", 3, "WALL_BUDGET_EXCEEDED")
    assert model.calls == 0 and result["last_checkpoint"] == path
    # Within one call: every clock read advances half an hour, so the 1 h budget runs out mid-run.
    ticks = iter(range(0, 10 ** 6, 1800))
    monkeypatch.setattr(train_module, "_monotonic", lambda: float(next(ticks)))
    clocked = _setup(tmp_path, "clocked")
    model = _model(clocked["scope"])
    result = _train(clocked, model)
    assert result["status"] == "FAILED" and result["failure_code"] == "WALL_BUDGET_EXCEEDED"
    assert 0 < result["step"] < 6 and model.calls == result["step"]
    record = json.loads((clocked["run_dir"] / "failure.json").read_text())
    assert record["code"] == "WALL_BUDGET_EXCEEDED" and record["step"] == result["step"]


def test_training_requires_registered_run_and_exposure(tmp_path):
    ctx = _setup(tmp_path)
    model = _model(ctx["scope"])
    run = ctx["run"]
    _raises("RUN_NOT_REGISTERED", "$.run_dir", _train, ctx, model, run_dir=ctx["root"] / "runs" / "never-written")
    # run.json without its exposure file: no recorded exposure, no training (L6).
    half = tmp_path / "half"
    write_private_json(half / "runs" / run["run_id"] / "run.json", run)
    _raises("RUN_NOT_REGISTERED", "$.run_dir", _train, ctx, model, run_dir=half / "runs" / run["run_id"],
            private_root=half)
    # A directory registered for another run.
    other = _setup(tmp_path, "other", checkpoint_every=2)
    _raises("RUN_NOT_REGISTERED", "$.run_dir", _train, ctx, model, run_dir=other["run_dir"],
            private_root=other["root"])
    # Recipe, supervision, model and samples must be the registered ones.
    _raises("RUN_MISMATCH", "$.recipe", _train, ctx, model, recipe=other["recipe"])
    _raises("RUN_MISMATCH", "$.supervision", _train, ctx, model,
            supervision=make_supervision(ctx["scope"], tasks={"A": {"task_weight": 2.0}}))
    _raises("RUN_MISMATCH", "$.model", _train, ctx, _model(_au_scope()))
    au = _setup(tmp_path, "au", scope=_au_scope())
    _raises("RUN_MISMATCH", "$.model", _train, au, _model(au["scope"], tasks=["A"]))
    locked = make_training_sample_v5(ctx["scope"], "s-4", shape=(6, 6, 6))    # s-4 is locked_test
    _raises("SAMPLE_NOT_REGISTERED", "$.samples", _train, ctx, model, samples=dict(ctx["samples"], **{"s-4": locked}))
    fewer = {uid: sample for uid, sample in ctx["samples"].items() if uid != "s-2"}
    _raises("SAMPLE_NOT_REGISTERED", "$.samples", _train, ctx, model, samples=fewer)
    swapped = dict(ctx["samples"], **{"s-2": ctx["samples"]["s-1"]})
    _raises("SAMPLE_NOT_REGISTERED", "$.samples", _train, ctx, model, samples=swapped)
    assert model.calls == 0 and not (ctx["run_dir"] / "metrics.jsonl").exists()
    assert copy.deepcopy(run) == ctx["run"]
