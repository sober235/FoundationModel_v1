"""Unified training loop (Proposal_v5 §0.6; design 2026-11-10 §5.3): one batch per protocol each step.

Every step draws one batch from each member (protocols in sorted order), runs that member's view, scales its loss
by the member's loss_weight and accumulates the gradients; the optimizer, scaler and scheduler then step once.
All members therefore take the same number of steps whatever their sample counts.

Rules are train_v5's, per member where they concern data: the unified run must be registered on disk, each member
run must be the one the unified run binds, each member's samples must be exactly the samples its registered
exposure marks `training`, a fresh start must be build_unified_model_v5's seeded initialisation bit for bit, and a
checkpoint (train_v5's format, bound to the unified run) holds every state a bitwise CPU resume needs. A non-finite
logit or loss, an OOM or an exhausted wall budget writes failure.json and stops without retrying.

A member with `box_supervision` trains U with box_mil_objective_v5 (its U target is the bag map of
box_data_v5); every other task uses masked_objective_v5 as in train_v5.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..contracts_v5 import V5Error, canonical_sha256
from ..losses import box_mil_objective_v5, masked_objective_v5
from ..models.interface_v5 import ModelOutputV5
from ..models.unified_v5 import UnifiedModelV5, build_unified_model_v5
from ..paths import resolve_private_output, write_private_json
from .box_data_v5 import box_positive_centers_v5
from .data_v5 import PatchSamplerV5, TrainingSampleV5, make_batch_v5, positive_centers_v5, training_data_sha256_v5
from .registration_v5 import run_sha256_v5, validate_recipe_v5
from .train_v5 import (
    STOP_CODES, _append_jsonl, _checkpoint_steps, _rng_state, _set_rng_state, _support, _to_device, autocast_v5,
    build_optimizer_v5, build_scaler_v5, build_scheduler_v5, load_checkpoint_v5, save_checkpoint_v5)

_SCHEDULE_KEYS = ("budget", "checkpoint_every", "precision", "optimizer", "scheduler")
_monotonic = time.monotonic     # read through the module so the wall budget can be driven in tests


def _bag_count(bags: torch.Tensor, e_pos: torch.Tensor) -> int:
    count = 0
    for b in range(bags.shape[0]):
        for c in range(bags.shape[1]):
            count += int(torch.unique(bags[b, c][e_pos[b, c] & (bags[b, c] > 0)]).numel())
    return count


def unified_loss_v5(output: ModelOutputV5, targets: dict, eligibility: dict, supervision: dict,
                    box_supervision: dict | None) -> tuple[torch.Tensor, dict]:
    """Σ task_weight·L_t of one member; with box_supervision the U loss is the box objective on the bag map."""
    if not isinstance(output, ModelOutputV5):
        raise V5Error("INVALID_TYPE", "$.output")
    tasks = supervision.get("tasks") if isinstance(supervision, dict) else None
    if not isinstance(tasks, dict) or not tasks:
        raise V5Error("INVALID_TYPE", "$.supervision.tasks")
    total, parts = None, {}
    for task in sorted(tasks):
        logits = output.anatomy_logits if task == "A" else output.abnormality_logits
        if logits is None:
            raise V5Error("MODULE_TASKS_INVALID", f"$.output.{task}")
        if task not in targets or not isinstance(eligibility.get(task), dict):
            raise V5Error("TARGET_MISSING", f"$.targets.{task}")
        e_pos = eligibility[task]["e_pos"].to(torch.bool)
        e_neg = eligibility[task]["e_neg"].to(torch.bool)
        if task == "U" and box_supervision is not None:
            loss = box_mil_objective_v5(logits.float(), targets[task], e_pos, e_neg, box_supervision)
            support = _bag_count(targets[task], e_pos) + int(e_neg.sum())
        else:
            loss = masked_objective_v5(logits.float(), targets[task], e_pos, e_neg, tasks[task])
            support = _support(task, targets[task], e_pos, e_neg)
        if not bool(torch.isfinite(loss)):
            raise V5Error("NON_FINITE_LOSS", f"$.tasks.{task}")
        weighted = tasks[task]["task_weight"] * loss
        total = weighted if total is None else total + weighted
        parts[task] = {"loss": float(loss.detach()), "support": support}
    return total, parts


def _check_members(unified_run: dict, model: UnifiedModelV5, members: dict, private_root: Path, repo: Path) -> list:
    if not isinstance(members, dict) or set(members) != set(unified_run["members"]) \
            or tuple(sorted(members)) != model.protocol_ids:
        raise V5Error("UNIFIED_MEMBER_MISMATCH", "$.members")
    pids = sorted(members)
    for pid in pids:
        member, path = members[pid], f"$.members.{pid}"
        run, recipe, samples = member["run"], member["recipe"], member["samples"]
        validate_recipe_v5(recipe)
        view = model.view(pid)
        if run_sha256_v5(run) != unified_run["members"][pid]["run_sha256"] \
                or canonical_sha256(recipe) != run["recipe_sha256"] \
                or canonical_sha256(member["supervision"]) != run["supervision_sha256"] \
                or {"scope_id": view.scope.scope_id, "scope_hash": view.scope.scope_hash} != run["scope"] \
                or sorted(view.tasks) != sorted(recipe["model"]["tasks"]):
            raise V5Error("RUN_MISMATCH", path)
        exposure_path = resolve_private_output(Path(private_root) / "runs" / run["run_id"] / "exposure.json",
                                               private_root, repo)
        try:
            exposure = json.loads(exposure_path.read_text())
        except (OSError, ValueError):
            raise V5Error("RUN_NOT_REGISTERED", path) from None
        if not isinstance(exposure, dict) or exposure.get("run_sha256") != run_sha256_v5(run) \
                or not isinstance(exposure.get("entries"), dict):
            raise V5Error("RUN_NOT_REGISTERED", path)
        trained = {uid for uid, entries in exposure["entries"].items()
                   if isinstance(entries, list) and any(isinstance(e, dict) and e.get("role") == "training"
                                                        for e in entries)}
        if not isinstance(samples, dict) or set(samples) != trained \
                or any(not isinstance(sample, TrainingSampleV5) or sample.observation.sample_uid != uid
                       for uid, sample in samples.items()):
            raise V5Error("SAMPLE_NOT_REGISTERED", f"{path}.samples")
    first = members[pids[0]]["recipe"]
    if any(members[pid]["recipe"][key] != first[key] for pid in pids for key in _SCHEDULE_KEYS):
        raise V5Error("UNIFIED_MEMBER_MISMATCH", "$.members")
    return pids


def _check_init(model: UnifiedModelV5, plan: dict) -> None:
    expected = build_unified_model_v5(plan, {pid: model.view(pid).scope for pid in model.protocol_ids})
    actual, reference = model.state_dict(), expected.state_dict()
    if set(actual) != set(reference) \
            or any(not torch.equal(actual[key].detach().cpu(), reference[key]) for key in reference):
        raise V5Error("MODEL_INIT_MISMATCH", "$.model")


def train_unified_v5(unified_run: dict, model: UnifiedModelV5, members: dict, plan: dict, run_dir: Path,
                     private_root: Path, repo: Path, resume: tuple | None, *, pause_at: int | None = None) -> dict:
    """Train a registered unified run to its step budget; the result has train_v5's shape and statuses."""
    if not isinstance(model, UnifiedModelV5):
        raise V5Error("INVALID_TYPE", "$.model")
    run_dir = Path(run_dir)
    run_path = resolve_private_output(run_dir / "run.json", private_root, repo)
    try:
        registered = json.loads(run_path.read_text())
    except (OSError, ValueError):
        raise V5Error("RUN_NOT_REGISTERED", "$.run_dir") from None
    if registered != unified_run:
        raise V5Error("RUN_NOT_REGISTERED", "$.run_dir")
    pids = _check_members(unified_run, model, members, Path(private_root), repo)
    recipe = members[pids[0]]["recipe"]                 # schedule values are equal across members (checked)
    max_steps, every = recipe["budget"]["max_steps"], recipe["checkpoint_every"]
    if pause_at is not None and not (isinstance(pause_at, int) and not isinstance(pause_at, bool)
                                     and 0 < pause_at < max_steps and pause_at % every == 0):
        raise V5Error("INVALID_PAUSE", "$.pause_at")
    if resume is None:
        _check_init(model, plan)
    data_sha256 = canonical_sha256({pid: training_data_sha256_v5(members[pid]["samples"]) for pid in pids})
    metrics_path = resolve_private_output(run_dir / "metrics.jsonl", private_root, repo)
    failure_path = resolve_private_output(run_dir / "failure.json", private_root, repo)
    checkpoint_dir = resolve_private_output(run_dir / "checkpoints", private_root, repo)
    data_path = resolve_private_output(run_dir / "data.json", private_root, repo)
    if resume is None and any(path.exists() for path in (metrics_path, failure_path, checkpoint_dir, data_path)):
        raise V5Error("RUN_ALREADY_STARTED", "$.run_dir")
    if resume is not None and failure_path.exists():
        raise V5Error("RUN_FAILED", "$.run_dir")

    device = next(model.parameters()).device
    samplers = {}
    for index, pid in enumerate(pids):
        member = members[pid]
        centers = box_positive_centers_v5 if member["box_supervision"] is not None else positive_centers_v5
        samplers[pid] = PatchSamplerV5(sorted(member["samples"]),
                                       {uid: centers(sample) for uid, sample in member["samples"].items()},
                                       member["recipe"]["sampling"]["positive_fraction"],
                                       plan["shared"]["seeds"]["sampler"] + index)
    optimizer = build_optimizer_v5(model, recipe)
    scheduler = build_scheduler_v5(optimizer, recipe)
    scaler = build_scaler_v5(recipe, device.type)
    step, prior_wall, last = 0, 0.0, (None, None)
    if resume is not None:
        if not (isinstance(resume, tuple) and len(resume) == 2):
            raise V5Error("INVALID_TYPE", "$.resume")
        state = load_checkpoint_v5(resume[0], unified_run, resume[1], device.type)
        if any(other > state["step"] for other in _checkpoint_steps(checkpoint_dir)):
            raise V5Error("CHECKPOINT_NOT_LATEST", "$.resume")
        if state["data_sha256"] != data_sha256:
            raise V5Error("DATA_MISMATCH", "$.samples")
        if not isinstance(state["sampler"], dict) or set(state["sampler"]) != set(pids):
            raise V5Error("INVALID_SAMPLER_STATE", "$.sampler")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        for pid in pids:
            samplers[pid].load_state_dict(state["sampler"][pid])
        _set_rng_state(state["rng"])
        step, prior_wall, last = state["step"], float(state["wall_seconds"]), (Path(resume[0]), resume[1])
        _append_jsonl(metrics_path, {"event": "resume", "step": step, "checkpoint_sha256": resume[1]})
    else:
        write_private_json(data_path, {"data_sha256": data_sha256})    # exclusive create: the start marker

    def stop(code: str) -> dict:
        write_private_json(failure_path, {"code": code, "step": step,
                                          "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return {"status": "FAILED", "step": step, "last_checkpoint": last[0], "last_checkpoint_sha256": last[1],
                "failure_code": code}

    budget_seconds = recipe["budget"]["max_wall_hours"] * 3600.0
    start = _monotonic()
    model.train()
    while step < max_steps:
        if prior_wall + (_monotonic() - start) > budget_seconds:
            return stop("WALL_BUDGET_EXCEEDED")
        lr = optimizer.param_groups[0]["lr"]
        total, parts = 0.0, {}
        optimizer.zero_grad(set_to_none=True)
        try:
            for pid in pids:
                member, view = members[pid], model.view(pid)
                batch, targets, eligibility = _to_device(
                    *make_batch_v5(member["samples"], samplers[pid], member["recipe"], view.scope), device)
                with autocast_v5(recipe["precision"], device.type):
                    output = view(batch)
                for logits in (output.anatomy_logits, output.abnormality_logits):
                    if logits is not None and not bool(torch.isfinite(logits).all()):
                        raise V5Error("NON_FINITE_LOGITS", "$.output")
                loss, task_parts = unified_loss_v5(output, targets, eligibility, member["supervision"],
                                                   member["box_supervision"])
                scaler.scale(member["loss_weight"] * loss).backward()
                total += member["loss_weight"] * float(loss.detach())
                parts[pid] = {"loss": float(loss.detach()), "tasks": task_parts}
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        except torch.cuda.OutOfMemoryError:
            return stop("OUT_OF_MEMORY")
        except V5Error as err:
            if err.code not in STOP_CODES:
                raise
            return stop(err.code)
        step += 1
        _append_jsonl(metrics_path, {"event": "step", "step": step, "loss": total, "lr": lr, "members": parts})
        if step % every == 0 or step == max_steps:
            path = checkpoint_dir / f"step_{step:08d}.pt"
            state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "sampler": {pid: samplers[pid].state_dict() for pid in pids}, "rng": _rng_state(),
                     "wall_seconds": prior_wall + (_monotonic() - start), "data_sha256": data_sha256}
            last = (path, save_checkpoint_v5(path, state, unified_run, private_root, repo))
            if step == pause_at:
                _append_jsonl(metrics_path, {"event": "pause", "step": step, "checkpoint_sha256": last[1]})
                return {"status": "PAUSED", "step": step, "last_checkpoint": last[0],
                        "last_checkpoint_sha256": last[1], "failure_code": None}
    return {"status": "COMPLETED", "step": step, "last_checkpoint": last[0], "last_checkpoint_sha256": last[1],
            "failure_code": None}
