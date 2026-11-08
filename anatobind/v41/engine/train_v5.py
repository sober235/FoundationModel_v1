"""v5.1 resumable training core (plan_v5 T09b; Proposal §4.3, §6.3; L6).

The loss is Σ task_weight·L_t over the supervised tasks, each from masked_objective_v5, so a task without
eligible voxels adds a differentiable zero and leaves the other tasks' scale alone. Training does not start
until the run and its exposure are on disk and the recipe, supervision, model and samples are the registered
ones (L6). A fresh start records the digest of the sample content in data.json (written once; it also marks
the run as started) and every checkpoint carries it: a resume on other content is DATA_MISMATCH. A checkpoint
holds every state a bitwise CPU resume needs; it is written once and never overwritten,
and it is loaded only after its file hash matches, with weights_only=True. A non-finite logit or loss, an OOM or
an exhausted wall budget writes failure.json (code, step, time; no values) and stops without retrying.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import random
import re
import time
from pathlib import Path

import torch

from ..contracts_v5 import V5Error, canonical_sha256
from ..losses import masked_objective_v5
from ..models.interface_v5 import ModelOutputV5, PerceptionModelV5
from ..paths import PrivatePathError, _make_private_dirs, resolve_private_output, write_private_bytes, write_private_json
from .data_v5 import (
    PatchSamplerV5, TrainingSampleV5, make_batch_v5, positive_centers_v5, training_data_sha256_v5)
from .registration_v5 import run_sha256_v5, validate_recipe_v5, validate_supervision_v5

CHECKPOINT_FORMAT = "anatobind_ckpt_v5"
STATE_KEYS = ("step", "model", "optimizer", "scheduler", "scaler", "sampler", "rng", "wall_seconds", "data_sha256")
STOP_CODES = ("NON_FINITE_LOGITS", "NON_FINITE_LOSS")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_NAME = re.compile(r"step_(\d{8})\.pt")
_monotonic = time.monotonic     # read through the module so the wall budget can be driven in tests


# ---------------------------------------------------------------------------
# Loss, optimizer, schedule


def _support(task: str, target: torch.Tensor, e_pos: torch.Tensor, e_neg: torch.Tensor) -> int:
    if task == "A":
        return int((e_pos.all(dim=1) & e_neg.all(dim=1)).sum())
    target = target.to(torch.bool)
    return int(((target & e_pos) | (~target & e_neg)).sum())


def training_loss_v5(output: ModelOutputV5, targets: dict, eligibility: dict,
                     supervision: dict) -> tuple[torch.Tensor, dict]:
    """Σ task_weight·L_t; returns the total and, per task, its unweighted loss and supervised voxel count."""
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
        loss = masked_objective_v5(logits.float(), targets[task], e_pos, e_neg, tasks[task])
        if not bool(torch.isfinite(loss)):
            raise V5Error("NON_FINITE_LOSS", f"$.tasks.{task}")
        weighted = tasks[task]["task_weight"] * loss
        total = weighted if total is None else total + weighted
        parts[task] = {"loss": float(loss.detach()), "support": _support(task, targets[task], e_pos, e_neg)}
    return total, parts


def build_optimizer_v5(model: torch.nn.Module, recipe: dict) -> torch.optim.Optimizer:
    validate_recipe_v5(recipe)
    return torch.optim.AdamW(model.parameters(), lr=recipe["optimizer"]["lr"],
                             weight_decay=recipe["optimizer"]["weight_decay"])


def build_scheduler_v5(optimizer: torch.optim.Optimizer, recipe: dict):
    """poly: lr_k = lr · (1 - k / max_steps) ** exponent."""
    validate_recipe_v5(recipe)
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=recipe["budget"]["max_steps"],
                                                 power=recipe["scheduler"]["exponent"])


def build_scaler_v5(recipe: dict, device_type: str) -> torch.amp.GradScaler:
    """Enabled for amp_fp16 only; fp32 and amp_bf16 keep a disabled scaler whose (empty) state is still saved."""
    validate_recipe_v5(recipe)
    return torch.amp.GradScaler(device_type, enabled=recipe["precision"] == "amp_fp16")


PRECISIONS = ("fp32", "amp_bf16", "amp_fp16")


def autocast_v5(precision: str, device_type: str):
    """No autocast for fp32; bf16 or fp16 autocast for the amp precisions (training and inference alike)."""
    if precision not in PRECISIONS:
        raise V5Error("INVALID_PRECISION", "$.precision")
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "amp_bf16" else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)


def training_step_v5(model: PerceptionModelV5, batch, targets: dict, eligibility: dict, supervision: dict,
                     optimizer, scheduler, scaler, precision: str) -> tuple[torch.Tensor, dict]:
    """One step as train_v5 runs it: forward under autocast, every logit finite, loss, scaled backward, then the
    optimizer, scaler and scheduler steps. Raises V5Error NON_FINITE_LOGITS / NON_FINITE_LOSS (STOP_CODES)."""
    with autocast_v5(precision, batch.image.device.type):
        output = model(batch)
    for logits in (output.anatomy_logits, output.abnormality_logits):
        if logits is not None and not bool(torch.isfinite(logits).all()):
            raise V5Error("NON_FINITE_LOGITS", "$.output")
    loss, parts = training_loss_v5(output, targets, eligibility, supervision)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss, parts


# ---------------------------------------------------------------------------
# Checkpoints


def _rng_state() -> dict:
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint_v5(path: Path, state: dict, run: dict, private_root: Path, repo: Path) -> str:
    """Write {format, run_sha256, *state} once (temp file + link, 600) and a `<name>.json` sidecar; return sha256."""
    if not isinstance(state, dict) or set(state) != set(STATE_KEYS):
        raise V5Error("INVALID_CHECKPOINT_STATE", "$.state")
    step = state["step"]
    if not (isinstance(step, int) and not isinstance(step, bool) and step >= 0):
        raise V5Error("INVALID_CHECKPOINT_STATE", "$.state.step")
    path = resolve_private_output(Path(path), private_root, repo)
    sidecar = resolve_private_output(path.with_name(f"{path.name}.json"), private_root, repo)
    if path.exists() or sidecar.exists():
        raise PrivatePathError("PRIVATE_FILE_EXISTS")
    payload = {"format": CHECKPOINT_FORMAT, "run_sha256": run_sha256_v5(run), **state}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    write_private_bytes(path, data)
    write_private_json(sidecar, {"sha256": digest, "step": step, "run_sha256": payload["run_sha256"]})
    return digest


def load_checkpoint_v5(path: Path, run: dict, expected_sha256: str, device: str) -> dict:
    """Hash first, then weights_only load of the very bytes that were hashed; the run must be this run."""
    if not (isinstance(expected_sha256, str) and _SHA256.fullmatch(expected_sha256)):
        raise V5Error("INVALID_SHA256", "$.expected_sha256")
    try:
        data = Path(path).read_bytes()
    except OSError:
        raise V5Error("CHECKPOINT_MISSING", "$.checkpoint") from None
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise V5Error("CHECKPOINT_HASH_MISMATCH", "$.checkpoint")
    try:
        state = torch.load(io.BytesIO(data), map_location=device, weights_only=True)
    except Exception:  # noqa: BLE001 - any unpickling refusal is a format error; never echo its message
        raise V5Error("CHECKPOINT_FORMAT_INVALID", "$.checkpoint") from None
    if not (isinstance(state, dict) and state.get("format") == CHECKPOINT_FORMAT
            and set(state) == {"format", "run_sha256", *STATE_KEYS}):
        raise V5Error("CHECKPOINT_FORMAT_INVALID", "$.checkpoint")
    if state["run_sha256"] != run_sha256_v5(run):
        raise V5Error("RUN_MISMATCH", "$.run_sha256")
    return state


# ---------------------------------------------------------------------------
# Training


def _registered_exposure(run: dict, run_dir: Path, private_root: Path, repo: Path) -> dict:
    run_path = resolve_private_output(Path(run_dir) / "run.json", private_root, repo)
    exposure_path = resolve_private_output(Path(run_dir) / "exposure.json", private_root, repo)
    try:
        registered = json.loads(run_path.read_text())
        exposure = json.loads(exposure_path.read_text())
    except (OSError, ValueError):
        raise V5Error("RUN_NOT_REGISTERED", "$.run_dir") from None
    if registered != run or not isinstance(exposure, dict) or exposure.get("run_id") != run.get("run_id") \
            or exposure.get("run_sha256") != run_sha256_v5(run) or not isinstance(exposure.get("entries"), dict):
        raise V5Error("RUN_NOT_REGISTERED", "$.run_dir")
    return exposure


def _check_binding(run: dict, model: PerceptionModelV5, samples: dict, recipe: dict, supervision: dict,
                   exposure: dict) -> None:
    if canonical_sha256(recipe) != run["recipe_sha256"]:
        raise V5Error("RUN_MISMATCH", "$.recipe")
    if canonical_sha256(supervision) != run["supervision_sha256"]:
        raise V5Error("RUN_MISMATCH", "$.supervision")
    scope = model.scope
    if {"scope_id": scope.scope_id, "scope_hash": scope.scope_hash} != run["scope"] \
            or sorted(model.tasks) != sorted(recipe["model"]["tasks"]):
        raise V5Error("RUN_MISMATCH", "$.model")
    if not isinstance(samples, dict) or not samples:
        raise V5Error("INVALID_TYPE", "$.samples")
    trained = {uid for uid, entries in exposure["entries"].items()
               if isinstance(entries, list) and any(isinstance(e, dict) and e.get("role") == "training"
                                                    for e in entries)}
    if set(samples) != trained or any(not isinstance(sample, TrainingSampleV5)
                                      or sample.observation.sample_uid != uid for uid, sample in samples.items()):
        raise V5Error("SAMPLE_NOT_REGISTERED", "$.samples")


def _append_jsonl(path: Path, value: dict) -> None:
    _make_private_dirs(path.parent)
    with os.fdopen(os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600), "a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _checkpoint_steps(directory: Path) -> list[int]:
    if not directory.is_dir():
        return []
    return [int(match.group(1)) for match in (_CHECKPOINT_NAME.fullmatch(p.name) for p in directory.iterdir())
            if match]


def _to_device(batch, targets: dict, eligibility: dict, device: torch.device):
    batch = dataclasses.replace(batch, image=batch.image.to(device), valid_fov=batch.valid_fov.to(device),
                                valid_slices=batch.valid_slices.to(device))
    targets = {task: value.to(device) for task, value in targets.items()}
    eligibility = {task: {key: value.to(device) for key, value in masks.items()}
                   for task, masks in eligibility.items()}
    return batch, targets, eligibility


def train_v5(run: dict, model: PerceptionModelV5, samples: dict, recipe: dict, supervision: dict, run_dir: Path,
             private_root: Path, repo: Path, resume: tuple | None) -> dict:
    """Train a registered run to budget.max_steps; returns {status, step, last_checkpoint, last_checkpoint_sha256,
    failure_code}. `resume` is (checkpoint path, expected sha256) or None for a fresh start."""
    if not isinstance(model, PerceptionModelV5):
        raise V5Error("INVALID_TYPE", "$.model")
    validate_recipe_v5(recipe)
    exposure = _registered_exposure(run, run_dir, private_root, repo)
    _check_binding(run, model, samples, recipe, supervision, exposure)
    validate_supervision_v5(supervision, model.scope)
    data_sha256 = training_data_sha256_v5(samples)
    run_dir = Path(run_dir)
    metrics_path = resolve_private_output(run_dir / "metrics.jsonl", private_root, repo)
    failure_path = resolve_private_output(run_dir / "failure.json", private_root, repo)
    checkpoint_dir = resolve_private_output(run_dir / "checkpoints", private_root, repo)
    data_path = resolve_private_output(run_dir / "data.json", private_root, repo)
    if resume is None and any(path.exists() for path in (metrics_path, failure_path, checkpoint_dir, data_path)):
        raise V5Error("RUN_ALREADY_STARTED", "$.run_dir")
    if resume is not None and failure_path.exists():
        raise V5Error("RUN_FAILED", "$.run_dir")

    device = next(model.parameters()).device
    sampler = PatchSamplerV5(sorted(samples), {uid: positive_centers_v5(s) for uid, s in samples.items()},
                             recipe["sampling"]["positive_fraction"], recipe["seeds"]["sampler"])
    optimizer = build_optimizer_v5(model, recipe)
    scheduler = build_scheduler_v5(optimizer, recipe)
    scaler = build_scaler_v5(recipe, device.type)
    step, prior_wall, last = 0, 0.0, (None, None)
    if resume is not None:
        if not (isinstance(resume, tuple) and len(resume) == 2):
            raise V5Error("INVALID_TYPE", "$.resume")
        state = load_checkpoint_v5(resume[0], run, resume[1], device.type)
        if any(other > state["step"] for other in _checkpoint_steps(checkpoint_dir)):
            raise V5Error("CHECKPOINT_NOT_LATEST", "$.resume")
        if state["data_sha256"] != data_sha256:
            raise V5Error("DATA_MISMATCH", "$.samples")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        sampler.load_state_dict(state["sampler"])
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

    max_steps = recipe["budget"]["max_steps"]
    budget_seconds = recipe["budget"]["max_wall_hours"] * 3600.0
    start = _monotonic()
    model.train()
    while step < max_steps:
        if prior_wall + (_monotonic() - start) > budget_seconds:
            return stop("WALL_BUDGET_EXCEEDED")
        batch, targets, eligibility = _to_device(*make_batch_v5(samples, sampler, recipe, model.scope), device)
        lr = optimizer.param_groups[0]["lr"]
        try:
            loss, parts = training_step_v5(model, batch, targets, eligibility, supervision, optimizer, scheduler,
                                           scaler, recipe["precision"])
        except torch.cuda.OutOfMemoryError:
            return stop("OUT_OF_MEMORY")
        except V5Error as err:
            if err.code not in STOP_CODES:
                raise
            return stop(err.code)
        step += 1
        _append_jsonl(metrics_path, {"event": "step", "step": step, "loss": float(loss.detach()), "lr": lr,
                                     "tasks": parts})
        if step % recipe["checkpoint_every"] == 0 or step == max_steps:
            path = checkpoint_dir / f"step_{step:08d}.pt"
            state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "sampler": sampler.state_dict(), "rng": _rng_state(),
                     "wall_seconds": prior_wall + (_monotonic() - start), "data_sha256": data_sha256}
            last = (path, save_checkpoint_v5(path, state, run, private_root, repo))
    return {"status": "COMPLETED", "step": step, "last_checkpoint": last[0], "last_checkpoint_sha256": last[1],
            "failure_code": None}
