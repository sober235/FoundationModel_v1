"""v5.1 GPU measurement and small-sample overfit check (plan_v5 T10, real-run step 2).

measure_candidates_v5 runs each explicit {patch_size, batch_size, precision} candidate in turn, with no default
grid: one warm-up forward+backward, then one timed forward+backward, recording peak device memory, seconds per
step and samples per second. A candidate that runs out of device memory is recorded as "OOM" and the next one
runs; it is never retried. overfit_check_v5 trains a fresh model on a few train samples with train_v5's own step
and judges loss drop and Dice by explicit criteria. render_measurement_md_v5 writes the device model and count,
memory, throughput, configuration and pass/fail only: never paths, case IDs or file names.
"""
from __future__ import annotations

import copy
import re
import time

import numpy as np
import torch

from ..contracts_v5 import UNKNOWN_KEY, V5Error
from ..evaluation.segmentation import segmentation_metrics_v5
from ..models.interface_v5 import PerceptionModelV5, build_model_input_v5
from .data_v5 import PatchSamplerV5, TrainingSampleV5, make_batch_v5, positive_centers_v5
from .infer_v5 import predict_volume_v5
from .registration_v5 import _is_int, _is_number, _positive_int, _vector3, validate_recipe_v5, validate_supervision_v5
from .train_v5 import (PRECISIONS, STOP_CODES, _to_device, autocast_v5, build_optimizer_v5, build_scaler_v5,
                       build_scheduler_v5, training_step_v5)

CANDIDATE_KEYS = ("patch_size", "batch_size", "precision")
CRITERIA_KEYS = ("min_loss_drop", "min_dice")
MEASUREMENT_STATUS = ("OK", "OOM")
OVERFIT_FAILURES = (*STOP_CODES, "OUT_OF_MEMORY")
_SAFE_TEXT = re.compile(r"[A-Za-z0-9 _.()+-]{1,80}")
_GIB = float(2 ** 30)
_NO_PHYSICAL = {"physical_geometry_status": None, "spacing_mm": None}
_DICE_ONLY = {"metrics": ["dice"]}


# ---------------------------------------------------------------------------
# Measurement


def _check_candidates(candidates: object) -> None:
    if not isinstance(candidates, list) or not candidates:
        raise V5Error("CANDIDATES_REQUIRED", "$.candidates")
    for index, candidate in enumerate(candidates):
        path = f"$.candidates[{index}]"
        if not isinstance(candidate, dict):
            raise V5Error("INVALID_TYPE", path)
        if any(key not in CANDIDATE_KEYS for key in candidate):
            raise V5Error("UNKNOWN_FIELD", f"{path}.{UNKNOWN_KEY}")
        for key in CANDIDATE_KEYS:
            if candidate.get(key) is None:
                raise V5Error("CANDIDATE_VALUE_MISSING", f"{path}.{key}")
        valid = {"patch_size": _vector3(candidate["patch_size"]), "batch_size": _positive_int(candidate["batch_size"]),
                 "precision": candidate["precision"] in PRECISIONS}
        for key in CANDIDATE_KEYS:
            if not valid[key]:
                raise V5Error("INVALID_CANDIDATE_VALUE", f"{path}.{key}")


def _forward_backward(model, batch, loss_fn, precision: str, device_type: str) -> None:
    with autocast_v5(precision, device_type):
        output = model(batch)
    loss = loss_fn(output)
    model.zero_grad(set_to_none=True)
    loss.backward()


def _measure(build_model, make_batch, candidate: dict, device: torch.device) -> dict:
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(copy.deepcopy(candidate)).to(device)
    model.train()
    batch, loss_fn = make_batch(copy.deepcopy(candidate), device)
    _forward_backward(model, batch, loss_fn, candidate["precision"], device.type)  # warm-up, not timed
    if cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    _forward_backward(model, batch, loss_fn, candidate["precision"], device.type)
    if cuda:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    return {"peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if cuda else None,
            "seconds_per_step": seconds, "samples_per_second": candidate["batch_size"] / seconds}


def measure_candidates_v5(build_model, make_batch, candidates: list, device: str) -> list:
    """Measure every explicit candidate once; all candidates are checked before any runs.

    build_model(candidate) returns the model; make_batch(candidate, device) returns (batch on the device,
    loss_fn(output) -> scalar loss). Peak memory is torch.cuda.max_memory_allocated over both steps, None off
    CUDA. An out-of-memory candidate becomes status "OOM" and the next one runs; any other error is raised.
    """
    _check_candidates(candidates)
    device = torch.device(device)
    rows = []
    for candidate in candidates:
        row = {key: copy.deepcopy(candidate[key]) for key in CANDIDATE_KEYS}
        try:
            row.update(_measure(build_model, make_batch, candidate, device), status="OK")
        except torch.cuda.OutOfMemoryError:
            row.update(status="OOM", peak_memory_bytes=None, seconds_per_step=None, samples_per_second=None)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Small-sample overfit check


def _check_criteria(criteria: object) -> None:
    if not isinstance(criteria, dict):
        raise V5Error("INVALID_TYPE", "$.criteria")
    if any(key not in CRITERIA_KEYS for key in criteria):
        raise V5Error("UNKNOWN_FIELD", f"$.criteria.{UNKNOWN_KEY}")
    for key in CRITERIA_KEYS:
        value = criteria.get(key)
        if value is None:
            raise V5Error("CRITERIA_VALUE_MISSING", f"$.criteria.{key}")
        if not (_is_number(value) and (0 <= value < 1 if key == "min_loss_drop" else 0 <= value <= 1)):
            raise V5Error("INVALID_CRITERIA_VALUE", f"$.criteria.{key}")


def _dice(pred: np.ndarray, truth: np.ndarray, eligible: np.ndarray):
    return segmentation_metrics_v5(pred, truth, eligible, _NO_PHYSICAL, _DICE_ONLY)["dice"]


def _sample_dice(model: PerceptionModelV5, samples: dict, recipe: dict) -> dict:
    """Per task, the mean Dice over samples and classes that have one; whole samples on the model grid."""
    scope, inference = model.scope, recipe["inference"]
    scores = {task: [] for task in model.tasks}
    for uid in sorted(samples):
        sample = samples[uid]
        observation = sample.observation
        batch = build_model_input_v5(observation.channels, scope, observation.valid_fov, observation.grid,
                                     list(observation.to_original))
        prob = predict_volume_v5(model, batch, inference["window"], inference["stride"], recipe["precision"])
        if "A" in model.tasks:
            labels = prob["anatomy"].argmax(axis=0)
            for k in range(1, 1 + len(scope.anatomy_classes)):
                scores["A"].append(_dice(labels == k, sample.targets["A"] == k,
                                         sample.e_pos["A"][k - 1] & sample.e_neg["A"][k - 1]))
        if "U" in model.tasks:
            for c in range(len(scope.abnormality_classes)):
                scores["U"].append(_dice(prob["abnormality"][c] >= 0.5, sample.targets["U"][c].astype(bool),
                                         sample.e_pos["U"][c] & sample.e_neg["U"][c]))
    means = {}
    for task, values in scores.items():
        found = [value for value in values if value is not None]
        means[task] = sum(found) / len(found) if found else None
    return means


def overfit_check_v5(model: PerceptionModelV5, samples: dict, recipe: dict, supervision: dict, steps: int,
                     criteria: dict) -> dict:
    """Train a fresh `model` for `steps` steps on a few train samples, then judge it by explicit criteria.

    Samples come from load_training_set_v5 within a registered smoke run (train assignment only, L4/L6). Each step
    is training_step_v5 with the recipe's sampler, optimizer, schedule, scaler and precision; no checkpoint is
    written. steps may not exceed budget.max_steps, where the poly schedule reaches zero. passed: no stop code,
    loss drop 1 - final/initial >= min_loss_drop, and for every task the mean Dice >= min_dice. Dice is over whole
    samples on the model grid within e_pos & e_neg, A by argmax and U by p >= 0.5. The result holds no identifiers.
    """
    if not isinstance(model, PerceptionModelV5):
        raise V5Error("INVALID_TYPE", "$.model")
    validate_recipe_v5(recipe)
    validate_supervision_v5(supervision, model.scope)
    if sorted(recipe["model"]["tasks"]) != sorted(model.tasks):
        raise V5Error("MODULE_TASKS_INVALID", "$.recipe.model.tasks")
    if not isinstance(samples, dict) or not samples \
            or not all(isinstance(sample, TrainingSampleV5) for sample in samples.values()):
        raise V5Error("INVALID_TYPE", "$.samples")
    if not _positive_int(steps) or steps > recipe["budget"]["max_steps"]:
        raise V5Error("INVALID_STEPS", "$.steps")
    _check_criteria(criteria)

    device = next(model.parameters()).device
    sampler = PatchSamplerV5(sorted(samples), {uid: positive_centers_v5(s) for uid, s in samples.items()},
                             recipe["sampling"]["positive_fraction"], recipe["seeds"]["sampler"])
    optimizer = build_optimizer_v5(model, recipe)
    scheduler = build_scheduler_v5(optimizer, recipe)
    scaler = build_scaler_v5(recipe, device.type)
    losses, failure = [], None
    model.train()
    for _ in range(steps):
        batch, targets, eligibility = _to_device(*make_batch_v5(samples, sampler, recipe, model.scope), device)
        try:
            loss, _ = training_step_v5(model, batch, targets, eligibility, supervision, optimizer, scheduler,
                                       scaler, recipe["precision"])
        except torch.cuda.OutOfMemoryError:
            failure = "OUT_OF_MEMORY"
            break
        except V5Error as err:
            if err.code not in STOP_CODES:
                raise
            failure = err.code
            break
        losses.append(float(loss.detach()))
    dice = {task: None for task in model.tasks} if failure else _sample_dice(model, samples, recipe)
    drop = 1.0 - losses[-1] / losses[0] if losses and losses[0] > 0 else None
    passed = (failure is None and drop is not None and drop >= criteria["min_loss_drop"]
              and all(dice[task] is not None and dice[task] >= criteria["min_dice"] for task in model.tasks))
    return {"steps": steps, "steps_completed": len(losses), "loss_curve": losses,
            "initial_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None,
            "loss_drop": drop, "dice": dice, "criteria": dict(criteria), "passed": bool(passed),
            "failure_code": failure}


# ---------------------------------------------------------------------------
# Redacted report (shared with module_eval_v5)


def _text(value: object, path: str) -> str:
    """A shown text field must look like a plain name: no path separators, nothing long."""
    if not isinstance(value, str) or not _SAFE_TEXT.fullmatch(value):
        raise V5Error("UNSAFE_REPORT_VALUE", path)
    return value


def _number(value: object, path: str, digits: int) -> str:
    if value is None:
        return "—"
    if not _is_number(value):
        raise V5Error("INVALID_SUMMARY", path)
    return f"{value:.{digits}f}"


def _count(value: object, path: str) -> str:
    if not (_is_int(value) and value >= 0):
        raise V5Error("INVALID_SUMMARY", path)
    return str(value)


def _shape(value: object, path: str) -> str:
    if not _vector3(value):
        raise V5Error("INVALID_SUMMARY", path)
    return "×".join(str(v) for v in value)


def _candidate_line(row: object, path: str) -> str:
    if not isinstance(row, dict):
        raise V5Error("INVALID_SUMMARY", path)
    if row.get("status") not in MEASUREMENT_STATUS:
        raise V5Error("INVALID_SUMMARY", f"{path}.status")
    if row.get("precision") not in PRECISIONS:
        raise V5Error("INVALID_SUMMARY", f"{path}.precision")
    peak = row.get("peak_memory_bytes")
    if peak is not None and not _is_number(peak):
        raise V5Error("INVALID_SUMMARY", f"{path}.peak_memory_bytes")
    cells = [_shape(row.get("patch_size"), f"{path}.patch_size"), _count(row.get("batch_size"), f"{path}.batch_size"),
             row["precision"], row["status"], _number(None if peak is None else peak / _GIB, f"{path}.peak", 2),
             _number(row.get("seconds_per_step"), f"{path}.seconds_per_step", 3),
             _number(row.get("samples_per_second"), f"{path}.samples_per_second", 2)]
    return "| " + " | ".join(cells) + " |"


def _overfit_lines(overfit: object) -> list:
    if overfit is None:
        return ["- 未运行。"]
    if not isinstance(overfit, dict) or not isinstance(overfit.get("criteria"), dict) \
            or not isinstance(overfit.get("dice"), dict) or not isinstance(overfit.get("passed"), bool):
        raise V5Error("INVALID_SUMMARY", "$.overfit")
    failure, criteria, dice = overfit.get("failure_code"), overfit["criteria"], overfit["dice"]
    if failure is not None and failure not in OVERFIT_FAILURES:
        raise V5Error("INVALID_SUMMARY", "$.overfit.failure_code")
    if not dice or any(task not in ("A", "U") for task in dice):
        raise V5Error("INVALID_SUMMARY", "$.overfit.dice")
    scores = "；".join(f"{task} {_number(dice[task], '$.overfit.dice', 3)}" for task in sorted(dice))
    return [f"- 步数：{_count(overfit.get('steps'), '$.overfit.steps')}"
            f"（完成 {_count(overfit.get('steps_completed'), '$.overfit.steps_completed')}）",
            f"- 损失：初始 {_number(overfit.get('initial_loss'), '$.overfit.initial_loss', 4)} → "
            f"末 {_number(overfit.get('final_loss'), '$.overfit.final_loss', 4)}；"
            f"下降比例 {_number(overfit.get('loss_drop'), '$.overfit.loss_drop', 3)}"
            f"（判据 ≥ {_number(criteria.get('min_loss_drop'), '$.overfit.criteria.min_loss_drop', 3)}）",
            f"- Dice（样本与类别均值）：{scores}"
            f"（判据 ≥ {_number(criteria.get('min_dice'), '$.overfit.criteria.min_dice', 3)}）",
            f"- 结果：{'PASS' if overfit['passed'] else 'FAIL'}" + (f"（{failure}）" if failure else "")]


def render_measurement_md_v5(summary: dict) -> str:
    """Markdown from {scope_id, device: {name, count}, candidates: [measurement rows], overfit: result | None}.

    Only these fields are read, so anything else in the summary (paths, case IDs, file names) is never written;
    a shown text field that is not a plain name is refused.
    """
    if not isinstance(summary, dict) or not isinstance(summary.get("device"), dict):
        raise V5Error("INVALID_SUMMARY", "$.device")
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise V5Error("INVALID_SUMMARY", "$.candidates")
    device = summary["device"]
    lines = [f"# GPU 实测：{_text(summary.get('scope_id'), '$.scope_id')}", "",
             f"- 设备：{_text(device.get('name'), '$.device.name')} × {_count(device.get('count'), '$.device.count')}",
             "- 每个候选先做一次预热前向+反向，再计时一次前向+反向；峰值显存覆盖这两次。OOM 记录后进入下一项，不重试。",
             "- 本报告只含设备、显存、吞吐、配置与通过与否。", "",
             "| patch_size | batch_size | precision | 状态 | 峰值显存 (GiB) | 每步秒数 | 样本/秒 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [_candidate_line(row, f"$.candidates[{index}]") for index, row in enumerate(candidates)]
    lines += ["", "## 小样本过拟合检查", ""] + _overfit_lines(summary.get("overfit"))
    return "\n".join(lines) + "\n"
