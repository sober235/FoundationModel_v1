"""v5.1 GPU measurement and small-sample overfit check (plan_v5 T10, real-run step 2).

measure_candidates_v5 runs each explicit {patch_size, batch_size, precision} candidate in turn, with no default
grid, in two phases on fresh models: (1) forward+backward only (warm-up, then one timed step); (2) the full
training step exactly as train_v5 runs it (training_step_v5: autocast forward, finite-logit check, loss, scaled
backward, optimizer.step, scaler.update, scheduler.step) with the recipe's optimizer, scaler and poly schedule
built anew, so the peak memory of phase 2 includes building the AdamW state (one warm-up step, then one timed
step). A candidate is OK only if its full training step ran: forward+backward alone never makes it usable. An
out-of-memory (or non-finite) candidate is recorded with the phase it failed in and the next one runs; it is
never retried, and the same candidate may not be listed twice. overfit_check_v5 trains a fresh model on a few
train samples with train_v5's own step and judges loss drop and Dice by explicit criteria.
render_measurement_md_v5 writes the device model and count, memory,
throughput, configuration and pass/fail only: never paths, case IDs or file names.
"""
from __future__ import annotations

import copy
import math
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
                       build_scheduler_v5, training_loss_v5, training_step_v5)

CANDIDATE_KEYS = ("patch_size", "batch_size", "precision")
CRITERIA_KEYS = ("min_loss_drop", "min_dice", "loss_window")
RESUME_TOLERANCE_KEYS = ("loss_atol", "loss_rtol", "param_rtol")
RESUME_EXACT_KEYS = ("step", "sampler", "scheduler", "scaler", "rng", "data_sha256", "optimizer_counters",
                     "steps_after_pause")
MEASUREMENT_STATUS = ("OK", "OOM", "NON_FINITE")
MEASUREMENT_PHASES = ("forward_backward", "training_step")
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
    seen = []
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
        key = (tuple(candidate["patch_size"]), candidate["batch_size"], candidate["precision"])
        if key in seen:
            raise V5Error("CANDIDATE_DUPLICATE", path)      # one measurement per configuration, never a retry
        seen.append(key)


def _candidate_recipe(recipe: dict, candidate: dict) -> dict:
    return validate_recipe_v5({**copy.deepcopy(recipe), **{key: copy.deepcopy(candidate[key]) for key in CANDIDATE_KEYS}})


def _timed(step, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak(device: torch.device):
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def _forward_backward_phase(build_model, make_batch, candidate: dict, supervision: dict, device: torch.device):
    _reset_peak(device)
    model = build_model(copy.deepcopy(candidate)).to(device)
    model.train()
    batch, targets, eligibility = make_batch(copy.deepcopy(candidate), device)

    def step() -> None:
        with autocast_v5(candidate["precision"], device.type):
            output = model(batch)
        loss, _ = training_loss_v5(output, targets, eligibility, supervision)
        model.zero_grad(set_to_none=True)
        loss.backward()

    step()                                                   # warm-up, not timed
    seconds = _timed(step, device)
    return _peak(device), seconds


def _training_step_phase(build_model, make_batch, candidate: dict, recipe: dict, supervision: dict,
                         device: torch.device):
    _reset_peak(device)
    model = build_model(copy.deepcopy(candidate)).to(device)
    model.train()
    batch, targets, eligibility = make_batch(copy.deepcopy(candidate), device)
    optimizer = build_optimizer_v5(model, recipe)
    scheduler = build_scheduler_v5(optimizer, recipe)
    scaler = build_scaler_v5(recipe, device.type)

    def step() -> None:
        training_step_v5(model, batch, targets, eligibility, supervision, optimizer, scheduler, scaler,
                         recipe["precision"])

    step()                                                   # warm-up builds the optimizer state, not timed
    seconds = _timed(step, device)
    return _peak(device), seconds


def measure_candidates_v5(build_model, make_batch, candidates: list, device: str, *, recipe: dict,
                          supervision: dict) -> list:
    """Measure every explicit candidate once, in two phases; all candidates are checked before any runs.

    build_model(candidate) returns a fresh model (called once per phase); make_batch(candidate, device) returns
    (batch, targets, eligibility) on the device, as make_batch_v5 builds them. `recipe` gives the optimizer,
    schedule and everything but the candidate's own patch_size, batch_size and precision. Peak memory is
    torch.cuda.max_memory_allocated within each phase, None off CUDA. Out-of-memory or a non-finite stop code
    ends the candidate with its phase recorded and the next one runs; any other error is raised.
    """
    _check_candidates(candidates)
    validate_recipe_v5(recipe)
    device = torch.device(device)
    rows = []
    for candidate in candidates:
        candidate_recipe = _candidate_recipe(recipe, candidate)
        row = {key: copy.deepcopy(candidate[key]) for key in CANDIDATE_KEYS}
        row.update(status="OK", failure_phase=None, fb_peak_memory_bytes=None, fb_seconds_per_step=None,
                   peak_memory_bytes=None, seconds_per_step=None, samples_per_second=None,
                   optimizer_state_included=False, scaler_enabled=candidate["precision"] == "amp_fp16",
                   scheduler=recipe["scheduler"]["name"])
        phase = "forward_backward"
        try:
            row["fb_peak_memory_bytes"], row["fb_seconds_per_step"] = _forward_backward_phase(
                build_model, make_batch, candidate, supervision, device)
            phase = "training_step"
            row["optimizer_state_included"] = True
            peak, seconds = _training_step_phase(build_model, make_batch, candidate, candidate_recipe, supervision,
                                                 device)
            row.update(peak_memory_bytes=peak, seconds_per_step=seconds,
                       samples_per_second=candidate["batch_size"] / seconds)
        except torch.cuda.OutOfMemoryError:
            row.update(status="OOM", failure_phase=phase)
        except V5Error as err:
            if err.code not in STOP_CODES:
                raise
            row.update(status="NON_FINITE", failure_phase=phase)
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
        valid = _positive_int(value) if key == "loss_window" else \
            _is_number(value) and (0 <= value < 1 if key == "min_loss_drop" else 0 <= value <= 1)
        if not valid:
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


def judge_overfit_v5(model: PerceptionModelV5, samples: dict, recipe: dict, loss_curve: list, criteria: dict, *,
                     failure_code: str | None) -> dict:
    """Judge a model trained on a few train samples (by overfit_check_v5 or by train_v5) against explicit criteria.

    With w = criteria.loss_window: initial = mean of the first w step losses, final = mean of the last w, drop =
    1 - final/initial (None when the curve is shorter than 2w or initial is not positive). passed: no failure
    code, drop >= min_loss_drop, and for every task the mean Dice >= min_dice. Dice is over whole samples on the
    model grid within e_pos & e_neg, A by argmax and U by p >= 0.5. The result holds no identifiers.
    """
    _check_criteria(criteria)
    if failure_code is not None and failure_code not in OVERFIT_FAILURES:
        raise V5Error("INVALID_FAILURE_CODE", "$.failure_code")
    losses = [float(value) for value in loss_curve]
    window = criteria["loss_window"]
    initial = final = drop = None
    if len(losses) >= 2 * window:
        initial, final = sum(losses[:window]) / window, sum(losses[-window:]) / window
        drop = 1.0 - final / initial if initial > 0 else None
    dice = {task: None for task in model.tasks} if failure_code else _sample_dice(model, samples, recipe)
    passed = (failure_code is None and drop is not None and drop >= criteria["min_loss_drop"]
              and all(dice[task] is not None and dice[task] >= criteria["min_dice"] for task in model.tasks))
    return {"steps_completed": len(losses), "loss_curve": losses, "initial_loss": initial, "final_loss": final,
            "loss_drop": drop, "dice": dice, "criteria": dict(criteria), "passed": bool(passed),
            "failure_code": failure_code}


def overfit_check_v5(model: PerceptionModelV5, samples: dict, recipe: dict, supervision: dict, steps: int,
                     criteria: dict) -> dict:
    """Train a fresh `model` for `steps` steps on a few train samples, then judge it with judge_overfit_v5.

    Samples come from load_training_set_v5 within a registered smoke run (train assignment only, L4/L6). Each step
    is training_step_v5 with the recipe's sampler, optimizer, schedule, scaler and precision; no checkpoint is
    written. steps may not exceed budget.max_steps, where the poly schedule reaches zero, and must cover two loss
    windows.
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
    _check_criteria(criteria)
    if not _positive_int(steps) or steps > recipe["budget"]["max_steps"] or steps < 2 * criteria["loss_window"]:
        raise V5Error("INVALID_STEPS", "$.steps")

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
    return {"steps": steps, **judge_overfit_v5(model, samples, recipe, losses, criteria, failure_code=failure)}


# ---------------------------------------------------------------------------
# Resume check: continuous execution against pause + resume of the same registered run


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype \
            and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float(torch.linalg.vector_norm(a - b) / max(float(torch.linalg.vector_norm(a)), 1e-12))


def _step_losses(lines: list, pause_at: int) -> dict:
    return {line["step"]: float(line["loss"]) for line in lines
            if isinstance(line, dict) and line.get("event") == "step" and line["step"] > pause_at}


def compare_resume_v5(continuous: dict, resumed: dict, continuous_steps: list, resumed_steps: list, *,
                      pause_at: int, tolerances: dict) -> dict:
    """Compare the final checkpoint states and the per-step losses after `pause_at` of two executions.

    Exact: final step, sampler, scheduler, scaler and RNG states, data digest, the optimizer's param groups and
    per-parameter step counts, and the set of steps logged after the pause. Within the registered tolerances
    {loss_atol, loss_rtol, param_rtol}: every logged loss (|a - b| <= atol + rtol·|a|), and the relative L2
    difference of every parameter and AdamW moment. Bitwise equality is reported but only required on CPU tests.
    """
    if not isinstance(tolerances, dict) or set(tolerances) != set(RESUME_TOLERANCE_KEYS) \
            or not all(_is_number(v) and v >= 0 for v in tolerances.values()):
        raise V5Error("INVALID_TOLERANCE", "$.tolerances")
    if not _positive_int(pause_at):
        raise V5Error("INVALID_PAUSE", "$.pause_at")
    losses_a, losses_b = _step_losses(continuous_steps, pause_at), _step_losses(resumed_steps, pause_at)
    opt_a, opt_b = continuous["optimizer"], resumed["optimizer"]
    counters = _same(opt_a["param_groups"], opt_b["param_groups"]) and set(opt_a["state"]) == set(opt_b["state"]) \
        and all(_same(opt_a["state"][k].get("step"), opt_b["state"][k].get("step")) for k in opt_a["state"])
    exact = {"step": continuous["step"] == resumed["step"],
             **{key: _same(continuous[key], resumed[key]) for key in ("sampler", "scheduler", "scaler", "rng")},
             "data_sha256": continuous["data_sha256"] == resumed["data_sha256"],
             "optimizer_counters": bool(counters),
             "steps_after_pause": sorted(losses_a) == sorted(losses_b)
             == list(range(pause_at + 1, continuous["step"] + 1))}
    model_a, model_b = continuous["model"], resumed["model"]
    same_keys = set(model_a) == set(model_b)
    param_diff = max((_relative(model_a[k], model_b[k]) for k in model_a), default=0.0) if same_keys else math.inf
    moments = [(opt_a["state"][k][m], opt_b["state"][k][m]) for k in opt_a["state"] if k in opt_b["state"]
               for m in ("exp_avg", "exp_avg_sq") if m in opt_a["state"][k] and m in opt_b["state"][k]]
    moment_diff = max((_relative(a, b) for a, b in moments), default=0.0)
    shared = sorted(set(losses_a) & set(losses_b))
    loss_diff = max((abs(losses_a[s] - losses_b[s]) for s in shared), default=0.0)
    loss_ok = all(abs(losses_a[s] - losses_b[s]) <= tolerances["loss_atol"] + tolerances["loss_rtol"] * abs(losses_a[s])
                  for s in shared)
    bitwise = same_keys and all(_same(model_a[k], model_b[k]) for k in model_a)
    passed = all(exact.values()) and loss_ok and param_diff <= tolerances["param_rtol"] \
        and moment_diff <= tolerances["param_rtol"]
    return {"exact": exact, "steps_compared": len(shared), "max_loss_abs_diff": loss_diff,
            "max_param_rel_diff": param_diff, "max_optimizer_rel_diff": moment_diff, "bitwise_parameters": bool(bitwise),
            "tolerances": dict(tolerances), "pause_at": pause_at, "passed": bool(passed)}


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


def _gib(row: dict, key: str, path: str) -> str:
    value = row.get(key)
    if value is not None and not _is_number(value):
        raise V5Error("INVALID_SUMMARY", f"{path}.{key}")
    return _number(None if value is None else value / _GIB, f"{path}.{key}", 2)


def _candidate_line(row: object, path: str) -> str:
    if not isinstance(row, dict):
        raise V5Error("INVALID_SUMMARY", path)
    if row.get("status") not in MEASUREMENT_STATUS:
        raise V5Error("INVALID_SUMMARY", f"{path}.status")
    if row.get("precision") not in PRECISIONS:
        raise V5Error("INVALID_SUMMARY", f"{path}.precision")
    phase = row.get("failure_phase")
    if (row["status"] == "OK") != (phase is None) or (phase is not None and phase not in MEASUREMENT_PHASES):
        raise V5Error("INVALID_SUMMARY", f"{path}.failure_phase")
    for key in ("optimizer_state_included", "scaler_enabled"):
        if not isinstance(row.get(key), bool):
            raise V5Error("INVALID_SUMMARY", f"{path}.{key}")
    status = row["status"] if phase is None else f"{row['status']}（{phase}）"
    cells = [_shape(row.get("patch_size"), f"{path}.patch_size"), _count(row.get("batch_size"), f"{path}.batch_size"),
             row["precision"], status, _gib(row, "fb_peak_memory_bytes", path),
             _number(row.get("fb_seconds_per_step"), f"{path}.fb_seconds_per_step", 3),
             _gib(row, "peak_memory_bytes", path), _number(row.get("seconds_per_step"), f"{path}.seconds_per_step", 3),
             _number(row.get("samples_per_second"), f"{path}.samples_per_second", 2),
             "是" if row["optimizer_state_included"] else "否", "开" if row["scaler_enabled"] else "关"]
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
            f"（判据 ≥ {_number(criteria.get('min_loss_drop'), '$.overfit.criteria.min_loss_drop', 3)}；"
            f"首末各 {_count(criteria.get('loss_window'), '$.overfit.criteria.loss_window')} 步平均）",
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
             "- 每个候选分两段、各用新建模型测量，每段先预热一次再计时一次：",
             "  1. 前向+反向；",
             "  2. 完整训练步，与 train_v5 相同：autocast 前向、logits 有限性检查、损失、（scaler）反向、"
             "optimizer.step、scaler.update、scheduler.step；优化器与调度器新建，峰值显存包含 AdamW 状态的建立。",
             "- 只有完整训练步成功才记 OK；仅前向+反向成功不能作为锁定配置的依据。"
             "OOM 或非有限值记录失败阶段后进入下一项，同一配置不重试、不重复列出。",
             "- 本报告只含设备、显存、吞吐、配置与通过与否。", "",
             "| patch_size | batch_size | precision | 状态 | 前向+反向峰值 (GiB) | 前向+反向秒数 "
             "| 完整训练步峰值 (GiB) | 完整训练步秒数 | 样本/秒 | 含优化器状态 | GradScaler |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [_candidate_line(row, f"$.candidates[{index}]") for index, row in enumerate(candidates)]
    lines += ["", "## 小样本过拟合检查", ""] + _overfit_lines(summary.get("overfit"))
    return "\n".join(lines) + "\n"
