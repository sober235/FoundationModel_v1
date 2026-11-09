"""Real T10 smoke driver for v5.1 (plan_v5 §2C T10, real-run step 2; Proposal_v5 §6.3, §9).

It only orchestrates what B2 built: registration_v5 (smoke run, frozen split, subset), data_v5 (observations and
training samples), train_v5 (training, checkpoints, controlled pause and resume), smoke_v5 (measurement, overfit
judgement, resume comparison, report) and perception/infer_v5 (full-volume inference, restore, validation, private
save). No training framework is rebuilt here.

A smoke runs from one smoke plan (configs/v5/proposals/smoke_<scope>.yaml). The plan states everything the
researcher approves: the base recipe (without patch, batch and precision), the explicit candidates and the rule
that picks one, the supervision and acceptance policies, the subset rule, the overfit criteria, the pause step and
tolerances of the resume check, and the wall budget of the whole smoke. The researcher's `resources` approval names
the plan's sha256, so any change to the plan needs a new approval.

For one scope the smoke does, in order and each phase only within the remaining budget:
1. the rule-derived train subset (select_subset_v5) and its samples;
2. a real-batch check: channel order, original grid, label maps, FOV and supervised support (counts only);
3. the two-phase measurement of every candidate (forward+backward, then the full training step);
4. one registered smoke run (run_kind smoke, readable_roles [train], the subset), written under two private
   execution roots: `continuous` trains it to max_steps; `resumed` trains it to pause_at, then a fresh model
   resumes it from that checkpoint to max_steps;
5. the overfit judgement of the continuous model (windowed loss drop, whole-volume Dice on the subset);
6. the resume comparison (exact counters, registered tolerances);
7. one full-volume inference through infer_and_save_v5 on a subset sample, restored to the original grid,
   validated, saved privately and reloaded by hash.
A failure, out-of-memory or timeout stops the smoke at that phase; it is recorded, never retried.

Private products stay under the private root (700/600, never overwritten); render_smoke_md_v5 shows counts,
configuration, metrics and hashes only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .contracts_v5 import ActiveScope, V5Error, canonical_sha256
from .engine.data_v5 import (AXIS_ORDER, PatchSamplerV5, load_observation_v5, load_training_sample_v5,
                             make_batch_v5, original_geometry_v5, positive_centers_v5)
from .engine.registration_v5 import (_is_number, _non_empty_str, _positive_int, register_run_v5, run_sha256_v5,
                                     select_subset_v5, validate_recipe_v5, validate_supervision_v5, write_run_v5)
from .engine.smoke_v5 import (CANDIDATE_KEYS, _count, _number, _text, compare_resume_v5, judge_overfit_v5,
                              measure_candidates_v5, render_measurement_md_v5)
from .engine.train_v5 import _to_device, load_checkpoint_v5, train_v5
from .gates_v5 import FIXTURE_APPROVERS, gate_task_v5
from .models.interface_v5 import build_model_input_v5, collate_model_inputs_v5
from .models.resunet_v5 import build_model_v5
from .paths import resolve_private_output, write_private_json
from .perception import infer_and_save_v5, validate_acceptance_policy_v5
from .perception_io import load_output_v5

PLAN_KEYS = ("plan_id", "scope_id", "scope_hash", "source", "lineage_id", "recipe", "supervision",
             "acceptance_policy", "candidates", "candidate_rule", "subset", "overfit", "resume", "budget")
CANDIDATE_RULES = ("first_ok_within_memory",)
SMOKE_STATUS = ("PASS", "FAIL")


# ---------------------------------------------------------------------------
# Plan, approval and candidate choice


def _keys(value: object, keys: tuple, path: str) -> dict:
    if not isinstance(value, dict):
        raise V5Error("INVALID_SMOKE_PLAN", path)
    if set(value) != set(keys):
        raise V5Error("INVALID_SMOKE_PLAN", path)
    return value


def recipe_for_candidate_v5(plan: dict, candidate: dict) -> dict:
    """The run recipe: the plan's base recipe with the candidate's patch_size, batch_size and precision."""
    return validate_recipe_v5({**copy.deepcopy(plan["recipe"]),
                               **{key: copy.deepcopy(candidate[key]) for key in CANDIDATE_KEYS}})


def validate_smoke_plan_v5(plan: dict, scope: ActiveScope, source: dict) -> dict:
    """Every value is explicit and bound to this scope and source; each candidate must give a valid recipe."""
    _keys(plan, PLAN_KEYS, "$")
    if not _non_empty_str(plan["plan_id"]) or not _non_empty_str(plan["lineage_id"]):
        raise V5Error("INVALID_SMOKE_PLAN", "$.plan_id")
    if plan["scope_id"] != scope.scope_id or plan["scope_hash"] != scope.scope_hash:
        raise V5Error("SMOKE_PLAN_SCOPE_MISMATCH", "$.scope_hash")
    if plan["source"] != source:
        raise V5Error("SMOKE_PLAN_SOURCE_MISMATCH", "$.source")
    if not isinstance(plan["recipe"], dict) or any(key in plan["recipe"] for key in CANDIDATE_KEYS):
        raise V5Error("INVALID_SMOKE_PLAN", "$.recipe")
    if plan["recipe"].get("readable_roles") != ["train"]:
        raise V5Error("SMOKE_ROLES_INVALID", "$.recipe.readable_roles")
    validate_supervision_v5(plan["supervision"], scope)
    validate_acceptance_policy_v5(plan["acceptance_policy"], scope)
    if not isinstance(plan["candidates"], list) or not plan["candidates"]:
        raise V5Error("CANDIDATES_REQUIRED", "$.candidates")
    for index, candidate in enumerate(plan["candidates"]):
        if not isinstance(candidate, dict) or set(candidate) != set(CANDIDATE_KEYS):
            raise V5Error("INVALID_SMOKE_PLAN", f"$.candidates[{index}]")
        recipe = recipe_for_candidate_v5(plan, candidate)
        build_model_v5(recipe, scope)          # PATCH_NOT_DIVISIBLE etc. before anything runs
    rule = _keys(plan["candidate_rule"], ("name", "max_memory_fraction"), "$.candidate_rule")
    if rule["name"] not in CANDIDATE_RULES or not (_is_number(rule["max_memory_fraction"])
                                                   and 0 < rule["max_memory_fraction"] <= 1):
        raise V5Error("INVALID_SMOKE_PLAN", "$.candidate_rule")
    subset = _keys(plan["subset"], ("salt", "count", "strata"), "$.subset")
    if not _non_empty_str(subset["salt"]) or not _positive_int(subset["count"]):
        raise V5Error("INVALID_SMOKE_PLAN", "$.subset")
    _keys(plan["overfit"], ("criteria",), "$.overfit")
    resume = _keys(plan["resume"], ("pause_at", "tolerances", "deterministic"), "$.resume")
    max_steps, every = plan["recipe"]["budget"]["max_steps"], plan["recipe"]["checkpoint_every"]
    if not (_positive_int(resume["pause_at"]) and resume["pause_at"] < max_steps and resume["pause_at"] % every == 0):
        raise V5Error("INVALID_PAUSE", "$.resume.pause_at")
    if not isinstance(resume["deterministic"], bool):
        raise V5Error("INVALID_SMOKE_PLAN", "$.resume.deterministic")
    budget = _keys(plan["budget"], ("max_wall_hours",), "$.budget")
    if not (_is_number(budget["max_wall_hours"]) and budget["max_wall_hours"] > 0):
        raise V5Error("INVALID_SMOKE_PLAN", "$.budget.max_wall_hours")
    if plan["recipe"]["budget"]["max_wall_hours"] > budget["max_wall_hours"]:
        raise V5Error("INVALID_SMOKE_PLAN", "$.recipe.budget.max_wall_hours")
    return plan


def smoke_plan_sha256_v5(plan: dict) -> str:
    return canonical_sha256(plan)


def check_smoke_approval_v5(approval: object, plan_sha256: str, scope: ActiveScope, fixture_mode: bool) -> dict:
    """The researcher's resources approval for this scope must name this exact plan."""
    if not (isinstance(approval, dict) and _non_empty_str(approval.get("approved_by"))
            and _non_empty_str(approval.get("approved_at"))):
        raise V5Error("APPROVAL_MISSING", "$.approvals.resources")
    if approval.get("scope_hash") != scope.scope_hash:
        raise V5Error("GATE_BINDING_MISMATCH", "$.approvals.resources.scope_hash")
    if approval.get("smoke_plan_sha256") != plan_sha256:
        raise V5Error("SMOKE_PLAN_NOT_APPROVED", "$.approvals.resources.smoke_plan_sha256")
    if not fixture_mode and approval["approved_by"] in FIXTURE_APPROVERS:
        raise V5Error("GATE_FIXTURE_NOT_ALLOWED", "$.approvals.resources")
    return approval


def choose_candidate_v5(rows: list, rule: dict, device_memory_bytes: int | None) -> dict | None:
    """first_ok_within_memory: the first listed candidate whose full training step ran (status OK) with a peak at
    most max_memory_fraction of the device memory (no cap off CUDA). None when no candidate qualifies."""
    if rule.get("name") != "first_ok_within_memory":
        raise V5Error("INVALID_SMOKE_PLAN", "$.candidate_rule.name")
    for row in rows:
        if row["status"] != "OK":
            continue
        if device_memory_bytes is not None and row["peak_memory_bytes"] is not None \
                and row["peak_memory_bytes"] > rule["max_memory_fraction"] * device_memory_bytes:
            continue
        return {key: copy.deepcopy(row[key]) for key in CANDIDATE_KEYS}
    return None


# ---------------------------------------------------------------------------
# Batches


def synthetic_make_batch_v5(scope: ActiveScope, tasks: tuple, seed: int):
    """make_batch(candidate, device) of random tensors of the candidate's shape, for measurement before any data
    is read: memory and time depend on shapes, not on content. Every voxel is supervised."""
    def make(candidate, device):
        generator = np.random.default_rng(seed)
        patch = tuple(candidate["patch_size"])
        grid = {"shape": list(patch), "axis_order": AXIS_ORDER, "spacing_mm": None, "affine": None,
                "units_reliable": False, "transform_chain": []}
        items = []
        for _ in range(candidate["batch_size"]):
            channels = {name: generator.standard_normal(patch).astype(np.float32) for name in scope.input_modalities}
            items.append(build_model_input_v5(channels, scope, np.ones(patch, dtype=bool), grid, []))
        size = candidate["batch_size"]
        targets, eligibility = {}, {}
        for task in sorted(tasks):
            classes = len(scope.anatomy_classes if task == "A" else scope.abnormality_classes)
            if task == "A":
                targets[task] = torch.from_numpy(generator.integers(0, classes + 1, (size, *patch)).astype(np.int64))
            else:
                targets[task] = torch.from_numpy(generator.random((size, classes, *patch)) < 0.05)
            full = torch.ones((size, classes, *patch), dtype=torch.bool)
            eligibility[task] = {"e_pos": full, "e_neg": full.clone()}
        return _to_device(collate_model_inputs_v5(items), targets, eligibility, torch.device(device))
    return make


def real_make_batch_v5(samples: dict, recipe: dict, scope: ActiveScope):
    """make_batch(candidate, device) from the subset's training samples with a fresh seeded sampler."""
    def make(candidate, device):
        merged = {**recipe, **{key: candidate[key] for key in CANDIDATE_KEYS}}
        sampler = PatchSamplerV5(sorted(samples), {uid: positive_centers_v5(s) for uid, s in samples.items()},
                                 merged["sampling"]["positive_fraction"], merged["seeds"]["sampler"])
        return _to_device(*make_batch_v5(samples, sampler, merged, scope), torch.device(device))
    return make


def real_batch_report_v5(samples: dict, records: dict, scope: ActiveScope, recipe: dict) -> dict:
    """Counts of what training reads, on the subset and on one batch of the given recipe; no identifiers."""
    order, grid_is_original, shapes, fov, labels, visibility = True, True, [], [], {}, {}
    channel_stats = {name: [] for name in scope.input_modalities}
    support = {}
    for uid in sorted(samples):
        sample, record = samples[uid], records[uid]
        observation = sample.observation
        order &= list(observation.channels) == list(scope.input_modalities)
        grid_is_original &= not observation.to_original and tuple(observation.grid["shape"]) == \
            tuple(observation.original_shape)
        shapes.append(list(observation.original_shape))
        fov.append(float(observation.valid_fov.mean()))
        for name, array in observation.channels.items():
            inside = array[observation.valid_fov]
            channel_stats[name].append((float(inside.mean()), float(inside.std()), bool(np.isfinite(array).all())))
        for annotation in record.annotation_records:
            labels[annotation.label_map_version] = labels.get(annotation.label_map_version, 0) + 1
            key = f"{annotation.task}:{annotation.input_visibility}"
            visibility[key] = visibility.get(key, 0) + 1
        for task in sorted(sample.targets):
            entry = support.setdefault(task, {"positive_voxels": 0, "e_pos_voxels": 0, "e_neg_voxels": 0})
            target = sample.targets[task]
            entry["positive_voxels"] += int((target > 0).sum()) if task == "A" else int(target.sum())
            entry["e_pos_voxels"] += int(sample.e_pos[task].sum())
            entry["e_neg_voxels"] += int(sample.e_neg[task].sum())
    batch, targets, eligibility = real_make_batch_v5(samples, recipe, scope)(
        {key: recipe[key] for key in CANDIDATE_KEYS}, "cpu")
    batch_support = {task: {"positive_voxels": int((targets[task] > 0).sum()) if task == "A"
                            else int(targets[task].sum()),
                            "supervised_voxels": int((eligibility[task]["e_pos"] | eligibility[task]["e_neg"]).sum())}
                     for task in sorted(targets)}
    return {"samples": len(samples), "channels_in_scope_order": bool(order),
            "input_modalities": list(scope.input_modalities), "model_grid_is_original": bool(grid_is_original),
            "original_shapes": sorted(shapes), "fov_fraction_mean": float(np.mean(fov)),
            "channel_mean_std_finite": {name: [round(sum(v[0] for v in vals) / len(vals), 4),
                                               round(sum(v[1] for v in vals) / len(vals), 4),
                                               all(v[2] for v in vals)] for name, vals in channel_stats.items()},
            "label_maps": dict(sorted(labels.items())), "input_visibility": dict(sorted(visibility.items())),
            "support": support,
            "batch": {"image_shape": list(batch.image.shape), "finite": bool(torch.isfinite(batch.image).all()),
                      "fov_fraction": float(batch.valid_fov.float().mean()), "support": batch_support}}


def batch_check_v5(plan: dict, scope: ActiveScope, records: list, registry: dict, raw: Path,
                   label_maps: dict) -> dict:
    """Before any run is registered: load the plan's subset and report one real batch (CPU; no model, no
    training, so no exposure). Counts only; the subset rule is the one the smoke will use."""
    uids = select_subset_v5(records, registry, count=plan["subset"]["count"], salt=plan["subset"]["salt"],
                            strata=plan["subset"]["strata"])
    by_uid = {record.sample_uid: record for record in records}
    recipe = recipe_for_candidate_v5(plan, plan["candidates"][0])
    samples = {uid: load_training_sample_v5(raw, by_uid[uid], scope, plan["supervision"], recipe, label_maps,
                                            {"review": {}, "visibility": {}}) for uid in uids}
    return real_batch_report_v5(samples, by_uid, scope, recipe)


# ---------------------------------------------------------------------------
# The smoke


def _read_steps(run_dir: Path) -> list:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line]


def _losses(lines: list) -> list:
    return [line["loss"] for line in sorted((x for x in lines if x.get("event") == "step"), key=lambda x: x["step"])]


def _dice(pred: np.ndarray, truth: np.ndarray):
    total = int(pred.sum()) + int(truth.sum())
    return None if total == 0 else 2.0 * int((pred & truth).sum()) / total


def _inference_check(output, record_path: Path, sample, scope: ActiveScope, reloaded) -> dict:
    """Shape, acceptance and Dice of the saved original-grid output against the sample's released labels."""
    original = tuple(sample.observation.original_shape)
    dice = []
    if scope.anatomy_classes:
        labels = np.asarray(output.anatomy_map)
        for k in range(len(scope.anatomy_classes)):
            dice.append(_dice(labels == k + 1, sample.targets["A"] == k + 1))
        shape, accepted = labels.shape, np.asarray(output.acceptance["anatomy"])
    else:
        maps = np.asarray(output.abnormality_map)
        for c in range(len(scope.abnormality_classes)):
            dice.append(_dice(maps[c].astype(bool), sample.targets["U"][c].astype(bool)))
        shape, accepted = maps.shape[1:], np.asarray(output.acceptance["abnormality"])
    found = [d for d in dice if d is not None]
    return {"validated": True, "reloaded_by_hash": reloaded is not None,
            "original_shape_matches": tuple(shape) == original, "accepted_fraction": float(accepted.mean()),
            "dice_mean": sum(found) / len(found) if found else None, "dice_classes": len(found),
            "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest()}


def run_smoke_v5(*, plan: dict, plan_sha256: str, scope: ActiveScope, source: dict, records: list, registry: dict,
                 raw: Path, label_maps: dict, gate_decision: dict, approval: dict, private_root: Path, repo: Path,
                 device: str, run_id: str, created_at: str, environment: dict, code: dict,
                 fixture_mode: bool) -> dict:
    """Run the whole smoke for one scope; returns the summary (also written privately as summary.json)."""
    started = time.monotonic()
    validate_smoke_plan_v5(plan, scope, source)
    if smoke_plan_sha256_v5(plan) != plan_sha256:
        raise V5Error("SMOKE_PLAN_HASH_MISMATCH", "$.plan_sha256")
    check_smoke_approval_v5(approval, plan_sha256, scope, fixture_mode)
    private_root = Path(private_root)
    device_t = torch.device(device)
    summary = {"scope_id": scope.scope_id, "plan_id": plan["plan_id"], "plan_sha256": plan_sha256,
               "task": gate_task_v5(scope), "device": {"name": "CPU", "count": 1}, "status": "FAIL",
               "stop_code": None, "stop_phase": None, "subset": None, "batch_check": None, "candidates": None,
               "chosen": None, "run": None, "continuous": None, "overfit": None, "resumed": None,
               "resume_check": None, "inference": None, "elapsed_hours": None,
               "budget_hours": plan["budget"]["max_wall_hours"]}
    if device_t.type == "cuda":
        summary["device"] = {"name": torch.cuda.get_device_name(device_t), "count": 1}
    budget_seconds = plan["budget"]["max_wall_hours"] * 3600.0

    def finish(stop_code: str | None = None, phase: str | None = None) -> dict:
        summary["stop_code"], summary["stop_phase"] = stop_code, phase
        summary["elapsed_hours"] = (time.monotonic() - started) / 3600.0
        write_private_json(resolve_private_output(private_root / "summary.json", private_root, repo), summary)
        return summary

    def over_budget() -> bool:
        return time.monotonic() - started > budget_seconds

    # 1. subset and its samples
    subset = {"rule": "hash_rank_v1", **copy.deepcopy(plan["subset"])}
    subset["sample_uids"] = select_subset_v5(records, registry, count=subset["count"], salt=subset["salt"],
                                             strata=subset["strata"])
    subset = {key: subset[key] for key in ("rule", "salt", "strata", "count", "sample_uids")}
    summary["subset"] = {key: subset[key] for key in ("rule", "salt", "strata", "count")}
    by_uid = {record.sample_uid: record for record in records}
    base = recipe_for_candidate_v5(plan, plan["candidates"][0])
    samples = {uid: load_training_sample_v5(raw, by_uid[uid], scope, plan["supervision"], base, label_maps,
                                            {"review": {}, "visibility": {}})
               for uid in subset["sample_uids"]}
    # 2. real batch
    summary["batch_check"] = real_batch_report_v5(samples, by_uid, scope, base)
    if over_budget():
        return finish("SMOKE_BUDGET_EXCEEDED", "batch_check")
    # 3. measurement
    rows = measure_candidates_v5(lambda c: build_model_v5(recipe_for_candidate_v5(plan, c), scope),
                                 real_make_batch_v5(samples, base, scope), copy.deepcopy(plan["candidates"]), device,
                                 recipe=base, supervision=plan["supervision"])
    summary["candidates"] = rows
    memory = torch.cuda.get_device_properties(device_t).total_memory if device_t.type == "cuda" else None
    chosen = choose_candidate_v5(rows, plan["candidate_rule"], memory)
    summary["chosen"] = chosen
    if chosen is None:
        return finish("NO_USABLE_CANDIDATE", "measurement")
    if over_budget():
        return finish("SMOKE_BUDGET_EXCEEDED", "measurement")
    # 4. one registered smoke run, two executions
    recipe = recipe_for_candidate_v5(plan, chosen)
    run = register_run_v5(run_id=run_id, lineage_id=plan["lineage_id"], run_kind="smoke", recipe=recipe,
                          supervision=plan["supervision"], scope=scope, source=source, split_registry=registry,
                          records=records, gate_decisions={"G-DATA": gate_decision},
                          approvals={"resources": approval}, parents=[], environment=environment, code=code,
                          created_at=created_at, fixture_mode=fixture_mode, sample_subset=subset)
    summary["run"] = {"run_id": run_id, "run_sha256": run_sha256_v5(run), "recipe_sha256": run["recipe_sha256"],
                      "supervision_sha256": run["supervision_sha256"],
                      "data_manifest_sha256": run["data_manifest_sha256"], "split_sha256": run["split"]["split_sha256"]}
    roots = {name: private_root / name for name in ("continuous", "resumed")}
    dirs = {name: write_run_v5(run, registry, root, repo) for name, root in roots.items()}
    if plan["resume"]["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_model_v5(recipe, scope).to(device_t)
    result = train_v5(run, model, samples, recipe, plan["supervision"], dirs["continuous"], roots["continuous"], repo,
                      None)
    summary["continuous"] = {key: result[key] for key in ("status", "step", "failure_code")}
    summary["continuous"]["checkpoint_sha256"] = result["last_checkpoint_sha256"]
    if result["status"] != "COMPLETED":
        return finish(result["failure_code"], "continuous")
    # 5. overfit judgement of the continuous model
    lines_c = _read_steps(dirs["continuous"])
    summary["overfit"] = {"steps": recipe["budget"]["max_steps"],
                          **judge_overfit_v5(model, samples, recipe, _losses(lines_c), plan["overfit"]["criteria"],
                                             failure_code=None)}
    if over_budget():
        return finish("SMOKE_BUDGET_EXCEEDED", "overfit")
    # 6. pause, resume with a fresh model, compare
    pause_at = plan["resume"]["pause_at"]
    paused = train_v5(run, build_model_v5(recipe, scope).to(device_t), samples, recipe, plan["supervision"],
                      dirs["resumed"], roots["resumed"], repo, None, pause_at=pause_at)
    summary["resumed"] = {"pause_status": paused["status"], "pause_step": paused["step"]}
    if paused["status"] != "PAUSED":
        return finish(paused["failure_code"], "resumed")
    fresh = build_model_v5(recipe, scope)
    with torch.no_grad():                               # other weights: only the checkpoint can restore them
        for parameter in fresh.parameters():
            parameter.add_(1.0)
    torch.manual_seed(0)                                # other RNG state: only the checkpoint can restore it
    again = train_v5(run, fresh.to(device_t), samples, recipe, plan["supervision"], dirs["resumed"], roots["resumed"],
                     repo, (paused["last_checkpoint"], paused["last_checkpoint_sha256"]))
    summary["resumed"].update({key: again[key] for key in ("status", "step", "failure_code")},
                              checkpoint_sha256=again["last_checkpoint_sha256"])
    if again["status"] != "COMPLETED":
        return finish(again["failure_code"], "resumed")
    final_c = load_checkpoint_v5(result["last_checkpoint"], run, result["last_checkpoint_sha256"], "cpu")
    final_r = load_checkpoint_v5(again["last_checkpoint"], run, again["last_checkpoint_sha256"], "cpu")
    summary["resume_check"] = compare_resume_v5(final_c, final_r, lines_c, _read_steps(dirs["resumed"]),
                                                pause_at=pause_at, tolerances=plan["resume"]["tolerances"])
    if over_budget():
        return finish("SMOKE_BUDGET_EXCEEDED", "resume_check")
    # 7. one full-volume inference through the formal entry
    uid = subset["sample_uids"][0]
    record = by_uid[uid]
    observation = load_observation_v5(raw, record, scope, recipe)
    batch = build_model_input_v5(observation.channels, scope, observation.valid_fov, observation.grid,
                                 list(observation.to_original))
    output, record_path = infer_and_save_v5(
        model, batch, scope, observation_uid=uid, checkpoint_sha256=result["last_checkpoint_sha256"],
        original_geometry=original_geometry_v5(observation, record), acceptance_policy=plan["acceptance_policy"],
        window=tuple(recipe["inference"]["window"]), stride=tuple(recipe["inference"]["stride"]),
        precision=recipe["precision"], out_dir=roots["continuous"] / "inference", private_root=roots["continuous"],
        repo=repo, run=run, registry=registry)
    summary["inference"] = _inference_check(output, record_path, samples[uid], scope,
                                            load_output_v5(record_path, scope))
    passed = (summary["batch_check"]["channels_in_scope_order"] and summary["overfit"]["passed"]
              and summary["resume_check"]["passed"] and summary["inference"]["original_shape_matches"])
    summary["status"] = "PASS" if passed else "FAIL"
    return finish(None, None)


# ---------------------------------------------------------------------------
# Redacted report and training_smoke evidence


def _yes(value: object, path: str) -> str:
    if not isinstance(value, bool):
        raise V5Error("INVALID_SUMMARY", path)
    return "是" if value else "否"


def render_smoke_md_v5(summary: dict) -> str:
    """The measurement report plus the smoke sections; shows counts, configuration, metrics and hashes only."""
    if summary.get("status") not in SMOKE_STATUS:
        raise V5Error("INVALID_SUMMARY", "$.status")
    if summary.get("candidates"):
        text = render_measurement_md_v5({"scope_id": summary["scope_id"], "device": summary["device"],
                                         "candidates": summary["candidates"], "overfit": summary["overfit"]})
    else:
        text = f"# GPU 实测：{_text(summary.get('scope_id'), '$.scope_id')}\n\n- 候选测量未运行。\n"
    lines = [text.rstrip("\n"), "", "## 冒烟运行", "",
             f"- 计划：`{_text(summary['plan_id'], '$.plan_id')}`，sha256 `{_text(summary['plan_sha256'], '$.plan_sha256')}`",
             f"- 结果：{summary['status']}" + (f"（停止于 {_text(summary['stop_phase'], '$.stop_phase')}："
                                                f"{_text(summary['stop_code'], '$.stop_code')}）"
                                                if summary.get("stop_code") else ""),
             f"- 用时 {_number(summary['elapsed_hours'], '$.elapsed_hours', 3)} h"
             f"（预算 {_number(summary['budget_hours'], '$.budget_hours', 2)} h）"]
    chosen = summary.get("chosen")
    lines.append("- 选定候选：" + ("无" if chosen is None else
                                  f"patch {'×'.join(str(v) for v in chosen['patch_size'])}，batch "
                                  f"{_count(chosen['batch_size'], '$.chosen.batch_size')}，"
                                  f"{_text(chosen['precision'], '$.chosen.precision')}"))
    subset = summary.get("subset")
    if subset:
        strata = "全部" if subset["strata"] is None else "、".join(_text(s, "$.subset.strata") for s in subset["strata"])
        lines.append(f"- 小样本：规则 `{_text(subset['rule'], '$.subset.rule')}`，盐 "
                     f"`{_text(subset['salt'], '$.subset.salt')}`，{_count(subset['count'], '$.subset.count')} 例，"
                     f"层：{strata}（只读 train）")
    batch = summary.get("batch_check")
    if batch:
        lines += ["", "### 真实 batch 核对", "",
                  f"- 通道按 scope 顺序：{_yes(batch['channels_in_scope_order'], '$.batch_check')}"
                  f"（{'、'.join(_text(m, '$.batch_check.input_modalities') for m in batch['input_modalities'])}）",
                  f"- 模型网格即原网格：{_yes(batch['model_grid_is_original'], '$.batch_check')}；"
                  f"FOV 占比均值 {_number(batch['fov_fraction_mean'], '$.batch_check', 3)}",
                  "- 标签映射：" + "；".join(f"{_text(k, '$.batch_check.label_maps')} {_count(v, '$.batch_check')}"
                                         for k, v in batch["label_maps"].items()),
                  "- 输入可见性：" + "；".join(f"{_text(k.replace(':', ' '), '$.batch_check.input_visibility')} "
                                          f"{_count(v, '$.batch_check')}" for k, v in batch["input_visibility"].items()),
                  "- 监督支持（小样本合计）：" + "；".join(
                      f"{_text(t, '$.batch_check.support')} 阳性 {_count(v['positive_voxels'], '$.s')}，"
                      f"e_pos {_count(v['e_pos_voxels'], '$.s')}，e_neg {_count(v['e_neg_voxels'], '$.s')}"
                      for t, v in batch["support"].items()),
                  "- 一个 batch：形状 " + "×".join(str(v) for v in batch["batch"]["image_shape"])
                  + f"，有限值 {_yes(batch['batch']['finite'], '$.batch_check.batch.finite')}，FOV 占比 "
                  f"{_number(batch['batch']['fov_fraction'], '$.batch_check.batch', 3)}；"
                  + "；".join(f"{_text(t, '$.b')} 阳性 {_count(v['positive_voxels'], '$.b')} / 受监督 "
                             f"{_count(v['supervised_voxels'], '$.b')}" for t, v in batch["batch"]["support"].items())]
    run = summary.get("run")
    if run:
        lines += ["", "### 登记与训练", "",
                  f"- run_sha256 `{_text(run['run_sha256'], '$.run')}`；recipe_sha256 `{_text(run['recipe_sha256'], '$.run')}`",
                  f"- supervision_sha256 `{_text(run['supervision_sha256'], '$.run')}`；data_manifest_sha256 "
                  f"`{_text(run['data_manifest_sha256'], '$.run')}`；split_sha256 `{_text(run['split_sha256'], '$.run')}`"]
        cont = summary.get("continuous")
        if cont:
            lines.append(f"- 连续训练：{_text(cont['status'], '$.continuous.status')}，步数 "
                         f"{_count(cont['step'], '$.continuous.step')}")
        resumed = summary.get("resumed")
        if resumed:
            lines.append(f"- 暂停与恢复：暂停 {_text(resumed['pause_status'], '$.resumed')} 于第 "
                         f"{_count(resumed['pause_step'], '$.resumed')} 步；恢复后 "
                         f"{_text(resumed.get('status') or 'NOT_RUN', '$.resumed.status')}")
    check = summary.get("resume_check")
    if check:
        lines += ["", "### 恢复核对", "",
                  "- 精确一致：" + "；".join(f"{_text(k, '$.resume_check.exact')} {_yes(v, '$.resume_check.exact')}"
                                         for k, v in check["exact"].items()),
                  f"- 比较步数 {_count(check['steps_compared'], '$.resume_check')}；最大损失差 "
                  f"{_number(check['max_loss_abs_diff'], '$.resume_check', 8)}；参数最大相对差 "
                  f"{_number(check['max_param_rel_diff'], '$.resume_check', 8)}；AdamW 矩最大相对差 "
                  f"{_number(check['max_optimizer_rel_diff'], '$.resume_check', 8)}；参数逐位一致 "
                  f"{_yes(check['bitwise_parameters'], '$.resume_check')}",
                  "- 容差：" + "；".join(f"{_text(k, '$.t')} {_number(v, '$.t', 8)}"
                                       for k, v in check["tolerances"].items()),
                  f"- 结果：{'PASS' if check['passed'] else 'FAIL'}"]
    inference = summary.get("inference")
    if inference:
        lines += ["", "### 全体积推理", "",
                  f"- 经 `infer_and_save_v5` 落盘并通过 `validate_module_output_v5`；按哈希重新读取 "
                  f"{_yes(inference['reloaded_by_hash'], '$.inference')}；原网格形状一致 "
                  f"{_yes(inference['original_shape_matches'], '$.inference')}",
                  f"- 接受占比 {_number(inference['accepted_fraction'], '$.inference', 3)}；与发布标签的 Dice 均值 "
                  f"{_number(inference['dice_mean'], '$.inference', 3)}（{_count(inference['dice_classes'], '$.i')} 类，"
                  "训练样本，只作冒烟核对，不是开发集结果）",
                  f"- 输出记录 sha256 `{_text(inference['record_sha256'], '$.inference')}`"]
    lines += ["", "- 只读 train；`scientific_conclusion: NOT_EVALUATED`。"]
    return "\n".join(lines) + "\n"


def training_smoke_evidence_v5(summary: dict, markdown: str, scope: ActiveScope, source: dict) -> dict:
    """G-BASE `training_smoke` evidence: PASS only for a PASS smoke; bound to the report's sha256."""
    return {"status": "PASS" if summary.get("status") == "PASS" else "FAIL",
            "sha256": hashlib.sha256(markdown.encode()).hexdigest(), "scope_hash": scope.scope_hash,
            "task": gate_task_v5(scope), "source_id": source["source_id"], "source_version": source["source_version"]}


# ---------------------------------------------------------------------------
# Command line


def _load_yaml(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def main(argv: list | None = None) -> int:
    import argparse

    from .contracts_v5 import validate_sample_v5, validate_scope_v5
    from .engine.data_v5 import load_label_maps_v5
    from .engine.registration_v5 import capture_code_v5, capture_environment_v5
    from .gates_v5 import evaluate_gate_v5
    from .source_policy_v5 import load_source_policies_v5, source_binding_v5
    from .splits_v5 import verify_split_registry_v5

    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(prog="anatobind.v41.real_t10_v5")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan-hash", "premeasure", "batch-check", "smoke"):
        command = sub.add_parser(name)
        command.add_argument("--plan", required=True)
        command.add_argument("--scope", required=True)
        command.add_argument("--policies", required=True)
        command.add_argument("--source-id", required=True)
        if name in ("premeasure", "batch-check", "smoke"):
            command.add_argument("--out", required=True)
        if name in ("premeasure", "smoke"):
            command.add_argument("--device", required=True)
        if name in ("batch-check", "smoke"):
            for key in ("--raw", "--records", "--split-registry"):
                command.add_argument(key, required=True)
        if name == "smoke":
            for key in ("--gdata-decision", "--approvals", "--run-id"):
                command.add_argument(key, required=True)
    gate = sub.add_parser("gdata-gate")
    for key in ("--evidence", "--approvals", "--scope", "--policies", "--source-id", "--out"):
        gate.add_argument(key, required=True)
    args = parser.parse_args(argv)
    scope = validate_scope_v5(_load_yaml(args.scope))
    source = source_binding_v5(load_source_policies_v5(Path(args.policies))[args.source_id])

    if args.command == "gdata-gate":
        evidence = json.loads(Path(args.evidence).read_text())
        approvals = (_load_yaml(args.approvals) or {}).get(scope.scope_id, {}).get("G-DATA", {})
        decision = evaluate_gate_v5("G-DATA", scope, evidence, approvals, source)
        out = Path(args.out)
        out.mkdir(mode=0o700)                                            # FileExistsError: never overwritten
        write_private_json(out / "gdata_decision.json", decision)
        print(json.dumps({k: decision[k] for k in ("gate_id", "scope_id", "engineering_status", "blockers")}))
        return 0

    plan = validate_smoke_plan_v5(_load_yaml(args.plan), scope, source)
    digest = smoke_plan_sha256_v5(plan)
    if args.command == "plan-hash":
        print(digest)
        return 0
    out = Path(args.out)
    out.mkdir(mode=0o700)                                                # never overwritten
    os.chmod(out, 0o700)
    if args.command == "premeasure":
        tasks = tuple(plan["recipe"]["model"]["tasks"])
        rows = measure_candidates_v5(lambda c: build_model_v5(recipe_for_candidate_v5(plan, c), scope),
                                     synthetic_make_batch_v5(scope, tasks, seed=0), copy.deepcopy(plan["candidates"]),
                                     args.device, recipe=recipe_for_candidate_v5(plan, plan["candidates"][0]),
                                     supervision=plan["supervision"])
        device = torch.device(args.device)
        name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        result = {"scope_id": scope.scope_id, "plan_sha256": digest, "device": {"name": name, "count": 1},
                  "candidates": rows, "input": "synthetic_random_tensors"}
        write_private_json(out / "premeasure.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0

    records = [validate_sample_v5(json.loads(line)) for line in Path(args.records).read_text().splitlines() if line]
    registry = verify_split_registry_v5(json.loads(Path(args.split_registry).read_text()))
    label_maps = load_label_maps_v5(repo / "configs" / "v5" / "label_maps_v5.yaml")
    if args.command == "batch-check":
        report = batch_check_v5(plan, scope, records, registry, Path(args.raw), label_maps)
        write_private_json(out / "batch_check.json", report)
        print(json.dumps(report, sort_keys=True))
        return 0
    decision = json.loads(Path(args.gdata_decision).read_text())
    approval = (_load_yaml(args.approvals) or {}).get(scope.scope_id, {}).get("resources")
    summary = run_smoke_v5(plan=plan, plan_sha256=digest, scope=scope, source=source, records=records,
                           registry=registry, raw=Path(args.raw), label_maps=label_maps,
                           gate_decision=decision, approval=approval, private_root=out, repo=repo,
                           device=args.device, run_id=args.run_id,
                           created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           environment=capture_environment_v5(), code=capture_code_v5(repo), fixture_mode=False)
    markdown = render_smoke_md_v5(summary)
    fd = os.open(out / "gpu_measurement.md", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(markdown)
    write_private_json(out / "training_smoke_evidence.json", training_smoke_evidence_v5(summary, markdown, scope, source))
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
