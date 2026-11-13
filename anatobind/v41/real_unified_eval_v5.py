"""Evaluate-only entry and sampling replay of a FINISHED unified run (diagnostics of 2026-11-11).

Both read a completed run of real_unified_v5 and never train: no optimizer is built, every forward pass runs in
inference mode, and the weights' hash is compared before and after (a difference is an error and nothing is
reported). The checkpoint is the run's final one, loaded by the hash the run's own summary recorded — or, for the
evaluation, a checkpoint of the run selected by step AND hash (2026-11-12): its sidecar must name that step, that
hash and this run, the file must hash to it and hold that step. Nothing of the run is rewritten for it; the
summary of the evaluation says which checkpoint was read and which one is the run's final one.

What must hold before any data is read: the roles asked for are train / development_val (locked_test is refused),
the code is clean, the plan and the approvals are those of the run, the inputs register exactly the run stored on
disk, the run completed, the wall budget asked for fits in what the plan's overall budget has left, and the
destination is a new directory outside the run. Samples are read through the frozen split by role, under the run's
own registered member run: the same samples the run already recorded as exposed, so no new exposure arises.

`run_unified_evaluation_v5` scores a box-supervised member on each role with the training driver's own steps
(run_perception_v5 → findings_from_output_v5 → score_findings_v5 / anatomy_dice_v5), each role against all of its
own primary boxes, and adds the diagnostics of evaluation.findings_diagnostics_v5. For development_val the outputs
the run saved are loaded and used as the official ones; the new inference only supplies the probabilities the
saved outputs do not hold, and is compared with them array by array. A saved output is reused only when it
records the hash of the checkpoint being evaluated: for a checkpoint other than the run's final one reuse is
switched off, and `reuse_inference=None` asks for new predictions throughout. With `module_development` every
voxel-supervised member is scored on development_val with the training driver's own steps (run_perception_v5 →
module_metrics_v5 → summarize_module_metrics_v5).

`run_unified_replay_v5` redraws the member's training patches (engine.replay_v5), checks every step against the
run's log and the final sampler state, reports how often the boxes were shown, and recomputes the two parts of the
box objective where the weights are known: the seeded initialisation, each checkpoint and the end. Two things come
of that. History: the batch of the step right after such a moment, whose logged loss the recomputed sum must
reproduce. Trajectory: every such moment on one fixed set of batches (the first loss_steps batches of the training
replay, and as many drawn the same way from development_val), newly computed and labelled as a diagnostic.

Private rows stay under the new directory; the renderers show configuration, counts, metrics and hashes only.
scientific_conclusion stays NOT_EVALUATED; values on train are diagnostics, never a generalisation result.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import real_unified_v5 as driver
from .contracts_v5 import V5Error
from .engine.data_v5 import make_batch_v5, original_geometry_v5
from .engine.module_eval_v5 import freeze_module_reference_v5, module_metrics_v5, summarize_module_metrics_v5
from .engine.registration_v5 import READABLE_ROLES, _is_sha256, _positive_int, readable_role_v5, run_sha256_v5
from .engine.replay_v5 import batch_exposure_v5, box_objective_parts_v5, full_bag_voxels_v5, member_sampler_v5
from .engine.train_v5 import _append_jsonl, _to_device, autocast_v5, load_checkpoint_v5
from .engine.unified_registration_v5 import member_recipe_v5
from .evaluation.findings_diagnostics_v5 import (
    _distribution, diagnose_scan_v5, render_diagnostics_md_v5, summarize_diagnostics_v5)
from .evaluation.findings_v5 import OUTSIDE_NOTE, anatomy_dice_v5, score_findings_v5, summarize_findings_v5
from .findings_v5 import findings_from_output_v5, host_relation_fn_v5, render_findings_zh_v5
from .models.interface_v5 import build_model_input_v5
from .models.unified_v5 import build_unified_model_v5
from .paths import resolve_private_output, write_private_json
from .perception import run_perception_v5
from .perception_io import load_output_v5, output_prefix_v5

CONCLUSION = "NOT_EVALUATED"
_BUDGET_CHECK_EVERY = 100


def state_sha256_v5(model: torch.nn.Module) -> str:
    """One hash over every tensor of the model's state, keys sorted."""
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _check_roles(roles: object) -> list:
    if not isinstance(roles, list) or not roles:
        raise V5Error("INVALID_ROLE", "$.roles")
    for index, role in enumerate(roles):
        if role == "locked_test":
            raise V5Error("LOCKED_TEST_NOT_READABLE", f"$.roles[{index}]")
        if role not in READABLE_ROLES or role in roles[:index]:
            raise V5Error("INVALID_ROLE", f"$.roles[{index}]")
    return list(roles)


def _open_finished_run(*, plan: dict, plan_sha256: str, protocols: dict, members: dict, approvals: dict,
                       run_root: Path, run_id: str, private_root: Path, max_wall_hours: float, code: dict,
                       fixture_mode: bool, checkpoint: tuple | None = None) -> dict:
    """Everything that must hold before any data is read; returns the run, its checkpoint and the budget left.

    `checkpoint` is None for the run's final checkpoint, or (step, sha256) for a checkpoint of the run: its sidecar
    must name that step, that hash and this run."""
    if not (isinstance(code, dict) and set(code) == {"git_commit", "dirty"} and isinstance(code["dirty"], bool)):
        raise V5Error("INVALID_TYPE", "$.code")
    if checkpoint is not None and not (isinstance(checkpoint, tuple) and len(checkpoint) == 2
                                       and _positive_int(checkpoint[0]) and _is_sha256(checkpoint[1])):
        raise V5Error("EVAL_CHECKPOINT_INVALID", "$.checkpoint")
    if code["dirty"] and not fixture_mode:
        raise V5Error("CODE_DIRTY", "$.code.dirty")
    pids = driver._precheck(plan, plan_sha256, protocols, members, approvals, fixture_mode)
    run_root, out = Path(run_root).resolve(), Path(private_root).resolve()
    if out == run_root or out.is_relative_to(run_root) or run_root.is_relative_to(out):
        raise V5Error("EVAL_OUT_INSIDE_RUN", "$.out")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise V5Error("EVAL_OUT_EXISTS", "$.out")
    try:
        stored = json.loads((run_root / "runs" / run_id / "run.json").read_text())
    except (OSError, ValueError):
        raise V5Error("EVAL_RUN_MISSING", "$.run") from None
    if not isinstance(stored, dict) or not all(key in stored for key in ("environment", "code", "created_at")):
        raise V5Error("EVAL_RUN_MISSING", "$.run")
    unified, member_runs = driver._register(
        plan=plan, plan_sha256=plan_sha256, protocols=protocols, members=members, approvals=approvals, run_id=run_id,
        run_kind="baseline", subsets=None, environment=stored["environment"], code=stored["code"],
        created_at=stored["created_at"], fixture_mode=fixture_mode)
    if unified != stored:
        raise V5Error("EVAL_RUN_MISMATCH", "$.run")
    try:
        summary = json.loads((run_root / "summary.json").read_text())
    except (OSError, ValueError):
        raise V5Error("EVAL_RUN_NOT_COMPLETED", "$.run") from None
    training = summary.get("training") if isinstance(summary, dict) else None
    if not isinstance(training, dict) or summary.get("status") != "COMPLETED" or training.get("status") != "COMPLETED" \
            or summary.get("plan_sha256") != plan_sha256 \
            or (summary.get("run") or {}).get("run_sha256") != run_sha256_v5(unified):
        raise V5Error("EVAL_RUN_NOT_COMPLETED", "$.run")
    remaining = float(plan["budget"]["max_wall_hours"]) - float(summary["elapsed_hours"])
    if not (isinstance(max_wall_hours, (int, float)) and not isinstance(max_wall_hours, bool)
            and 0 < max_wall_hours <= remaining):
        raise V5Error("EVAL_BUDGET_INVALID", "$.max_wall_hours")
    final = (int(training["step"]), training["checkpoint_sha256"])
    step, digest = final if checkpoint is None else checkpoint
    directory = run_root / "runs" / run_id / "checkpoints"
    if checkpoint is not None:
        try:
            sidecar = json.loads((directory / f"step_{step:08d}.pt.json").read_text())
        except (OSError, ValueError):
            raise V5Error("EVAL_CHECKPOINT_MISSING", "$.checkpoint") from None
        if not isinstance(sidecar, dict) or sidecar.get("step") != step or sidecar.get("sha256") != digest \
                or sidecar.get("run_sha256") != run_sha256_v5(unified):
            raise V5Error("EVAL_CHECKPOINT_MISMATCH", "$.checkpoint")
    return {"pids": pids, "unified": unified, "member_runs": member_runs, "summary": summary, "run_root": run_root,
            "checkpoint": (directory / f"step_{step:08d}.pt", digest, step),
            "selection": "final" if checkpoint is None else "explicit", "final_checkpoint": final,
            "remaining_hours": remaining}


def _load_selected(opened: dict, device: str) -> dict:
    """The state of the opened checkpoint: the file hashes to the selected hash, belongs to the run, holds the step."""
    path, digest, step = opened["checkpoint"]
    state = load_checkpoint_v5(path, opened["unified"], digest, device)
    if state["step"] != step:
        raise V5Error("EVAL_CHECKPOINT_MISMATCH", "$.checkpoint.step")
    return state


def _box_members(plan: dict, protocols: dict, pids: list, protocol_ids: list | None) -> list:
    wanted = [pid for pid in pids if plan["members"][pid]["box_supervision"] is not None] \
        if protocol_ids is None else list(protocol_ids)
    for pid in wanted:
        if pid not in pids or plan["members"][pid]["box_supervision"] is None \
                or protocols[pid]["scope"].scope_kind != "joint" or protocols[pid]["groups"] is None \
                or protocols[pid]["groups_by_class"] is None:
            raise V5Error("EVAL_MEMBER_NOT_SUPPORTED", "$.protocols")
    if not wanted:
        raise V5Error("EVAL_MEMBER_NOT_SUPPORTED", "$.protocols")
    return wanted


def _frozen_model(plan: dict, scopes: dict, state: dict, device: torch.device) -> torch.nn.Module:
    model = build_unified_model_v5(plan, scopes).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _header(plan: dict, plan_sha256: str, opened: dict, scopes: dict, kind: str, device: torch.device,
            state_sha256: str, max_wall_hours: float, details: dict) -> dict:
    unified, pids = opened["unified"], opened["pids"]
    return {"kind": kind, "plan_id": plan["plan_id"], "plan_sha256": plan_sha256,
            "device": driver._device_summary(device),
            "run": {"run_id": unified["run_id"], "run_sha256": run_sha256_v5(unified), "code": dict(unified["code"]),
                    "members": {pid: {"scope_id": scopes[pid].scope_id, "scope_hash": scopes[pid].scope_hash,
                                      "run_sha256": unified["members"][pid]["run_sha256"],
                                      "split_sha256": unified["members"][pid]["split_sha256"]} for pid in pids}},
            "checkpoint": {"sha256": opened["checkpoint"][1], "step": opened["checkpoint"][2],
                           "state_sha256": state_sha256, "weights_unchanged": None,
                           "selection": opened["selection"],
                           "is_final": opened["checkpoint"][1] == opened["final_checkpoint"][1],
                           "run_final": {"step": opened["final_checkpoint"][0],
                                         "sha256": opened["final_checkpoint"][1]}},
            "evaluation": details, "status": "FAILED", "stop_code": None, "stop_phase": None, "members": None,
            "elapsed_hours": None, "budget_hours": float(max_wall_hours),
            "plan_budget_remaining_hours": opened["remaining_hours"], "scientific_conclusion": CONCLUSION}


def _same_output(new, saved) -> bool:
    arrays = [("anatomy_map", None), ("abnormality_map", None), ("tissue_state_map", None),
              ("acceptance", "anatomy"), ("acceptance", "abnormality")]
    for name, key in arrays:
        a, b = getattr(new, name), getattr(saved, name)
        if key is not None:
            a, b = a[key], b[key]
        if not np.array_equal(a, b):
            return False
    return [(i["class_name"], i["bbox"], i["candidate_set_hash"]) for i in new.abnormal_instances] \
        == [(i["class_name"], i["bbox"], i["candidate_set_hash"]) for i in saved.abnormal_instances]


# ---------------------------------------------------------------------------
# Evaluation


def run_unified_evaluation_v5(*, plan: dict, plan_sha256: str, protocols: dict, members: dict, approvals: dict,
                              raw: Path, label_maps: dict, run_root: Path, private_root: Path, repo: Path,
                              device: str, run_id: str, roles: list, max_wall_hours: float, created_at: str,
                              environment: dict, code: dict, fixture_mode: bool, protocol_ids: list | None = None,
                              reuse_inference: str | None = "inference", checkpoint: tuple | None = None,
                              module_development: bool = False) -> dict:
    """Score the finished run `run_id` (under `run_root`) on `roles` with its final checkpoint, or with the
    checkpoint `(step, sha256)` of that run; summary.json and private rows go to the new directory `private_root`.
    `module_development` adds every voxel-supervised member on development_val. Returns the summary."""
    started = time.monotonic()
    roles = _check_roles(roles)
    opened = _open_finished_run(plan=plan, plan_sha256=plan_sha256, protocols=protocols, members=members,
                                approvals=approvals, run_root=run_root, run_id=run_id, private_root=private_root,
                                max_wall_hours=max_wall_hours, code=code, fixture_mode=fixture_mode,
                                checkpoint=checkpoint)
    private_root, device_t, pids = Path(private_root), torch.device(device), opened["pids"]
    scopes = {pid: protocols[pid]["scope"] for pid in pids}
    wanted = _box_members(plan, protocols, pids, protocol_ids)
    digest = opened["checkpoint"][1]
    final = digest == opened["final_checkpoint"][1]
    refused = None
    if reuse_inference is not None and not final:      # what the run saved are outputs of its final checkpoint
        reuse_inference, refused = None, "NOT_THE_FINAL_CHECKPOINT"
    model = _frozen_model(plan, scopes, _load_selected(opened, device_t.type), device_t)
    before = state_sha256_v5(model)
    summary = _header(plan, plan_sha256, opened, scopes, "unified_evaluation", device_t, before, max_wall_hours,
                      {"roles": roles, "code": dict(code), "environment": dict(environment),
                       "created_at": created_at, "reuse_inference": reuse_inference, "reuse_refused": refused,
                       "module_development": bool(module_development)})

    def finish(stop_code: str | None, phase: str | None) -> dict:
        summary["stop_code"], summary["stop_phase"] = stop_code, phase
        summary["elapsed_hours"] = (time.monotonic() - started) / 3600.0
        write_private_json(resolve_private_output(private_root / "summary.json", private_root, repo), summary)
        return summary

    originals = opened["summary"].get("members") or {}
    results = {}
    # voxel-supervised members first (development_val only: the module metrics are defined on that split)
    for pid in [pid for pid in pids if module_development and plan["members"][pid]["box_supervision"] is None]:
        member, scope, protocol = plan["members"][pid], scopes[pid], protocols[pid]
        member_run, registry = opened["member_runs"][pid], members[pid]["registry"]
        samples, _ = driver.load_member_samples_v5(
            plan=plan, protocol_id=pid, protocol=protocol, records=members[pid]["records"], registry=registry,
            raw=raw, label_maps=label_maps, roles=["development_val"])
        by_uid = {record.sample_uid: record for record in members[pid]["records"]}
        frozen = freeze_module_reference_v5(      # references are fixed before any prediction
            {uid: driver._reference(sample, by_uid[uid], scope) for uid, sample in samples.items()}, scope, registry)
        rows_path = resolve_private_output(private_root / "rows" / f"development_val.{pid}.jsonl", private_root, repo)
        rows, sentences = [], Counter()
        for uid in sorted(samples):
            if time.monotonic() - started > max_wall_hours * 3600.0:        # no partial result
                return finish("EVAL_BUDGET_EXCEEDED", "development_val")
            if readable_role_v5(uid, registry, member_run) != "development_val":
                raise V5Error("ROLE_NOT_READABLE", "$.sample_uid")
            observation = samples.pop(uid).observation
            batch = build_model_input_v5(observation.channels, scope, observation.valid_fov, observation.grid,
                                         list(observation.to_original))
            output = run_perception_v5(
                model.view(pid), batch, scope, observation_uid=uid, checkpoint_sha256=digest,
                original_geometry=original_geometry_v5(observation, by_uid[uid]),
                acceptance_policy=member["acceptance_policy"], window=tuple(member["inference"]["window"]),
                stride=tuple(member["inference"]["stride"]), precision=plan["shared"]["precision"])
            findings = findings_from_output_v5(output, scope, member["acceptance_policy"], protocol["groups"])
            sentences.update(render_findings_zh_v5(findings, scope, protocol["display"]))
            row = module_metrics_v5(output, frozen, member["metric_policy"], member["acceptance_policy"], member_run)
            _append_jsonl(rows_path, row)
            rows.append(row)
        entry = {"n_scans": len(rows), "module": summarize_module_metrics_v5(rows),
                 "sentences": dict(sorted(sentences.items())), "reproduces_original": None}
        if final and isinstance(originals.get(pid), dict):
            entry["reproduces_original"] = {key: entry[key] == originals[pid].get(key)
                                            for key in ("module", "sentences")}
        results[pid] = {"development_val": entry}
    results.update({pid: {} for pid in wanted})
    for role in roles:
        for pid in wanted:
            member, scope, protocol = plan["members"][pid], scopes[pid], protocols[pid]
            member_run, registry = opened["member_runs"][pid], members[pid]["registry"]
            samples, references = driver.load_member_samples_v5(
                plan=plan, protocol_id=pid, protocol=protocol, records=members[pid]["records"], registry=registry,
                raw=raw, label_maps=label_maps, roles=[role], box_references=members[pid].get("box_references"))
            by_uid = {record.sample_uid: record for record in members[pid]["records"]}
            relation_fn = host_relation_fn_v5(protocol["groups"], protocol["groups_by_class"])
            reuse_dir = None if reuse_inference is None or role != "development_val" \
                else opened["run_root"] / reuse_inference / pid
            rows_path = resolve_private_output(private_root / "rows" / f"{role}.{pid}.jsonl", private_root, repo)
            official_rows, diagnostic_rows, sentences = [], [], Counter()
            dice = {name: {} for name in scope.anatomy_classes}
            reuse = {"scans": 0, "reused": 0, "identical": 0, "other_checkpoint": 0}
            for uid in sorted(samples):
                if time.monotonic() - started > max_wall_hours * 3600.0:    # no partial result
                    return finish("EVAL_BUDGET_EXCEEDED", role)
                if readable_role_v5(uid, registry, member_run) != role:
                    raise V5Error("ROLE_NOT_READABLE", "$.sample_uid")
                sample, record = samples.pop(uid), by_uid[uid]
                observation = sample.observation
                batch = build_model_input_v5(observation.channels, scope, observation.valid_fov, observation.grid,
                                             list(observation.to_original))
                seen = {}
                output = run_perception_v5(
                    model.view(pid), batch, scope, observation_uid=uid, checkpoint_sha256=digest,
                    original_geometry=original_geometry_v5(observation, record),
                    acceptance_policy=member["acceptance_policy"], window=tuple(member["inference"]["window"]),
                    stride=tuple(member["inference"]["stride"]), precision=plan["shared"]["precision"],
                    relation_fn=relation_fn, probe=lambda restored: seen.update(abnormality=restored["abnormality"]))
                reuse["scans"] += 1
                reused = identical = False
                saved = None if reuse_dir is None else reuse_dir / f"{output_prefix_v5(uid)}.record.json"
                if saved is not None and saved.is_file():
                    stored_output = load_output_v5(saved, scope)
                    if stored_output.checkpoint_sha256 != digest:      # written by other weights: never reused
                        reuse["other_checkpoint"] += 1
                    else:                                              # the run's own output is the official one
                        reused, identical = True, _same_output(output, stored_output)
                        output = stored_output
                        reuse["reused"] += 1
                        reuse["identical"] += int(identical)
                findings = findings_from_output_v5(output, scope, member["acceptance_policy"], protocol["groups"])
                sentences.update(render_findings_zh_v5(findings, scope, protocol["display"]))
                truth = sample.targets["A"]
                eligible = (sample.e_pos["A"] & sample.e_neg["A"]).all(axis=0)
                for name, value in anatomy_dice_v5(output.anatomy_map, truth, eligible,
                                                   list(scope.anatomy_classes)).items():
                    dice[name].setdefault(record.patient_uid, []).append(value)
                official = score_findings_v5(findings, references[uid], record.patient_uid,
                                             list(scope.abnormality_classes), member["finding_policy"]["iobb_thresholds"])
                diagnostic = diagnose_scan_v5(
                    findings=findings, instances=list(output.abnormal_instances), reference=references[uid],
                    scope=scope, probability=seen["abnormality"], positive=output.abnormality_map,
                    accepted=output.acceptance["abnormality"], anatomy_predicted=output.anatomy_map,
                    anatomy_accepted=output.acceptance["anatomy"], anatomy_reference=truth,
                    anatomy_eligible=eligible, groups=protocol["groups"], groups_by_class=protocol["groups_by_class"],
                    acceptance_policy=member["acceptance_policy"],
                    topk_fraction=member["box_supervision"]["topk_fraction"], patient_uid=record.patient_uid)
                _append_jsonl(rows_path, {"official": official, "diagnostic": diagnostic, "reused": reused,
                                          "identical": identical})
                official_rows.append(official)
                diagnostic_rows.append(diagnostic)
            entry = {"n_scans": len(official_rows),
                     "anatomy_dice": {name: driver._patient_mean(values) for name, values in dice.items()},
                     "findings": summarize_findings_v5(official_rows), "sentences": dict(sorted(sentences.items())),
                     "diagnostics": summarize_diagnostics_v5(diagnostic_rows), "reuse": reuse,
                     "reproduces_original": None}
            original = originals.get(pid)       # the run's own development description is of its final checkpoint
            if role == "development_val" and final and reuse["scans"] and isinstance(original, dict) \
                    and reuse["reused"] in (0, reuse["scans"]):
                entry["reproduces_original"] = {key: entry[key] == original.get(key)
                                                for key in ("findings", "anatomy_dice", "sentences")}
            results[pid][role] = entry
    if state_sha256_v5(model) != before:
        raise V5Error("EVAL_WEIGHTS_CHANGED", "$.checkpoint")
    summary["checkpoint"]["weights_unchanged"] = True
    summary["members"] = results
    summary["status"] = "COMPLETED"
    return finish(None, None)


def _cell(value) -> str:
    if value is None:
        return "NA"
    return str(value) if isinstance(value, int) else f"{value:.3f}"


def _loss(value) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _intro(summary: dict, title: str) -> list:
    run, checkpoint, evaluation = summary["run"], summary["checkpoint"], summary["evaluation"]
    unchanged = {True: "相同，权重未被修改", False: "不同", None: "未完成核对"}[checkpoint["weights_unchanged"]]
    selected = []
    if checkpoint.get("selection") == "explicit":
        last = checkpoint["run_final"]
        selected = [f"- 检查点是按步数与哈希指定的：第 {checkpoint['step']} 步。该运行的最后一个检查点是第 {last['step']} 步"
                    f"（sha256 `{last['sha256']}`），" + ("就是本次读取的这一个。" if checkpoint["is_final"] else
                                                      "本次没有读取它，也没有使用它保存的任何输出；原运行的文件未被改动。")]
    return [f"# {title}：{summary['plan_id']}", "",
            "本报告只含配置、计数、指标与哈希；不含样本或患者 ID、路径或文件名。", "",
            f"- 计划 sha256 `{summary['plan_sha256']}`；被评价的运行 run_sha256 `{run['run_sha256']}`",
            f"- 检查点 sha256 `{checkpoint['sha256']}`（第 {checkpoint['step']} 步）；评价前后的权重哈希"
            f"{unchanged}（state sha256 `{checkpoint['state_sha256']}`）", *selected,
            f"- 训练代码 `{run['code']['git_commit']}`；本次评价代码 `{evaluation['code']['git_commit']}`",
            f"- 结果：{summary['status']}" + (f"（停止：{summary['stop_code']}，阶段 {summary['stop_phase']}）"
                                             if summary["stop_code"] else ""),
            f"- 设备：{summary['device']['name']} × {summary['device']['count']}；用时 "
            f"{summary['elapsed_hours']:.3f} h（本次上限 {summary['budget_hours']:.2f} h；计划整体预算在本次之前剩余 "
            f"{summary['plan_budget_remaining_hours']:.3f} h）",
            f"- scientific_conclusion: {summary['scientific_conclusion']}；只做推理与统计，不更新权重。"
            "train 上的数值只用于诊断，不能代替泛化评价。"]


def render_unified_evaluation_md_v5(summary: dict) -> str:
    """Configuration, counts, metrics and hashes of an evaluate-only run; no identifiers."""
    roles = summary["evaluation"]["roles"]
    lines = _intro(summary, "统一模型只评价运行")
    lines.append(f"- 读取的划分角色：{'、'.join(roles)}；locked_test 未读。")
    evaluation = summary["evaluation"]
    if evaluation.get("reuse_inference") is None:
        lines.append("- 全部预测都是本次用上述检查点重新推理得到的，没有复用任何已保存的输出"
                     + ("（被评价的不是该运行的最后一个检查点，已保存的输出不属于它）。"
                        if evaluation.get("reuse_refused") else "。"))
    for pid, by_role in (summary["members"] or {}).items():
        if "module" in by_role.get("development_val", {}):          # a voxel-supervised member
            entry = by_role["development_val"]
            module = entry["module"]
            lines += ["", f"## {pid}：development_val 的体素级评价（患者等权）", "",
                      f"- 患者 {module['n_patients']}，扫描 {module['n_scans']}；与训练驱动相同的步骤与指标实现"
                      "（整卷推理 → 接受策略 → 模块指标 → 患者等权汇总）；只读 development_val。", "",
                      "| 指标 | 值 |", "| --- | --- |"]
            lines += [f"| {label} | {driver._number(value)} |"
                      for _, label, value, _ in driver._comparison_rows(pid, {"module": module}, None)]
            same = entry["reproduces_original"]
            if same is not None:
                lines += ["", "- 与该运行自己的开发集总结（同一个检查点）比较：模块指标"
                              + ("完全一致" if same["module"] else "不完全一致") + "，句式计数"
                              + ("完全一致" if same["sentences"] else "不完全一致") + "。"]
            lines += ["", f"### {pid}：所见句式计数", "", "| 句式 | 条数 |", "| --- | --- |"]
            lines += [f"| {sentence} | {count} |" for sentence, count in entry["sentences"].items()]
            continue
        present = [role for role in roles if role in by_role]

        def table(header: str, rows: list) -> list:
            out = [f"| {header} | " + " | ".join(present) + " |", "| --- |" + " --- |" * len(present)]
            for label, getter in rows:
                out.append(f"| {label} | " + " | ".join(_cell(getter(by_role[role])) for role in present) + " |")
            return out

        def stats(entry: dict, cls: str | None) -> dict:
            return entry["findings"]["overall"] if cls is None else entry["findings"]["by_class"][cls]

        def funnel(entry: dict, cls: str | None) -> dict:
            block = entry["diagnostics"]["funnel"]
            return block["overall"] if cls is None else block["by_class"][cls]

        classes = list(by_role[present[0]]["findings"]["by_class"])
        lines += ["", f"## {pid}：正式口径的所见评价", "",
                  "各划分以自己的全部主分析框为分母，未命中的框计入每一项的分母。预处理、推理、阈值、体素数下限、"
                  "定位点规则、所在组织规则与统计方法在两个划分上完全相同。", ""]
        for title, cls in [("合计", None), *[(name, name) for name in classes]]:
            rows = [("主分析框", lambda e: stats(e, cls)["n_boxes"]),
                    ("命中的框", lambda e: funnel(e, cls)["hit"]),
                    ("框命中率（合并）", lambda e: stats(e, cls)["box_sensitivity"]["pooled"]),
                    ("框命中率（患者等权）", lambda e: stats(e, cls)["box_sensitivity"]["patient_mean"]),
                    ("命中且所在组织正确的框", lambda e: funnel(e, cls)["host_correct"]),
                    ("归属正确率（合并）", lambda e: stats(e, cls)["host_accuracy"]["pooled"]),
                    ("归属正确率（患者等权）", lambda e: stats(e, cls)["host_accuracy"]["patient_mean"]),
                    *[(f"IoBB ≥ {threshold} 的框比例", lambda e, t=threshold: stats(e, cls)["iobb_at"][t])
                      for threshold in stats(by_role[present[0]], cls)["iobb_at"]],
                    ("组织级阳性召回", lambda e: stats(e, cls)["tissue_positive_recall"])]
            if cls is None:
                rows = [("患者", lambda e: stats(e, None)["n_patients"]), ("扫描", lambda e: stats(e, None)["n_scans"]),
                        ("有主分析框的患者", lambda e: stats(e, None)["n_patients_with_boxes"]), *rows,
                        ("所见总数", lambda e: stats(e, None)["findings"]["total"]),
                        ("定位点在同类主分析框内", lambda e: stats(e, None)["findings"]["in_reference"]),
                        ("定位点在忽略区内", lambda e: stats(e, None)["findings"]["in_ignored"]),
                        ("其余（含定位点落在异类主分析框内的所见）", lambda e: stats(e, None)["findings"]["outside"]),
                        ("其余所见，每次扫描平均", lambda e: stats(e, None)["findings_outside_reference_per_scan"])]
            lines += [f"### {title}", "", *table("项", rows), ""]
        lines += [f"- “其余”是{OUTSIDE_NOTE}：医生没有标注不等于正常。拆分后的四类位置见下面的诊断。", "",
                  f"## {pid}：组织轮廓 Dice（患者等权）", "",
                  *table("组织", [(name, lambda e, name=name: e["anatomy_dice"][name]["mean"])
                                  for name in by_role[present[0]]["anatomy_dice"]]), "",
                  f"## {pid}：已有输出的复用与核对", ""]
        for role in present:
            reuse, same = by_role[role]["reuse"], by_role[role]["reproduces_original"]
            line = f"- {role}：扫描 {reuse['scans']}，复用原运行已保存的输出 {reuse['reused']}"
            if reuse["reused"]:
                line += f"，其中本次重新推理与已保存输出逐数组相同的 {reuse['identical']}"
            if reuse.get("other_checkpoint"):
                line += f"；已保存但记录着另一个检查点哈希、因此没有复用的输出 {reuse['other_checkpoint']}"
            if same is not None:
                line += ("；由" + ("复用输出" if reuse["reused"] else "本次重新推理") + "算出的所见评价、组织 Dice 与"
                         "句式计数与该运行自己的总结"
                         + ("完全一致" if all(same.values()) else "不完全一致：" + "、".join(
                             key for key, value in same.items() if not value)))
            lines.append(line)
        lines += ["- 只有记录着同一个检查点哈希的已保存输出才会被复用。复用时，正式口径的统计与诊断里的二值图、连通块、"
                  "所见都取自已保存的输出；本次推理只提供已保存输出里没有的概率。", ""]
        diagnostics = render_diagnostics_md_v5({role: by_role[role]["diagnostics"] for role in present})
        body = diagnostics.split("\n", 1)[1].lstrip("\n").replace("\n## ", "\n### ")
        lines += [f"## {pid}：所见链路诊断（对照医生的主分析框）", "", *body.rstrip("\n").splitlines()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Sampling replay


def _step_log(path: Path) -> dict:
    """{step: the members entry of that step} of a run's metrics.jsonl; other events are skipped."""
    log = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        raise V5Error("REPLAY_LOG_MISSING", "$.metrics") from None
    for line in lines:
        if line:
            entry = json.loads(line)
            if entry.get("event") == "step":
                log[int(entry["step"])] = entry["members"]
    return log


def _checkpoints(directory: Path) -> dict:
    found = {}
    for sidecar in sorted(Path(directory).glob("step_????????.pt.json")):
        body = json.loads(sidecar.read_text())
        found[int(body["step"])] = (sidecar.with_name(sidecar.name[:-5]), body["sha256"])
    return found


def _parts(view, batch, targets: dict, eligibility: dict, box: dict, precision: str, device: torch.device) -> dict:
    batch, targets, eligibility = _to_device(batch, targets, eligibility, device)
    with torch.inference_mode(), autocast_v5(precision, device.type):
        output = view(batch)
    return box_objective_parts_v5(output.abnormality_logits.float(), targets["U"], eligibility["U"]["e_pos"],
                                  eligibility["U"]["e_neg"], box)


def _weighted(parts: dict, box: dict) -> float:
    return float(box["positive_weight"]) * (parts["positive"] or 0.0) \
        + float(box["negative_weight"]) * (parts["negative"] or 0.0)


def _current(items: list) -> dict:
    with_bags = [item for item in items if item["bags"]]
    return {"batches": len(items), "batches_with_a_bag": len(with_bags), "bags": sum(item["bags"] for item in items),
            "positive": _distribution([item["positive"] for item in with_bags]),
            "negative": _distribution([item["negative"] for item in items])}


def run_unified_replay_v5(*, plan: dict, plan_sha256: str, protocols: dict, members: dict, approvals: dict,
                          raw: Path, label_maps: dict, run_root: Path, private_root: Path, repo: Path, device: str,
                          run_id: str, max_wall_hours: float, loss_steps: int, created_at: str, environment: dict,
                          code: dict, fixture_mode: bool, protocol_ids: list | None = None) -> dict:
    """Redraw every training patch of a box-supervised member of the finished run `run_id`, check the replay
    against the run's log, and recompute the parts of the box objective where the weights are known."""
    started = time.monotonic()
    if not (isinstance(loss_steps, int) and not isinstance(loss_steps, bool) and loss_steps >= 0):
        raise V5Error("INVALID_TYPE", "$.loss_steps")
    opened = _open_finished_run(plan=plan, plan_sha256=plan_sha256, protocols=protocols, members=members,
                                approvals=approvals, run_root=run_root, run_id=run_id, private_root=private_root,
                                max_wall_hours=max_wall_hours, code=code, fixture_mode=fixture_mode)
    private_root, device_t, pids = Path(private_root), torch.device(device), opened["pids"]
    scopes = {pid: protocols[pid]["scope"] for pid in pids}
    wanted = _box_members(plan, protocols, pids, protocol_ids)
    _, digest, final_step = opened["checkpoint"]
    final_state = _load_selected(opened, device_t.type)
    model = _frozen_model(plan, scopes, final_state, device_t)
    before = state_sha256_v5(model)
    run_dir = opened["run_root"] / "runs" / run_id
    log, checkpoints = _step_log(run_dir / "metrics.jsonl"), _checkpoints(run_dir / "checkpoints")
    summary = _header(plan, plan_sha256, opened, scopes, "unified_replay", device_t, before, max_wall_hours,
                      {"roles": ["train", "development_val"], "code": dict(code), "environment": dict(environment),
                       "created_at": created_at, "loss_steps": loss_steps})

    def finish(stop_code: str | None, phase: str | None) -> dict:
        summary["stop_code"], summary["stop_phase"] = stop_code, phase
        summary["elapsed_hours"] = (time.monotonic() - started) / 3600.0
        write_private_json(resolve_private_output(private_root / "summary.json", private_root, repo), summary)
        return summary

    def load(pid: str, role: str) -> dict:
        return driver.load_member_samples_v5(
            plan=plan, protocol_id=pid, protocol=protocols[pid], records=members[pid]["records"],
            registry=members[pid]["registry"], raw=raw, label_maps=label_maps, roles=[role],
            box_references=members[pid].get("box_references"))[0]

    def weights_after(step: int) -> torch.nn.Module:
        """The model as the run had it after `step` steps: the seeded initialisation, a checkpoint, or the end."""
        if step == final_step:
            return model
        if step == 0:
            return build_unified_model_v5(plan, scopes).to(device_t).eval()
        earlier, earlier_sha256 = checkpoints[step]
        return _frozen_model(plan, scopes, load_checkpoint_v5(earlier, opened["unified"], earlier_sha256,
                                                              device_t.type), device_t)

    results = {}
    for pid in wanted:
        member, scope = plan["members"][pid], scopes[pid]
        box, precision = member["box_supervision"], plan["shared"]["precision"]
        recipe = member_recipe_v5(plan, pid, "baseline")
        seed = plan["shared"]["seeds"]["sampler"] + sorted(plan["members"]).index(pid)
        max_steps, classes = recipe["budget"]["max_steps"], list(scope.abnormality_classes)
        samples = load(pid, "train")
        sampler = member_sampler_v5(samples, recipe, seed, box)
        full = full_bag_voxels_v5(samples)
        weights_steps = {0, *[step for step in checkpoints if step < max_steps]}
        shown, share, with_bag = Counter(), {c: [] for c in range(len(classes))}, Counter()
        compared = mismatches = steps_with_a_bag = 0
        history, trajectory = [], {}

        def probe(held: dict, weights: torch.nn.Module) -> dict:
            """The two parts under `weights` on the first loss_steps batches a fresh sampler draws from `held`."""
            fresh = member_sampler_v5(held, recipe, seed, box)
            return _current([_parts(weights.view(pid), *make_batch_v5(held, fresh, recipe, scope), box, precision,
                                    device_t) for _ in range(loss_steps)])

        for step in range(1, max_steps + 1):
            if step % _BUDGET_CHECK_EVERY == 0 and time.monotonic() - started > max_wall_hours * 3600.0:
                return finish("EVAL_BUDGET_EXCEEDED", "replay")
            drawn = len(sampler.drawn)
            batch, targets, eligibility = make_batch_v5(samples, sampler, recipe, scope)
            exposure = batch_exposure_v5(sampler.drawn[drawn:], targets, eligibility)
            logged = (log.get(step) or {}).get(pid)
            if logged is None:
                raise V5Error("REPLAY_LOG_MISSING", "$.metrics")
            for task, value in exposure["support"].items():
                compared += 1
                mismatches += int(value != logged["tasks"][task]["support"])
            steps_with_a_bag += int(bool(exposure["bags"]))
            for c in {c for _, c, _, _ in exposure["bags"]}:
                with_bag[c] += 1
            for uid, c, bag_id, voxels in exposure["bags"]:
                shown[(uid, c, bag_id)] += 1
                share[c].append(voxels / full[(uid, c, bag_id)])
            if step - 1 in weights_steps:               # the weights before this step are known exactly
                weights = weights_after(step - 1)
                parts = _parts(weights.view(pid), batch, targets, eligibility, box, precision, device_t)
                recomputed, logged_loss = _weighted(parts, box), logged["tasks"]["U"]["loss"]
                history.append({"step": step, "weights_step": step - 1, **parts, "recomputed_loss": recomputed,
                                "logged_loss": logged_loss, "abs_diff": abs(recomputed - logged_loss)})
                trajectory[step - 1] = {"train": probe(samples, weights)}
                del weights
        trajectory[final_step] = {"train": probe(samples, model)}
        sampler_equal = final_step == max_steps and sampler.state_dict() == final_state["sampler"].get(pid)
        by_class = {}
        for c, name in enumerate(classes):
            keys = [key for key in full if key[1] == c]
            by_class[name] = {"bags_in_split": len(keys), "steps_with_a_bag": with_bag[c],
                              "bag_occurrences": sum(shown[key] for key in keys),
                              "bags_never_shown": sum(1 for key in keys if shown[key] == 0),
                              "shown_per_bag": _distribution([shown[key] for key in keys]),
                              "share_of_bag_in_patch": _distribution(share[c])}
        tail = [log[step][pid]["tasks"] for step in range(max(1, max_steps - 999), max_steps + 1)]
        head = [log[step][pid]["tasks"] for step in range(1, min(100, max_steps) + 1)]
        logged_loss = {task: {"mean_first_100_steps": float(np.mean([item[task]["loss"] for item in head])),
                              "mean_last_1000_steps": float(np.mean([item[task]["loss"] for item in tail]))}
                       for task in sorted(member["tasks"])}
        del samples
        development = load(pid, "development_val")
        for step in sorted(trajectory):
            if time.monotonic() - started > max_wall_hours * 3600.0:
                return finish("EVAL_BUDGET_EXCEEDED", "trajectory")
            trajectory[step]["development_val"] = probe(development, weights_after(step))
        del development
        results[pid] = {
            "sampler_seed_offset": sorted(plan["members"]).index(pid),
            "verification": {"steps": max_steps, "support_compared": compared, "support_mismatches": mismatches,
                             "sampler_state_equal": bool(sampler_equal)},
            "exposure": {"steps": max_steps, "patches": max_steps * recipe["batch_size"], "bags_in_split": len(full),
                         "steps_with_a_bag": steps_with_a_bag,
                         "bags_never_shown": sum(1 for key in full if shown[key] == 0), "by_class": by_class},
            "history": history, "logged_loss": logged_loss,
            "weights": {"positive_weight": box["positive_weight"], "negative_weight": box["negative_weight"],
                        "topk_fraction": box["topk_fraction"]},
            "trajectory": [{"weights_step": step, **trajectory[step]} for step in sorted(trajectory)],
            "current": dict(trajectory[final_step])}
    if state_sha256_v5(model) != before:
        raise V5Error("EVAL_WEIGHTS_CHANGED", "$.checkpoint")
    summary["checkpoint"]["weights_unchanged"] = True
    summary["members"] = results
    summary["status"] = "COMPLETED"
    return finish(None, None)


def render_unified_replay_md_v5(summary: dict) -> str:
    """Counts, loss parts and hashes of a sampling replay; no identifiers."""
    lines = _intro(summary, "统一模型采样回放")
    lines.append("- 读 train 与 development_val；locked_test 未读。")
    for pid, item in (summary["members"] or {}).items():
        check, exposure, weights = item["verification"], item["exposure"], item["weights"]
        lines += ["", f"## {pid}：回放的核对", "",
                  f"- 采样器只依赖该成员的样本与种子，不依赖权重。按训练时相同的方式重建采样器，逐步重抽 {check['steps']} 步的批次。",
                  f"- 逐步核对：每一步每个任务的监督体素计数与训练日志比较，共 {check['support_compared']} 项，"
                  f"不一致 {check['support_mismatches']} 项。",
                  f"- 回放结束时的采样器状态与最终检查点里保存的状态{'相同' if check['sampler_state_equal'] else '不同'}。",
                  "", f"## {pid}：训练中框监督的实际曝光", "",
                  f"- 步数 {exposure['steps']}，裁块 {exposure['patches']}；train 的主分析框（袋）{exposure['bags_in_split']} 个；"
                  f"批次里至少有一个袋的步数 {exposure['steps_with_a_bag']}"
                  f"（{exposure['steps_with_a_bag'] / exposure['steps']:.3f}）；从未出现的袋 {exposure['bags_never_shown']} 个。",
                  "", "| 类别 | 袋数 | 有该类袋的步数 | 袋出现次数 | 每个袋出现次数：最小 | 中位数 | 最大 | 袋落在裁块内的体素比例：中位数 | 从未出现的袋 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, row in exposure["by_class"].items():
            lines.append(f"| {name} | {row['bags_in_split']} | {row['steps_with_a_bag']} | {row['bag_occurrences']} | "
                         f"{_cell(row['shown_per_bag']['min'])} | {_cell(row['shown_per_bag']['p50'])} | "
                         f"{_cell(row['shown_per_bag']['max'])} | {_cell(row['share_of_bag_in_patch']['p50'])} | "
                         f"{row['bags_never_shown']} |")
        lines += ["", f"## {pid}：框目标的正项与负项", "",
                  f"- 框目标 = {weights['positive_weight']} × 正项 + {weights['negative_weight']} × 负项"
                  f"（正项取每个袋里前 {weights['topk_fraction']} 比例的体素）。",
                  "- 训练日志只记了两项之和，**没有记录**各自的数值：历史上逐步的正项、负项无法恢复，记为缺失。",
                  "- 只有权重已知的时刻可以重算：随机初始化（第 0 步）与每个检查点。下表用该时刻的权重，对回放出的"
                  "下一步批次重算两项，并与日志里那一步的损失比较。这是恢复出的历史值，共 "
                  f"{len(item['history'])} 个时刻。", "",
                  "| 权重所在步 | 批次所在步 | 袋数 | 正项 | 负项 | 重算的损失 | 日志里的损失 | 绝对差 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in item["history"]:
            lines.append(f"| {row['weights_step']} | {row['step']} | {row['bags']} | {_loss(row['positive'])} | "
                         f"{_loss(row['negative'])} | {row['recomputed_loss']:.6f} | {row['logged_loss']:.6f} | "
                         f"{row['abs_diff']:.2e} |")
        lines += ["", "日志里的损失（各任务，分别取开头 100 步与最后 1000 步的平均）：", "",
                  "| 任务 | 开头 100 步 | 最后 1000 步 |", "| --- | --- | --- |"]
        lines += [f"| {task} | {row['mean_first_100_steps']:.4f} | {row['mean_last_1000_steps']:.4f} |"
                  for task, row in item["logged_loss"].items()]
        lines += ["", "各检查点的权重在同一组批次上的两项（新算的诊断值，不是训练时的历史记录）：批次固定为 train "
                      "回放出的前若干步批次，以及用同一采样方式从 development_val 抽出的同样多的批次；第 0 步是随机初始化。"
                      "正项是袋内前 k 比例体素的 −ln(概率) 的平均，0.693 对应概率 0.5，越小表示袋内的响应越高；"
                      "负项是负体素的 −ln(1−概率) 的平均。", "",
                  "| 权重所在步 | train 批次 | train 有袋的批次 | train 正项：平均 | train 正项：中位数 | train 负项：平均 | "
                  "dev 批次 | dev 有袋的批次 | dev 正项：平均 | dev 正项：中位数 | dev 负项：平均 |",
                  "| --- |" + " --- |" * 10]
        for row in item["trajectory"]:
            cells = [str(row["weights_step"])]
            for role in ("train", "development_val"):
                part = row[role]
                cells += [str(part["batches"]), str(part["batches_with_a_bag"]), _loss(part["positive"]["mean"]),
                          _loss(part["positive"]["p50"]), _loss(part["negative"]["mean"])]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "当前检查点的诊断（新算的值，不是历史值）：上表最后一行（最终权重）的分布。", "",
                  "| 划分 | 批次 | 有袋的批次 | 袋数 | 正项：平均 | 正项：中位数 | 正项：第 90 百分位 | 负项：平均 | 负项：中位数 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for role, row in item["current"].items():
            lines.append(f"| {role} | {row['batches']} | {row['batches_with_a_bag']} | {row['bags']} | "
                         f"{_loss(row['positive']['mean'])} | {_loss(row['positive']['p50'])} | "
                         f"{_loss(row['positive']['p90'])} | {_loss(row['negative']['mean'])} | "
                         f"{_loss(row['negative']['p50'])} |")
    return "\n".join(lines) + "\n"
