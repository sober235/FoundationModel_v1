"""Unified driver (Proposal_v5 §0.6; design 2026-11-10 §5.3–§5.6): smoke, G-BASE evidence, training, report.

It orchestrates what already exists, per member through the member's ordinary registered run: registration_v5
(runs, frozen split, exposure), train_unified_v5, infer_and_save_v5 (full volume, original grid), module_eval_v5
for a voxel-supervised member, findings_v5 and evaluation.findings_v5 for a box-supervised member. No training or
scoring code of its own.

Order of a training run: the plan hash, every member's approval naming that hash and every member's smoke evidence
(all before any data is read) → registration and exposure → the train part of each member (a box member's box
references are read for role train only) → training (to the end of the schedule, or to the plan's
shared.stop_at_step) → the final checkpoint reloaded by hash → the development
part (box references read again, for role development_val) with references fixed before any prediction → one
inference and one row per development scan → summary. A failure, timeout or exhausted budget stops at that phase;
it is recorded, never retried, and there is no partial result. locked_test is never read.

Private products stay under the private root; the renderers show configuration, counts, metrics and hashes only.
scientific_conclusion stays NOT_EVALUATED. A finding outside the readers' boxes is a descriptive count, not a
false positive, and an unmarked scan is not a normal scan.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import torch

from .contracts_v5 import ActiveScope, V5Error
from .engine.box_data_v5 import load_box_references_v5, load_box_training_sample_v5
from .engine.data_v5 import load_training_sample_v5, original_geometry_v5
from .engine.module_eval_v5 import freeze_module_reference_v5, module_metrics_v5, summarize_module_metrics_v5
from .engine.registration_v5 import (
    SUBSET_RULE, _non_empty_str, register_run_v5, run_sha256_v5, select_samples_v5, select_subset_v5)
from .engine.smoke_v5 import compare_resume_v5
from .engine.train_unified_v5 import train_unified_v5
from .engine.train_v5 import load_checkpoint_v5
from .engine.unified_registration_v5 import (
    member_recipe_v5, register_unified_run_v5, unified_plan_sha256_v5, validate_unified_plan_v5,
    write_unified_run_v5)
from .evaluation.findings_v5 import (
    OUTSIDE_NOTE, anatomy_dice_v5, render_findings_eval_md_v5, score_findings_v5, summarize_findings_v5)
from .findings_v5 import findings_from_output_v5, host_relation_fn_v5, render_findings_zh_v5
from .gates_v5 import FIXTURE_APPROVERS, gate_task_v5
from .models.interface_v5 import build_model_input_v5
from .models.unified_v5 import build_unified_model_v5
from .paths import resolve_private_output, write_private_json
from .perception import infer_and_save_v5
from .real_baseline_v5 import _reference

CONCLUSION = "NOT_EVALUATED"
_NO_REGIONS = {"review": {}, "visibility": {}}


def check_unified_approval_v5(approval: object, plan_sha256: str, scope: ActiveScope, fixture_mode: bool) -> dict:
    """The researcher's resources approval for this member scope must name this exact unified plan."""
    if not (isinstance(approval, dict) and _non_empty_str(approval.get("approved_by"))
            and _non_empty_str(approval.get("approved_at"))):
        raise V5Error("APPROVAL_MISSING", "$.approvals.resources")
    if approval.get("scope_hash") != scope.scope_hash:
        raise V5Error("GATE_BINDING_MISMATCH", "$.approvals.resources.scope_hash")
    if approval.get("unified_plan_sha256") != plan_sha256:
        raise V5Error("UNIFIED_PLAN_NOT_APPROVED", "$.approvals.resources.unified_plan_sha256")
    if not fixture_mode and approval["approved_by"] in FIXTURE_APPROVERS:
        raise V5Error("GATE_FIXTURE_NOT_ALLOWED", "$.approvals.resources")
    return approval


def _precheck(plan: dict, plan_sha256: str, protocols: dict, members: dict, approvals: dict,
              fixture_mode: bool) -> list:
    """Everything that must hold before any data is read; returns the protocol ids in order."""
    if unified_plan_sha256_v5(plan) != plan_sha256:
        raise V5Error("UNIFIED_PLAN_HASH_MISMATCH", "$.plan_sha256")
    pids = sorted(plan["members"])
    if not all(isinstance(group, dict) and set(group) >= set(pids) for group in (protocols, members, approvals)):
        raise V5Error("UNIFIED_MEMBER_MISMATCH", "$.members")
    validate_unified_plan_v5(plan, {pid: protocols[pid]["scope"] for pid in pids},
                             {pid: members[pid]["source"] for pid in pids})
    for pid in pids:
        check_unified_approval_v5(approvals[pid], plan_sha256, protocols[pid]["scope"], fixture_mode)
    return pids


def load_member_samples_v5(*, plan: dict, protocol_id: str, protocol: dict, records: list, registry: dict,
                           raw: Path, label_maps: dict, roles: list, box_references: Path | None = None,
                           subset: set | None = None) -> tuple:
    """(samples by uid, box references by uid or None) of the member's records in the frozen `roles`.

    A box-supervised member reads its box references through load_box_references_v5 for the same roles, so a
    training load never returns development boxes."""
    member, scope = plan["members"][protocol_id], protocol["scope"]
    recipe = member_recipe_v5(plan, protocol_id, "baseline")
    selected = [record for record in select_samples_v5(records, registry, roles)
                if subset is None or record.sample_uid in subset]
    if member["box_supervision"] is None:
        return {record.sample_uid: load_training_sample_v5(raw, record, scope, member["supervision"], recipe,
                                                           label_maps, _NO_REGIONS) for record in selected}, None
    path = f"$.members.{protocol_id}"
    policy = protocol["box_policy"]
    if policy is None or policy["policy_sha256"] != member["box_supervision"]["box_policy_sha256"]:
        raise V5Error("BOX_POLICY_MISMATCH", f"{path}.box_supervision.box_policy_sha256")
    if box_references is None:
        raise V5Error("BOX_REFERENCE_MISSING", f"{path}.box_references")
    references = load_box_references_v5(Path(box_references), registry, roles)
    samples = {}
    for record in selected:
        if record.sample_uid not in references:
            raise V5Error("BOX_REFERENCE_MISSING", f"{path}.box_references")
        samples[record.sample_uid] = load_box_training_sample_v5(
            raw, record, scope, member["supervision"], recipe, label_maps, _NO_REGIONS,
            references[record.sample_uid], policy)
    return samples, {uid: references[uid] for uid in samples}


def _register(*, plan: dict, plan_sha256: str, protocols: dict, members: dict, approvals: dict, run_id: str,
              run_kind: str, subsets: dict | None, environment: dict, code: dict, created_at: str,
              fixture_mode: bool) -> tuple:
    member_runs = {}
    for pid in sorted(plan["members"]):
        member = members[pid]
        gates = {"G-DATA": member["gdata_decision"]} if run_kind == "smoke" else dict(member["gbase_decisions"])
        member_runs[pid] = register_run_v5(
            run_id=f"{run_id}.{pid}", lineage_id=plan["members"][pid]["lineage_id"], run_kind=run_kind,
            recipe=member_recipe_v5(plan, pid, run_kind), supervision=plan["members"][pid]["supervision"],
            scope=protocols[pid]["scope"], source=member["source"], split_registry=member["registry"],
            records=member["records"], gate_decisions=gates, approvals={"resources": approvals[pid]}, parents=[],
            environment=environment, code=code, created_at=created_at, fixture_mode=fixture_mode,
            sample_subset=None if subsets is None else subsets[pid])
    unified = register_unified_run_v5(run_id=run_id, run_kind=run_kind, plan=plan, plan_sha256=plan_sha256,
                                      member_runs=member_runs, environment=environment, code=code,
                                      created_at=created_at, fixture_mode=fixture_mode)
    return unified, member_runs


def _train_members(plan: dict, member_runs: dict, samples: dict, run_kind: str) -> dict:
    return {pid: {"run": member_runs[pid], "samples": samples[pid], "recipe": member_recipe_v5(plan, pid, run_kind),
                  "supervision": plan["members"][pid]["supervision"],
                  "box_supervision": plan["members"][pid]["box_supervision"],
                  "loss_weight": plan["members"][pid]["loss_weight"]} for pid in sorted(plan["members"])}


def _infer(*, model, pid: str, plan: dict, protocol: dict, observation, record, member_run: dict, registry: dict,
           checkpoint_sha256: str, private_root: Path, repo: Path, folder: str = "inference") -> tuple:
    """One full-volume inference of a member's view through the formal entry, and its findings."""
    member, scope = plan["members"][pid], protocol["scope"]
    relation_fn = None
    if scope.scope_kind == "joint":
        if protocol["groups"] is None or protocol["groups_by_class"] is None:
            raise V5Error("PROTOCOL_GROUPS_MISSING", f"$.protocols.{pid}.groups")
        relation_fn = host_relation_fn_v5(protocol["groups"], protocol["groups_by_class"])
    batch = build_model_input_v5(observation.channels, scope, observation.valid_fov, observation.grid,
                                 list(observation.to_original))
    output, _ = infer_and_save_v5(
        model.view(pid), batch, scope, observation_uid=observation.sample_uid, checkpoint_sha256=checkpoint_sha256,
        original_geometry=original_geometry_v5(observation, record), acceptance_policy=member["acceptance_policy"],
        window=tuple(member["inference"]["window"]), stride=tuple(member["inference"]["stride"]),
        precision=plan["shared"]["precision"], out_dir=Path(private_root) / folder / pid,
        private_root=private_root, repo=repo, run=member_run, registry=registry, relation_fn=relation_fn)
    return output, findings_from_output_v5(output, scope, member["acceptance_policy"], protocol["groups"])


def _device_summary(device: torch.device) -> dict:
    return {"name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "count": 1}


# ---------------------------------------------------------------------------
# Smoke


def run_unified_smoke_v5(*, plan: dict, plan_sha256: str, protocols: dict, members: dict, approvals: dict,
                         raw: Path, label_maps: dict, private_root: Path, repo: Path, device: str, run_id: str,
                         created_at: str, environment: dict, code: dict, fixture_mode: bool) -> dict:
    """Two executions of plan.smoke.steps on each member's registered train subset (continuous, and paused at
    pause_at then resumed), their comparison, and one full-volume inference per member. PASS needs both executions
    COMPLETED, the resume comparison passed and every inference output valid."""
    started = time.monotonic()
    pids = _precheck(plan, plan_sha256, protocols, members, approvals, fixture_mode)
    private_root, device_t, smoke = Path(private_root), torch.device(device), plan["smoke"]
    scopes = {pid: protocols[pid]["scope"] for pid in pids}
    subsets = {}
    for pid in pids:
        subsets[pid] = {"rule": SUBSET_RULE, "salt": smoke["subset_salt"], "strata": None,
                        "count": smoke["subset_count"][pid]}
        subsets[pid]["sample_uids"] = select_subset_v5(members[pid]["records"], members[pid]["registry"],
                                                       count=smoke["subset_count"][pid], salt=smoke["subset_salt"],
                                                       strata=None)
    registered = {}
    for name, rid in (("continuous", run_id), ("resumed", f"{run_id}.r")):
        unified, member_runs = _register(plan=plan, plan_sha256=plan_sha256, protocols=protocols, members=members,
                                         approvals=approvals, run_id=rid, run_kind="smoke", subsets=subsets,
                                         environment=environment, code=code, created_at=created_at,
                                         fixture_mode=fixture_mode)
        root = private_root / name
        run_dir = write_unified_run_v5(unified, member_runs, {pid: members[pid]["registry"] for pid in pids}, root,
                                       repo)
        registered[name] = (unified, member_runs, root, run_dir)
    samples = {pid: load_member_samples_v5(
        plan=plan, protocol_id=pid, protocol=protocols[pid], records=members[pid]["records"],
        registry=members[pid]["registry"], raw=raw, label_maps=label_maps, roles=["train"],
        box_references=members[pid].get("box_references"), subset=set(subsets[pid]["sample_uids"]))[0]
        for pid in pids}
    executions, states, logs, trained, step_seconds = {}, {}, {}, None, None
    if device_t.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_t)
    for name in ("continuous", "resumed"):
        unified, member_runs, root, run_dir = registered[name]
        train_members = _train_members(plan, member_runs, samples, "smoke")
        model = build_unified_model_v5(plan, scopes).to(device_t)
        tick = time.monotonic()
        result = train_unified_v5(unified, model, train_members, plan, run_dir, root, repo, None,
                                  pause_at=smoke["pause_at"] if name == "resumed" else None)
        if name == "resumed" and result["status"] == "PAUSED":
            model = build_unified_model_v5(plan, scopes).to(device_t)
            result = train_unified_v5(unified, model, train_members, plan, run_dir, root, repo,
                                      (result["last_checkpoint"], result["last_checkpoint_sha256"]))
        executions[name] = {key: result[key] for key in ("status", "step", "failure_code")}
        if result["status"] == "COMPLETED":
            states[name] = load_checkpoint_v5(result["last_checkpoint"], unified, result["last_checkpoint_sha256"],
                                              device_t.type)
            logs[name] = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line]
            if name == "continuous":
                trained = (model, result["last_checkpoint_sha256"])
                step_seconds = (time.monotonic() - tick) / max(result["step"], 1)
    resume = None
    if len(states) == 2:
        resume = compare_resume_v5(states["continuous"], states["resumed"], logs["continuous"], logs["resumed"],
                                   pause_at=smoke["pause_at"], tolerances=smoke["tolerances"])
    member_summary = {pid: {"scope_id": scopes[pid].scope_id, "scope_hash": scopes[pid].scope_hash,
                            "samples": len(samples[pid]), "inference": None} for pid in pids}
    if trained is not None:
        _, member_runs, root, _ = registered["continuous"]
        by_uid = {pid: {record.sample_uid: record for record in members[pid]["records"]} for pid in pids}
        for pid in pids:
            uid = sorted(samples[pid])[0]
            output, findings = _infer(model=trained[0], pid=pid, plan=plan, protocol=protocols[pid],
                                      observation=samples[pid][uid].observation, record=by_uid[pid][uid],
                                      member_run=member_runs[pid], registry=members[pid]["registry"],
                                      checkpoint_sha256=trained[1], private_root=root, repo=repo)
            member_summary[pid]["inference"] = {"output_kind": output.output_kind, "findings": len(findings)}
    passed = all(execution["status"] == "COMPLETED" for execution in executions.values()) \
        and resume is not None and resume["passed"] \
        and all(member["inference"] is not None for member in member_summary.values())
    summary = {"kind": "unified_smoke", "status": "PASS" if passed else "FAIL", "plan_id": plan["plan_id"],
               "plan_sha256": plan_sha256, "device": _device_summary(device_t), "executions": executions,
               "resume": resume, "members": member_summary, "step_seconds": step_seconds,
               "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device_t)) if device_t.type == "cuda" else None,
               "elapsed_hours": (time.monotonic() - started) / 3600.0, "budget_hours": smoke["max_wall_hours"],
               "scientific_conclusion": CONCLUSION}
    write_private_json(resolve_private_output(private_root / "summary.json", private_root, repo), summary)
    return summary


def render_unified_smoke_md_v5(summary: dict) -> str:
    """Counts, timings and hashes of a unified smoke; no identifiers or paths."""
    resume = summary["resume"]
    lines = [f"# 统一冒烟：{summary['plan_id']}", "",
             "本报告只含配置、计数、耗时与哈希；不含样本或患者 ID、路径或文件名。", "",
             f"- 计划 sha256 `{summary['plan_sha256']}`",
             f"- 结果：{summary['status']}；scientific_conclusion: {summary['scientific_conclusion']}",
             f"- 设备：{summary['device']['name']} × {summary['device']['count']}；用时 "
             f"{summary['elapsed_hours']:.3f} h（预算 {summary['budget_hours']:.2f} h）"]
    if summary["step_seconds"] is not None:
        lines.append(f"- 连续执行的单步耗时：{summary['step_seconds']:.3f} s（每步各协议一个批次）")
    if summary["peak_memory_bytes"] is not None:
        lines.append(f"- 显存峰值：{summary['peak_memory_bytes'] / 2 ** 30:.2f} GiB")
    lines += ["", "| 执行 | 状态 | 步数 | 失败码 |", "| --- | --- | --- | --- |"]
    lines += [f"| {name} | {item['status']} | {item['step']} | {item['failure_code'] or '无'} |"
              for name, item in summary["executions"].items()]
    if resume is not None:
        lines += ["", f"- 续训比较：{'通过' if resume['passed'] else '未通过'}；暂停于第 {resume['pause_at']} 步；"
                      f"比较步数 {resume['steps_compared']}；参数逐位相同：{'是' if resume['bitwise_parameters'] else '否'}；"
                      f"参数最大相对差 {resume['max_param_rel_diff']:.3g}；损失最大绝对差 {resume['max_loss_abs_diff']:.3g}"]
    lines += ["", "| 协议 | scope | 训练样本 | 整卷推理 | 所见数 |", "| --- | --- | --- | --- | --- |"]
    for pid, member in summary["members"].items():
        inference = member["inference"]
        lines.append(f"| {pid} | {member['scope_id']} | {member['samples']} | "
                     f"{inference['output_kind'] if inference else '未完成'} | "
                     f"{inference['findings'] if inference else '—'} |")
    lines += ["", "- 只读各协议登记的 train 子集；development_val 与 locked_test 未读。"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Training and the development report


def _check_smoke_evidence(evidence: object, scope: ActiveScope, source: dict, pid: str) -> None:
    expected = {"status": "PASS", "scope_hash": scope.scope_hash, "task": gate_task_v5(scope), **source}
    if not isinstance(evidence, dict) or any(evidence.get(key) != value for key, value in expected.items()):
        raise V5Error("SMOKE_EVIDENCE_MISMATCH", f"$.members.{pid}.smoke_evidence")


def _patient_mean(values_by_patient: dict) -> dict:
    """Mean within each patient of the scans that have a value, then patients weighted equally."""
    means = []
    for values in values_by_patient.values():
        present = [value for value in values if value is not None]
        if present:
            means.append(sum(present) / len(present))
    return {"mean": sum(means) / len(means) if means else None, "n_patients": len(means)}


def run_unified_training_v5(*, plan: dict, plan_sha256: str, protocols: dict, members: dict, approvals: dict,
                            raw: Path, label_maps: dict, private_root: Path, repo: Path, device: str, run_id: str,
                            created_at: str, environment: dict, code: dict, fixture_mode: bool,
                            resume: bool = False) -> dict:
    """One unified training run and its development description; returns the summary (also summary.json).

    `resume=True` continues a run whose process died: the run registered under `private_root` must be the one these
    inputs register (same plan, records, splits, gates and approvals; its own created_at, environment and code are
    kept), training goes on from the latest checkpoint, and the development inference starts again in a new
    `inference_r<n>` folder, so nothing of the dead attempt is overwritten or reused. A run that already wrote its
    summary, or that stopped with a failure record, is not resumed."""
    started = time.monotonic()
    pids = _precheck(plan, plan_sha256, protocols, members, approvals, fixture_mode)
    for pid in pids:
        _check_smoke_evidence(members[pid].get("smoke_evidence"), protocols[pid]["scope"], members[pid]["source"], pid)
    private_root, device_t = Path(private_root), torch.device(device)
    scopes = {pid: protocols[pid]["scope"] for pid in pids}
    stored = None
    if resume:
        try:
            stored = json.loads(resolve_private_output(private_root / "runs" / run_id / "run.json", private_root,
                                                       repo).read_text())
        except (OSError, ValueError):
            raise V5Error("RESUME_RUN_MISSING", "$.run") from None
        if not isinstance(stored, dict) or not all(key in stored for key in ("environment", "code", "created_at")):
            raise V5Error("RESUME_RUN_MISSING", "$.run")
        environment, code, created_at = stored["environment"], stored["code"], stored["created_at"]
    unified, member_runs = _register(plan=plan, plan_sha256=plan_sha256, protocols=protocols, members=members,
                                     approvals=approvals, run_id=run_id, run_kind="baseline", subsets=None,
                                     environment=environment, code=code, created_at=created_at,
                                     fixture_mode=fixture_mode)
    carried_seconds, checkpoint, folder = 0.0, None, "inference"
    if resume:
        if unified != stored:
            raise V5Error("RESUME_RUN_MISMATCH", "$.run")
        if (private_root / "summary.json").exists():
            raise V5Error("RUN_ALREADY_FINISHED", "$.run")
        checkpoints = private_root / "runs" / run_id / "checkpoints"
        steps = sorted(int(path.name[5:13]) for path in checkpoints.glob("step_????????.pt")) \
            if checkpoints.is_dir() else []
        if not steps:
            raise V5Error("RESUME_NO_CHECKPOINT", "$.run")
        latest = checkpoints / f"step_{steps[-1]:08d}.pt"
        try:
            digest = json.loads(latest.with_name(f"{latest.name}.json").read_text())["sha256"]
        except (OSError, ValueError, KeyError, TypeError):
            raise V5Error("RESUME_NO_CHECKPOINT", "$.run.checkpoint") from None
        checkpoint = (latest, digest)
        carried_seconds = float(load_checkpoint_v5(latest, unified, digest, "cpu")["wall_seconds"])
        attempt = 1
        while (private_root / f"inference_r{attempt}").exists():
            attempt += 1
        folder = f"inference_r{attempt}" if (private_root / "inference").exists() else "inference"
    summary = {"kind": "unified_training", "plan_id": plan["plan_id"], "plan_sha256": plan_sha256,
               "device": _device_summary(device_t),
               "run": {"run_id": run_id, "run_sha256": run_sha256_v5(unified),
                       "members": {pid: {"scope_id": scopes[pid].scope_id, "scope_hash": scopes[pid].scope_hash,
                                         "run_sha256": unified["members"][pid]["run_sha256"],
                                         "split_sha256": unified["members"][pid]["split_sha256"]} for pid in pids}},
               "status": "FAILED", "stop_code": None, "stop_phase": None, "training": None, "members": None,
               "elapsed_hours": None, "budget_hours": plan["budget"]["max_wall_hours"],
               "scientific_conclusion": CONCLUSION}

    def finish(stop_code: str | None, phase: str | None) -> dict:
        summary["stop_code"], summary["stop_phase"] = stop_code, phase
        summary["elapsed_hours"] = (carried_seconds + time.monotonic() - started) / 3600.0
        write_private_json(resolve_private_output(private_root / "summary.json", private_root, repo), summary)
        return summary

    def over_budget() -> bool:          # a resumed run carries the training time its checkpoint recorded
        return carried_seconds + time.monotonic() - started > plan["budget"]["max_wall_hours"] * 3600.0

    def load(pid: str, roles: list) -> tuple:
        return load_member_samples_v5(
            plan=plan, protocol_id=pid, protocol=protocols[pid], records=members[pid]["records"],
            registry=members[pid]["registry"], raw=raw, label_maps=label_maps, roles=roles,
            box_references=members[pid].get("box_references"))

    if resume:
        run_dir = private_root / "runs" / run_id
    else:
        run_dir = write_unified_run_v5(unified, member_runs, {pid: members[pid]["registry"] for pid in pids},
                                       private_root, repo)
    # 1. the train part of every member, seeded random initialisation, selection rule `last`
    train = {pid: load(pid, ["train"])[0] for pid in pids}
    model = build_unified_model_v5(plan, scopes).to(device_t)
    result = train_unified_v5(unified, model, _train_members(plan, member_runs, train, "baseline"), plan, run_dir,
                              private_root, repo, checkpoint)
    summary["training"] = {"status": result["status"], "step": result["step"],
                           "schedule_steps": result["schedule_steps"], "controlled_stop": result["controlled_stop"],
                           "samples": {pid: len(train[pid]) for pid in pids},
                           "checkpoint_sha256": result["last_checkpoint_sha256"],
                           "failure_code": result["failure_code"]}
    del train
    if result["status"] != "COMPLETED":
        return finish(result["failure_code"], "training")
    if over_budget():
        return finish("UNIFIED_BUDGET_EXCEEDED", "training")
    state = load_checkpoint_v5(result["last_checkpoint"], unified, result["last_checkpoint_sha256"], device_t.type)
    if state["step"] != result["step"]:
        raise V5Error("CHECKPOINT_NOT_FINAL", "$.training.checkpoint")
    model.load_state_dict(state["model"])
    # 2. the development part: every member's references are fixed before any prediction
    development = {}
    for pid in pids:
        samples, references = load(pid, ["development_val"])
        by_uid = {record.sample_uid: record for record in members[pid]["records"]}
        scope, entry = scopes[pid], {"observations": {uid: s.observation for uid, s in samples.items()},
                                     "records": {uid: by_uid[uid] for uid in samples}}
        if references is None:
            entry["frozen"] = freeze_module_reference_v5(
                {uid: _reference(sample, by_uid[uid], scope) for uid, sample in samples.items()}, scope,
                members[pid]["registry"])
        else:
            entry["boxes"] = references
            entry["anatomy"] = {uid: (sample.targets["A"], (sample.e_pos["A"] & sample.e_neg["A"]).all(axis=0))
                                for uid, sample in samples.items()}
        development[pid] = entry
    # 3. one full-volume inference and one row per development scan
    member_summary = {}
    for pid in pids:
        entry, scope, member = development[pid], scopes[pid], plan["members"][pid]
        rows, sentences, dice = [], Counter(), {name: {} for name in scope.anatomy_classes}
        for uid in sorted(entry["observations"]):
            if over_budget():                   # no partial result: every development scan of every member or none
                return finish("UNIFIED_BUDGET_EXCEEDED", "evaluation")
            record = entry["records"][uid]
            output, findings = _infer(model=model, pid=pid, plan=plan, protocol=protocols[pid],
                                      observation=entry["observations"].pop(uid), record=record,
                                      member_run=member_runs[pid], registry=members[pid]["registry"],
                                      checkpoint_sha256=result["last_checkpoint_sha256"], private_root=private_root,
                                      repo=repo, folder=folder)
            sentences.update(render_findings_zh_v5(findings, scope, protocols[pid]["display"]))
            if "frozen" in entry:
                rows.append(module_metrics_v5(output, entry["frozen"], member["metric_policy"],
                                              member["acceptance_policy"], member_runs[pid]))
                continue
            truth, eligible = entry["anatomy"][uid]
            for name, value in anatomy_dice_v5(output.anatomy_map, truth, eligible, list(scope.anatomy_classes)).items():
                dice[name].setdefault(record.patient_uid, []).append(value)
            rows.append(score_findings_v5(findings, entry["boxes"][uid], record.patient_uid,
                                          list(scope.abnormality_classes), member["finding_policy"]["iobb_thresholds"]))
        counted = dict(sorted(sentences.items()))
        if "frozen" in entry:
            member_summary[pid] = {"module": summarize_module_metrics_v5(rows), "sentences": counted}
        else:
            member_summary[pid] = {"anatomy_dice": {name: _patient_mean(values) for name, values in dice.items()},
                                   "findings": summarize_findings_v5(rows), "sentences": counted}
    summary["members"] = member_summary
    summary["status"] = "COMPLETED"
    return finish(None, None)


def _number(value) -> str:
    return "NA" if value is None else f"{value:.3f}"


def _comparison_rows(pid: str, member: dict, baseline: dict | None) -> list:
    """(metric label, unified value, baseline value or None) of one member; the baseline is a module summary."""
    rows = []
    base = (baseline or {}).get("module") or {}
    if "module" in member:
        module = member["module"]
        for task, classes in module["segmentation"].items():
            for name, fields in classes.items():
                for field, stat in fields.items():
                    other = base.get("segmentation", {}).get(task, {}).get(name, {}).get(field, {}).get("mean")
                    rows.append((f"{task} {name} {field}", stat["mean"], other))
        for field, stat in (module["detection"] or {}).items():
            other = ((base.get("detection") or {}).get(field) or {}).get("mean")
            rows.append((f"病灶 {field}", stat["mean"], other))
    else:
        for name, stat in member["anatomy_dice"].items():
            other = base.get("segmentation", {}).get("A", {}).get(name, {}).get("dice", {}).get("mean")
            rows.append((f"A {name} dice", stat["mean"], other))
    return [(pid, *row) for row in rows]


def render_unified_md_v5(summary: dict, baseline_summaries: dict | None = None) -> str:
    """Configuration, counts, metrics and hashes of a unified run next to the module baselines; no identifiers."""
    baselines = baseline_summaries or {}
    run, training = summary["run"], summary["training"]
    lines = [f"# 统一模型运行：{summary['plan_id']}", "",
             "本报告只含配置、计数、指标与哈希；不含样本或患者 ID、路径或文件名。", "",
             f"- 计划 sha256 `{summary['plan_sha256']}`；run_sha256 `{run['run_sha256']}`",
             f"- 结果：{summary['status']}" + (f"（停止：{summary['stop_code']}，阶段 {summary['stop_phase']}）"
                                              if summary["stop_code"] else ""),
             f"- 设备：{summary['device']['name']} × {summary['device']['count']}；用时 "
             f"{summary['elapsed_hours']:.3f} h（预算 {summary['budget_hours']:.2f} h）",
             f"- scientific_conclusion: {summary['scientific_conclusion']}；只是开发集描述，不计算区间，不形成阈值或临床判定。",
             "", "| 协议 | scope | 成员 run_sha256 | split_sha256 |", "| --- | --- | --- | --- |"]
    lines += [f"| {pid} | {item['scope_id']} | `{item['run_sha256']}` | `{item['split_sha256']}` |"
              for pid, item in run["members"].items()]
    if training is not None:
        counts = "，".join(f"{pid} {n}" for pid, n in training["samples"].items())
        lines += ["", "## 训练", "",
                  f"- 同一个 checkpoint；随机初始化；每步各协议一个批次；完成步数 {training['step']}；状态 "
                  f"{training['status']}；选择规则 last",
                  f"- train 样本：{counts}", f"- 最终检查点 sha256 `{training['checkpoint_sha256']}`"]
        if training.get("controlled_stop"):
            lines.append(f"- 受控停止：按计划在第 {training['step']} 步结束；学习率仍按 {training['schedule_steps']} 步"
                         "的衰减期限计算，所以已训练的各步与完整训练的前同样多步学习率相同。这个检查点是本次运行"
                         "的最后一个检查点，不是完整期限训练的终点。")
    if summary["members"] is not None:
        lines += ["", "## 与模块基线的对照（development_val，患者等权）", "",
                  "| 协议 | 指标 | 统一模型 | 基线 | 差值 |", "| --- | --- | --- | --- | --- |"]
        for pid, member in summary["members"].items():
            for _, label, value, other in _comparison_rows(pid, member, baselines.get(pid)):
                difference = "未提供" if other is None or value is None else f"{value - other:+.3f}"
                lines.append(f"| {pid} | {label} | {_number(value)} | "
                             f"{'未提供' if other is None else _number(other)} | {difference} |")
        lines += ["", "- 基线一列来自各协议单独训练的模块基线汇总；没有给出时写“未提供”。差值只作并列对照，不设通过界限。"]
        for pid, member in summary["members"].items():
            if "findings" in member:
                body = render_findings_eval_md_v5(member["findings"]).split("\n", 1)[1].lstrip("\n")
                lines += ["", f"## {pid}：所见评价（对照医生的主分析框）", "", *body.splitlines()]
            lines += ["", f"## {pid}：所见句式计数", "", "| 句式 | 条数 |", "| --- | --- |"]
            lines += [f"| {sentence} | {count} |" for sentence, count in member["sentences"].items()]
    lines += ["", f"- 读 train 与 development_val；locked_test 未读。医生框外的所见是{OUTSIDE_NOTE}；"
                  "医生没有标注不等于正常。"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command line


def _member_args(values: list) -> dict:
    """--member <protocol>:key=path[,key=path...] -> {protocol: {key: path}}."""
    parsed = {}
    for value in values or []:
        pid, _, rest = value.partition(":")
        if not pid or not rest or pid in parsed:
            raise V5Error("INVALID_MEMBER_ARGUMENT", "$.member")
        parsed[pid] = dict(item.split("=", 1) for item in rest.split(","))
    return parsed


def main(argv: list | None = None) -> int:
    """plan-hash: the unified plan's sha256 for the approvals. smoke / train: one run into a new --out.
    gbase-gate: one member's five G-BASE evidence items and its G-BASE-A / G-BASE-U decisions.
    evaluate / replay: read the finished run --run-id under --run-root with its final checkpoint, never train,
    and write into a new --out (real_unified_eval_v5); the members are given exactly as they were for train.
    evaluate takes another checkpoint of the run with --checkpoint-step and --checkpoint-sha256 (both).
    probe: the box objective's terms and the reach of the hard-negative selection under the given checkpoints
    (--checkpoint STEP:SHA256, repeatable) of the registered run, on its first --loss-steps training batches; the
    run need not have finished. compare: whether the run of --plan and the run of --control-plan were run alike
    over their first --steps steps (plans, seeded initialisation, logs, checkpoint states); reads no sample."""
    import argparse
    import os

    import yaml

    from .contracts_v5 import validate_sample_v5
    from .engine.data_v5 import load_label_maps_v5
    from .engine.registration_v5 import capture_code_v5, capture_environment_v5
    from .findings_v5 import load_protocols_v5
    from .gates_v5 import evaluate_gate_v5
    from .real_t10_v5 import gbase_evidence_v5, training_smoke_evidence_v5
    from .source_policy_v5 import load_source_policies_v5, source_binding_v5
    from .splits_v5 import verify_split_registry_v5

    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(prog="anatobind.v41.real_unified_v5")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan-hash", "smoke", "gbase-gate", "train", "evaluate", "replay", "probe", "compare"):
        command = sub.add_parser(name)
        for key in ("--plan", "--protocols", "--policies"):
            command.add_argument(key, required=True)
        if name in ("smoke", "train", "evaluate", "replay", "probe"):
            for key in ("--raw", "--approvals", "--run-id", "--device", "--out"):
                command.add_argument(key, required=True)
            command.add_argument("--member", action="append", required=True)
        if name in ("evaluate", "replay", "probe"):
            command.add_argument("--run-root", required=True, help="the private root of the training run")
            command.add_argument("--max-wall-hours", required=True, type=float,
                                 help="wall budget of this command; must fit in what the plan's budget has left")
        if name == "probe":
            command.add_argument("--checkpoint", action="append", required=True, metavar="STEP:SHA256",
                                 help="a checkpoint of the run, by step and hash; repeat for several")
            command.add_argument("--loss-steps", required=True, type=int,
                                 help="the run's first training batches on which every checkpoint is read")
            command.add_argument("--negative-topk-fraction", type=float,
                                 help="try this hard-negative fraction on a run whose plan sets none")
        if name == "compare":
            for key in ("--control-plan", "--run-root", "--run-id", "--control-run-root", "--control-run-id", "--out"):
                command.add_argument(key, required=True)
            command.add_argument("--steps", required=True, type=int, help="the first steps of both runs to compare")
            command.add_argument("--checkpoint-step", action="append", type=int, default=[],
                                 help="a step at which both runs kept a checkpoint; repeat for several")
        if name == "evaluate":
            command.add_argument("--role", action="append", required=True, choices=["train", "development_val"])
            command.add_argument("--checkpoint-step", type=int,
                                 help="evaluate this checkpoint of the run instead of its final one (with its hash)")
            command.add_argument("--checkpoint-sha256", help="the sha256 of the checkpoint of --checkpoint-step")
            command.add_argument("--fresh-predictions", action="store_true",
                                 help="reuse no saved output of the run: every prediction is computed anew")
            command.add_argument("--module-development", action="store_true",
                                 help="also score every voxel-supervised member on development_val")
        if name == "replay":
            command.add_argument("--loss-steps", required=True, type=int,
                                 help="patches per split on which the final weights' loss parts are recomputed")
        if name == "train":
            command.add_argument("--baseline-summary", action="append", default=[])
            command.add_argument("--resume", action="store_true",
                                 help="continue the run already registered under --out from its latest checkpoint")
        if name == "gbase-gate":
            for key in ("--protocol", "--gdata-decision", "--split-summary", "--smoke-evidence", "--approvals", "--out"):
                command.add_argument(key, required=True)
    args = parser.parse_args(argv)

    def load_yaml(path: str):
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    def load_json(path: str):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    plan = load_yaml(args.plan)
    table = load_protocols_v5(Path(args.protocols), Path(args.protocols).parent)
    if not isinstance(plan, dict) or not isinstance(plan.get("members"), dict) or not set(plan["members"]) <= set(table):
        raise V5Error("PROTOCOL_UNKNOWN", "$.members")
    protocols = {pid: table[pid] for pid in plan["members"]}
    policies = load_source_policies_v5(Path(args.policies))
    sources = {pid: source_binding_v5(policies[protocol["source_id"]]) for pid, protocol in protocols.items()}
    validate_unified_plan_v5(plan, {pid: protocol["scope"] for pid, protocol in protocols.items()}, sources)
    digest = unified_plan_sha256_v5(plan)
    if args.command == "plan-hash":
        print(digest)
        return 0
    if args.command == "compare":
        from . import real_unified_eval_v5 as finished

        control = load_yaml(args.control_plan)
        if not isinstance(control, dict) or not isinstance(control.get("members"), dict) \
                or set(control["members"]) != set(protocols):
            raise V5Error("UNIFIED_MEMBER_MISMATCH", "$.control_plan.members")
        validate_unified_plan_v5(control, {pid: protocol["scope"] for pid, protocol in protocols.items()}, sources)
        out = Path(args.out)
        summary = finished.compare_unified_runs_v5(
            plan=plan, control_plan=control, protocols=protocols, run_root=Path(args.run_root), run_id=args.run_id,
            control_run_root=Path(args.control_run_root), control_run_id=args.control_run_id, steps=args.steps,
            checkpoint_steps=list(args.checkpoint_step), private_root=out, code=capture_code_v5(repo),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        markdown = finished.render_unified_comparison_md_v5(summary)
        fd = os.open(out / "unified_comparison.md", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(markdown)
        print(markdown, end="")
        return 0 if summary["status"] == "COMPLETED" else 1
    approvals_file = load_yaml(args.approvals) or {}

    if args.command == "gbase-gate":
        from .real_t08_v5 import render_split_md_v5

        pid = args.protocol
        if pid not in protocols:
            raise V5Error("PROTOCOL_UNKNOWN", "$.protocol")
        scope, source, member = protocols[pid]["scope"], sources[pid], plan["members"][pid]
        split = load_json(args.split_summary)
        if split.get("lineage_id") != member["lineage_id"]:
            raise V5Error("SPLIT_LINEAGE_MISMATCH", "$.split_summary.lineage_id")
        evidence = gbase_evidence_v5(
            scope=scope, source=source,
            plan={"scope_hash": scope.scope_hash, "recipe": member_recipe_v5(plan, pid, "baseline"),
                  "supervision": member["supervision"]},
            gdata_decision=load_json(args.gdata_decision), split_markdown=render_split_md_v5(split),
            development_groups=split["counts"]["development_val"], smoke_evidence=load_json(args.smoke_evidence))
        out = Path(args.out)
        out.mkdir(mode=0o700)                                           # FileExistsError: never overwritten
        write_private_json(out / "gbase_evidence.json", evidence)
        for task in sorted(member["tasks"]):
            gate_id = f"G-BASE-{task}"
            decision = evaluate_gate_v5(gate_id, scope, evidence,
                                        approvals_file.get(scope.scope_id, {}).get(gate_id, {}), source)
            write_private_json(out / f"gbase_decision.{task}.json", decision)
            print(json.dumps({k: decision[k] for k in ("gate_id", "scope_id", "engineering_status", "blockers")}))
        return 0

    given = _member_args(args.member)
    if set(given) != set(protocols):
        raise V5Error("UNIFIED_MEMBER_MISMATCH", "$.member")
    members, approvals = {}, {}
    for pid, paths in given.items():
        scope, tasks = protocols[pid]["scope"], sorted(plan["members"][pid]["tasks"])
        scoped = approvals_file.get(scope.scope_id, {})
        if args.command == "smoke":
            approvals[pid] = scoped.get("unified_smoke", {}).get("resources")
        else:                       # every task's G-BASE resources approval must name this plan
            granted = [scoped.get(f"G-BASE-{task}", {}).get("resources") for task in tasks]
            for approval in granted:
                check_unified_approval_v5(approval, digest, scope, fixture_mode=False)
            approvals[pid] = granted[0]
        check_unified_approval_v5(approvals[pid], digest, scope, fixture_mode=False)
        members[pid] = {
            "records": [validate_sample_v5(json.loads(line))
                        for line in Path(paths["records"]).read_text().splitlines() if line],
            "registry": verify_split_registry_v5(load_json(paths["split-registry"])), "source": sources[pid],
            "gdata_decision": load_json(paths["gdata-decision"]) if "gdata-decision" in paths else None,
            "gbase_decisions": {f"G-BASE-{task}": load_json(paths[f"gbase-{task.lower()}"]) for task in tasks}
            if args.command != "smoke" else None,
            "smoke_evidence": load_json(paths["smoke-evidence"]) if "smoke-evidence" in paths else None,
            "box_references": Path(paths["box-references"]) if "box-references" in paths else None}
    out = Path(args.out)
    reading = args.command in ("evaluate", "replay", "probe")           # their driver refuses an existing --out
    resuming = args.command == "train" and args.resume
    if not resuming and not reading:
        out.mkdir(mode=0o700)                                           # never overwritten
        os.chmod(out, 0o700)
    common = {"plan": plan, "plan_sha256": digest, "protocols": protocols, "members": members,
              "approvals": approvals, "raw": Path(args.raw),
              "label_maps": load_label_maps_v5(repo / "configs" / "v5" / "label_maps_v5.yaml"), "private_root": out,
              "repo": repo, "device": args.device, "run_id": args.run_id,
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "environment": capture_environment_v5(), "code": capture_code_v5(repo), "fixture_mode": False}
    if args.command == "smoke":
        summary = run_unified_smoke_v5(**common)
        markdown, name = render_unified_smoke_md_v5(summary), "unified_smoke.md"
        for pid in sorted(protocols):
            write_private_json(out / f"smoke_evidence.{pid}.json",
                               training_smoke_evidence_v5(summary, markdown, protocols[pid]["scope"], sources[pid]))
        ok = summary["status"] == "PASS"
    elif reading:
        from . import real_unified_eval_v5 as finished

        common.update(run_root=Path(args.run_root), max_wall_hours=args.max_wall_hours)
        if args.command == "evaluate":
            selected = None
            if args.checkpoint_step is not None or args.checkpoint_sha256 is not None:
                if args.checkpoint_step is None or args.checkpoint_sha256 is None:      # a step needs its hash
                    raise V5Error("EVAL_CHECKPOINT_INVALID", "$.checkpoint")
                selected = (args.checkpoint_step, args.checkpoint_sha256)
            summary = finished.run_unified_evaluation_v5(
                **common, roles=list(args.role), checkpoint=selected,
                reuse_inference=None if args.fresh_predictions else "inference",
                module_development=args.module_development)
            markdown, name = finished.render_unified_evaluation_md_v5(summary), "unified_evaluation.md"
        elif args.command == "probe":
            selected = []
            for value in args.checkpoint:
                step, _, sha = value.partition(":")
                if not step.isdigit() or not sha:
                    raise V5Error("EVAL_CHECKPOINT_INVALID", "$.checkpoints")
                selected.append((int(step), sha))
            summary = finished.run_unified_probe_v5(
                **common, checkpoints=selected, loss_steps=args.loss_steps,
                negative_topk_fraction=args.negative_topk_fraction)
            markdown, name = finished.render_unified_probe_md_v5(summary), "unified_probe.md"
        else:
            summary = finished.run_unified_replay_v5(**common, loss_steps=args.loss_steps)
            markdown, name = finished.render_unified_replay_md_v5(summary), "unified_replay.md"
        ok = summary["status"] == "COMPLETED"
    else:
        summary = run_unified_training_v5(**common, resume=resuming)
        baselines = {}
        for value in args.baseline_summary:
            pid, _, path = value.partition("=")
            baselines[pid] = load_json(path)
        markdown, name = render_unified_md_v5(summary, baselines), "unified_report.md"
        ok = summary["status"] == "COMPLETED"
    fd = os.open(out / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(markdown)
    print(markdown, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
