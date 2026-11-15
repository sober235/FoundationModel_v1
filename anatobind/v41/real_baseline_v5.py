"""Real T10 module baseline driver for v5.1 (plan_v5 §2C T10, real-run step 4; Proposal_v5 §6.1, §8.3, §9).

It orchestrates what B2 built, like the smoke driver: registration_v5 (a `baseline` run that reads train and
development_val), train_v5 (random initialisation, every train sample, selection rule `last`), infer_and_save_v5
(full volume, original grid) and module_eval_v5 (references frozen before any prediction, patient-equal summary
over every development_val scan). No training or scoring code of its own.

A baseline runs from one baseline plan (configs/v5/proposals/baseline_<scope>.yaml) that follows the scope's passed
smoke: the model, seeds, optimizer, scheduler, input policy, sampling, inference window, supervision and acceptance
policy are the smoke plan's, and patch, batch and precision are the candidate the smoke's rule chose. Only the
recipe id, the readable roles (train + development_val), the step and wall budget and the checkpoint interval may
differ; the plan adds the development metric policy. The researcher's G-BASE `resources` approval (approval C)
names the plan's sha256, and the run registers against the PASS G-BASE gate decision.

Order: approval and smoke binding (before any data is read) → registration → training on every train sample →
the final checkpoint reloaded by hash → development_val references frozen → one inference and one metric row per
development_val scan → patient-equal summary. A training failure, timeout or out-of-memory, or the plan's wall
budget running out (checked after training and before each development scan: no partial baseline), stops the
baseline at that phase; it is recorded, never retried. Private products stay under the private root;
render_baseline_md_v5 shows configuration, counts, metrics and hashes only. scientific_conclusion stays NOT_EVALUATED.

Command line: `plan-hash` prints the plan sha256 that approval C names; `baseline` reads approval C from the
approvals file under <scope_id>.G-BASE-<task>.resources, refuses before reading data when it is missing or names
another plan, and writes into a new --out directory (summary.json, module_baseline.md, the run and inferences).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .contracts_v5 import ActiveScope, V5Error, canonical_sha256
from .engine.data_v5 import load_training_sample_v5, original_geometry_v5
from .engine.module_eval_v5 import (
    freeze_module_reference_v5, module_metrics_v5, render_module_baseline_md_v5, summarize_module_metrics_v5)
from .engine.registration_v5 import (
    _is_number, _non_empty_str, register_run_v5, run_sha256_v5, select_samples_v5, validate_recipe_v5, write_run_v5)
from .engine.smoke_v5 import CANDIDATE_KEYS, _text
from .engine.train_v5 import load_checkpoint_v5, train_v5
from .evaluation.policy import validate_metric_policy_v5
from .gates_v5 import FIXTURE_APPROVERS, gate_task_v5
from .models.interface_v5 import build_model_input_v5
from .models.resunet_v5 import build_model_v5
from .paths import resolve_private_output, write_private_json
from .perception import infer_and_save_v5
from .real_t10_v5 import recipe_for_candidate_v5, smoke_plan_sha256_v5

PLAN_KEYS = ("plan_id", "scope_id", "scope_hash", "source", "lineage_id", "smoke", "recipe", "supervision",
             "acceptance_policy", "metric_policy", "budget")
SMOKE_KEYS = ("plan_sha256", "training_smoke_sha256", "chosen")
FREE_RECIPE_KEYS = ("recipe_id", "readable_roles", "budget", "checkpoint_every")
METRIC_POLICY_KEYS = ("metrics", "surface_tolerance_mm", "iou_match_threshold", "fp_budget_per_scan", "class_aware",
                      "size_strata_voxels")
BASELINE_ROLES = ["development_val", "train"]


def _bad(path: str) -> V5Error:
    return V5Error("INVALID_BASELINE_PLAN", path)


def validate_baseline_plan_v5(plan: dict, scope: ActiveScope, source: dict, smoke_plan: dict) -> dict:
    """The plan must be the passed smoke's configuration with the frozen candidate; see the module docstring."""
    if not isinstance(plan, dict) or set(plan) != set(PLAN_KEYS):
        raise _bad("$")
    if not _non_empty_str(plan["plan_id"]):
        raise _bad("$.plan_id")
    if plan["scope_id"] != scope.scope_id or plan["scope_hash"] != scope.scope_hash:
        raise V5Error("SMOKE_PLAN_SCOPE_MISMATCH", "$.scope_hash")
    if plan["source"] != source:
        raise V5Error("SMOKE_PLAN_SOURCE_MISMATCH", "$.source")
    smoke = plan["smoke"]
    if not isinstance(smoke, dict) or set(smoke) != set(SMOKE_KEYS):
        raise _bad("$.smoke")
    if smoke["plan_sha256"] != smoke_plan_sha256_v5(smoke_plan) or plan["lineage_id"] != smoke_plan["lineage_id"]:
        raise V5Error("SMOKE_PLAN_MISMATCH", "$.smoke.plan_sha256")
    if smoke["chosen"] not in smoke_plan["candidates"]:
        raise V5Error("BASELINE_CANDIDATE_UNKNOWN", "$.smoke.chosen")
    expected = recipe_for_candidate_v5(smoke_plan, smoke["chosen"])
    recipe = validate_recipe_v5(plan["recipe"])
    for key in sorted(set(expected) | set(recipe)):
        if key not in FREE_RECIPE_KEYS and recipe.get(key) != expected.get(key):
            raise V5Error("BASELINE_RECIPE_MISMATCH", f"$.recipe.{key}")
    if sorted(recipe["readable_roles"]) != BASELINE_ROLES:
        raise V5Error("BASELINE_ROLES_INVALID", "$.recipe.readable_roles")
    for key in ("supervision", "acceptance_policy"):
        if plan[key] != smoke_plan[key]:
            raise V5Error("BASELINE_POLICY_MISMATCH", f"$.{key}")
    metric = plan["metric_policy"]
    if not isinstance(metric, dict) or set(metric) != set(METRIC_POLICY_KEYS) or not metric["metrics"]:
        raise V5Error("METRIC_POLICY_INCOMPLETE", "$.metric_policy")
    if validate_metric_policy_v5(metric, "development")["blocked_metrics"]:
        raise V5Error("METRIC_POLICY_INCOMPLETE", "$.metric_policy")
    budget = plan["budget"]
    if not (isinstance(budget, dict) and set(budget) == {"max_wall_hours"} and _is_number(budget["max_wall_hours"])
            and budget["max_wall_hours"] >= recipe["budget"]["max_wall_hours"]):
        raise _bad("$.budget.max_wall_hours")
    return plan


def baseline_plan_sha256_v5(plan: dict) -> str:
    return canonical_sha256(plan)


def check_baseline_approval_v5(approval: object, plan_sha256: str, scope: ActiveScope, fixture_mode: bool) -> dict:
    """Approval C: the researcher's G-BASE resources approval for this scope names this exact baseline plan."""
    if not (isinstance(approval, dict) and _non_empty_str(approval.get("approved_by"))
            and _non_empty_str(approval.get("approved_at"))):
        raise V5Error("APPROVAL_MISSING", "$.approvals.resources")
    if approval.get("scope_hash") != scope.scope_hash:
        raise V5Error("GATE_BINDING_MISMATCH", "$.approvals.resources.scope_hash")
    if approval.get("baseline_plan_sha256") != plan_sha256:
        raise V5Error("BASELINE_PLAN_NOT_APPROVED", "$.approvals.resources.baseline_plan_sha256")
    if not fixture_mode and approval["approved_by"] in FIXTURE_APPROVERS:
        raise V5Error("GATE_FIXTURE_NOT_ALLOWED", "$.approvals.resources")
    return approval


def _check_smoke_evidence(evidence: object, plan: dict, scope: ActiveScope, source: dict) -> None:
    expected = {"status": "PASS", "sha256": plan["smoke"]["training_smoke_sha256"], "scope_hash": scope.scope_hash,
                "task": gate_task_v5(scope), **source}
    if not isinstance(evidence, dict) or any(evidence.get(k) != v for k, v in expected.items()):
        raise V5Error("SMOKE_EVIDENCE_MISMATCH", "$.smoke_evidence")


def _reference(sample, record, scope: ActiveScope) -> dict:
    """The released development reference on the original grid: truth and the supervised support (e_pos | e_neg)."""
    observation = sample.observation
    if observation.to_original or tuple(observation.grid["shape"]) != tuple(observation.original_shape):
        raise V5Error("BASELINE_GRID_NOT_ORIGINAL", "$.recipe.downsample_factor")
    tasks = {}
    for task in sorted(sample.targets):
        truth = sample.targets[task]
        truth = truth.astype(np.int16) if task == "A" else truth.astype(bool)
        tasks[task] = {"truth": truth, "eligible": np.asarray(sample.e_pos[task] | sample.e_neg[task], dtype=bool)}
    return {"patient_uid": record.patient_uid, "geometry": original_geometry_v5(observation, record), "tasks": tasks}


def run_baseline_v5(*, plan: dict, plan_sha256: str, scope: ActiveScope, source: dict, records: list,
                    registry: dict, raw: Path, label_maps: dict, gate_decision: dict, approval: dict,
                    smoke_evidence: dict, private_root: Path, repo: Path, device: str, run_id: str, created_at: str,
                    environment: dict, code: dict, fixture_mode: bool) -> dict:
    """One module baseline for one scope; returns the summary (also written privately as summary.json)."""
    started = time.monotonic()
    if baseline_plan_sha256_v5(plan) != plan_sha256:
        raise V5Error("BASELINE_PLAN_HASH_MISMATCH", "$.plan_sha256")
    check_baseline_approval_v5(approval, plan_sha256, scope, fixture_mode)
    _check_smoke_evidence(smoke_evidence, plan, scope, source)
    recipe, private_root, device_t = plan["recipe"], Path(private_root), torch.device(device)
    gate_id = f"G-BASE-{gate_task_v5(scope)}"
    run = register_run_v5(run_id=run_id, lineage_id=plan["lineage_id"], run_kind="baseline", recipe=recipe,
                          supervision=plan["supervision"], scope=scope, source=source, split_registry=registry,
                          records=records, gate_decisions={gate_id: gate_decision},
                          approvals={"resources": approval}, parents=[], environment=environment, code=code,
                          created_at=created_at, fixture_mode=fixture_mode)
    summary = {"scope_id": scope.scope_id, "plan_id": plan["plan_id"], "plan_sha256": plan_sha256,
               "task": gate_task_v5(scope), "device": {"name": "CPU", "count": 1},
               "chosen": {key: recipe[key] for key in CANDIDATE_KEYS},
               "smoke": {key: plan["smoke"][key] for key in ("plan_sha256", "training_smoke_sha256")},
               "run": {"run_id": run_id, "run_sha256": run_sha256_v5(run), "recipe_sha256": run["recipe_sha256"],
                       "supervision_sha256": run["supervision_sha256"],
                       "data_manifest_sha256": run["data_manifest_sha256"],
                       "split_sha256": run["split"]["split_sha256"]},
               "status": "FAILED", "stop_code": None, "stop_phase": None, "training": None, "module": None,
               "elapsed_hours": None, "budget_hours": plan["budget"]["max_wall_hours"]}
    if device_t.type == "cuda":
        summary["device"] = {"name": torch.cuda.get_device_name(device_t), "count": 1}

    def finish(stop_code: str | None, phase: str | None) -> dict:
        summary["stop_code"], summary["stop_phase"] = stop_code, phase
        summary["elapsed_hours"] = (time.monotonic() - started) / 3600.0
        write_private_json(resolve_private_output(private_root / "summary.json", private_root, repo), summary)
        return summary

    def over_budget() -> bool:
        return time.monotonic() - started > plan["budget"]["max_wall_hours"] * 3600.0

    run_dir = write_run_v5(run, registry, private_root, repo)
    by_uid = {record.sample_uid: record for record in records}
    regions = {"review": {}, "visibility": {}}
    # 1. every train sample, random initialisation, selection rule `last`
    train = {r.sample_uid: load_training_sample_v5(raw, r, scope, plan["supervision"], recipe, label_maps, regions)
             for r in select_samples_v5(records, registry, ["train"])}
    model = build_model_v5(recipe, scope).to(device_t)
    result = train_v5(run, model, train, recipe, plan["supervision"], run_dir, private_root, repo, None)
    summary["training"] = {"status": result["status"], "step": result["step"], "samples": len(train),
                           "checkpoint_sha256": result["last_checkpoint_sha256"],
                           "failure_code": result["failure_code"]}
    del train
    if result["status"] != "COMPLETED":
        return finish(result["failure_code"], "training")
    if over_budget():
        return finish("BASELINE_BUDGET_EXCEEDED", "training")
    state = load_checkpoint_v5(result["last_checkpoint"], run, result["last_checkpoint_sha256"], device_t.type)
    if state["step"] != result["step"]:
        raise V5Error("CHECKPOINT_NOT_FINAL", "$.training.checkpoint")
    model.load_state_dict(state["model"])
    # 2. development_val references, frozen before any prediction
    development = select_samples_v5(records, registry, ["development_val"])
    observations, references = {}, {}
    for record in development:
        sample = load_training_sample_v5(raw, record, scope, plan["supervision"], recipe, label_maps, regions)
        references[record.sample_uid] = _reference(sample, record, scope)
        observations[record.sample_uid] = sample.observation
    frozen = freeze_module_reference_v5(references, scope, registry)
    del references
    # 3. one full-volume inference and one metric row per development_val scan
    rows = []
    for uid in sorted(observations):
        if over_budget():                       # no partial baseline: every development_val scan or none
            return finish("BASELINE_BUDGET_EXCEEDED", "evaluation")
        observation = observations.pop(uid)
        batch = build_model_input_v5(observation.channels, scope, observation.valid_fov, observation.grid,
                                     list(observation.to_original))
        output, _ = infer_and_save_v5(
            model, batch, scope, observation_uid=uid, checkpoint_sha256=result["last_checkpoint_sha256"],
            original_geometry=original_geometry_v5(observation, by_uid[uid]),
            acceptance_policy=plan["acceptance_policy"], window=tuple(recipe["inference"]["window"]),
            stride=tuple(recipe["inference"]["stride"]), precision=recipe["precision"],
            out_dir=private_root / "inference", private_root=private_root, repo=repo, run=run, registry=registry)
        rows.append(module_metrics_v5(output, frozen, plan["metric_policy"], plan["acceptance_policy"], run))
    summary["module"] = summarize_module_metrics_v5(rows)
    summary["status"] = "COMPLETED"
    return finish(None, None)


def render_baseline_md_v5(summary: dict) -> str:
    """Configuration, training counts and hashes, then the module report; no identifiers or paths."""
    run, training, chosen = summary["run"], summary["training"], summary["chosen"]
    lines = [f"# 基线运行：{_text(summary['scope_id'], '$.scope_id')}", "",
             "本报告只含配置、计数、指标与哈希；不含样本或患者 ID、路径或文件名。", "",
             f"- 计划：`{_text(summary['plan_id'], '$.plan_id')}`，sha256 `{summary['plan_sha256']}`",
             f"- 所依据的 smoke：计划 sha256 `{summary['smoke']['plan_sha256']}`；training_smoke sha256 "
             f"`{summary['smoke']['training_smoke_sha256']}`",
             f"- 结果：{summary['status']}" + (f"（停止：{summary['stop_code']}，阶段 {summary['stop_phase']}）"
                                              if summary["stop_code"] else ""),
             f"- 设备：{summary['device']['name']} × {summary['device']['count']}；用时 "
             f"{summary['elapsed_hours']:.3f} h（预算 {summary['budget_hours']:.2f} h）",
             "- 配置（smoke 按规则冻结）：patch " + "×".join(str(v) for v in chosen["patch_size"])
             + f"，batch {chosen['batch_size']}，{chosen['precision']}",
             f"- run_sha256 `{run['run_sha256']}`；recipe_sha256 `{run['recipe_sha256']}`；supervision_sha256 "
             f"`{run['supervision_sha256']}`",
             f"- data_manifest_sha256 `{run['data_manifest_sha256']}`；split_sha256 `{run['split_sha256']}`", ""]
    if training is not None:
        lines += ["## 训练", "", f"- 随机初始化；train 样本 {training['samples']}；完成步数 {training['step']}；"
                  f"状态 {training['status']}；选择规则 last",
                  f"- 最终检查点 sha256 `{training['checkpoint_sha256']}`", ""]
    if summary["module"] is not None:
        body = render_module_baseline_md_v5(summary["module"])
        lines += ["## 开发集评估", ""] + body.split("\n", 1)[1].lstrip("\n").splitlines()
    lines += ["", "- 读 train 与 development_val；locked_test 未读；`scientific_conclusion: NOT_EVALUATED`。"]
    return "\n".join(lines) + "\n"


def main(argv: list | None = None) -> int:
    """plan-hash: the baseline plan's sha256 for approval C. baseline: one module baseline into a new --out."""
    import argparse
    import json
    import os

    from .contracts_v5 import validate_sample_v5, validate_scope_v5
    from .engine.data_v5 import load_label_maps_v5
    from .engine.registration_v5 import capture_code_v5, capture_environment_v5
    from .real_t10_v5 import _load_yaml, validate_smoke_plan_v5
    from .source_policy_v5 import load_source_policies_v5, source_binding_v5
    from .splits_v5 import verify_split_registry_v5

    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(prog="anatobind.v41.real_baseline_v5")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan-hash", "baseline"):
        command = sub.add_parser(name)
        for key in ("--plan", "--smoke-plan", "--scope", "--policies", "--source-id"):
            command.add_argument(key, required=True)
        if name == "baseline":
            for key in ("--raw", "--records", "--split-registry", "--gbase-decision", "--smoke-evidence",
                        "--approvals", "--run-id", "--device", "--out"):
                command.add_argument(key, required=True)
    args = parser.parse_args(argv)
    scope = validate_scope_v5(_load_yaml(args.scope))
    source = source_binding_v5(load_source_policies_v5(Path(args.policies))[args.source_id])
    smoke_plan = validate_smoke_plan_v5(_load_yaml(args.smoke_plan), scope, source)
    plan = validate_baseline_plan_v5(_load_yaml(args.plan), scope, source, smoke_plan)
    digest = baseline_plan_sha256_v5(plan)
    if args.command == "plan-hash":
        print(digest)
        return 0
    # approval C is keyed by gate like the approvals template; checked before any data is read or written
    approval = (_load_yaml(args.approvals) or {}).get(scope.scope_id, {}).get(
        f"G-BASE-{gate_task_v5(scope)}", {}).get("resources")
    check_baseline_approval_v5(approval, digest, scope, fixture_mode=False)
    records = [validate_sample_v5(json.loads(line)) for line in Path(args.records).read_text().splitlines() if line]
    registry = verify_split_registry_v5(json.loads(Path(args.split_registry).read_text()))
    decision = json.loads(Path(args.gbase_decision).read_text())
    evidence = json.loads(Path(args.smoke_evidence).read_text())
    out = Path(args.out)
    out.mkdir(mode=0o700)                                                # never overwritten
    os.chmod(out, 0o700)
    summary = run_baseline_v5(plan=plan, plan_sha256=digest, scope=scope, source=source, records=records,
                              registry=registry, raw=Path(args.raw),
                              label_maps=load_label_maps_v5(repo / "configs" / "v5" / "label_maps_v5.yaml"),
                              gate_decision=decision, approval=approval, smoke_evidence=evidence, private_root=out,
                              repo=repo, device=args.device, run_id=args.run_id,
                              created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              environment=capture_environment_v5(), code=capture_code_v5(repo), fixture_mode=False)
    markdown = render_baseline_md_v5(summary)
    fd = os.open(out / "module_baseline.md", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(markdown)
    print(markdown, end="")
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
