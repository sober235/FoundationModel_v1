"""Unified plan and run record (Proposal_v5 §0.6; design 2026-11-10 §5.3, §5.6).

A unified plan has the values every protocol shares (model, seeds, optimizer, schedule, precision, budget, and
optionally a stop step before the end of the schedule) and one
member per protocol (scope, source, lineage, tasks, patch, sampling, inference, supervision, policies). Each
member gets an ordinary recipe (member_recipe_v5) and an ordinary registered run (register_run_v5), so the frozen
split, exposure ledger, gates and inference checks of a member are the existing ones. The unified run binds the
member runs to the plan's sha256. Nothing here reads data; write_unified_run_v5 writes each member's run and
exposure (existing write_run_v5) and then the unified record, never overwriting.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

from ..contracts_v5 import ActiveScope, V5Error, canonical_sha256
from ..evaluation.policy import validate_metric_policy_v5
from ..losses import BOX_SPEC_KEYS, BOX_SPEC_OPTIONAL_KEYS
from ..paths import PrivatePathError, resolve_private_output, write_private_json
from ..perception import validate_acceptance_policy_v5
from .registration_v5 import (
    RUN_SCHEMA_VERSION, _is_number, _is_sha256, _non_empty_str, _positive_int, run_sha256_v5, trainable_tasks_v5,
    validate_recipe_v5, validate_supervision_v5, write_run_v5)
from .smoke_v5 import RESUME_TOLERANCE_KEYS

PLAN_KEYS = ("plan_id", "shared", "members", "smoke", "budget")
SHARED_KEYS = ("model", "seeds", "optimizer", "scheduler", "precision", "selection_rule", "checkpoint_every", "budget")
SHARED_OPTIONAL_KEYS = ("stop_at_step",)    # absent: the run goes to budget.max_steps, the end of its schedule
SHARED_MODEL_KEYS = ("architecture", "widths", "strides", "blocks_per_stage", "stem_channels")
MEMBER_KEYS = ("scope_id", "scope_hash", "source", "lineage_id", "tasks", "patch_size", "batch_size",
               "downsample_factor", "grid_tolerance", "normalization", "sampling", "inference", "loss_weight",
               "supervision", "box_supervision", "acceptance_policy", "metric_policy", "finding_policy")
SMOKE_KEYS = ("steps", "pause_at", "checkpoint_every", "subset_count", "subset_salt", "tolerances", "max_wall_hours")
FINDING_POLICY_KEYS = ("host", "iobb_thresholds")
FINDING_HOSTS = ("majority_accepted_anatomy", "none")
RUN_KINDS = ("smoke", "baseline")
_PROTOCOL_ID = re.compile(r"[a-z][a-z0-9_]*")
_MEMBER_RECIPE_KEYS = ("patch_size", "batch_size", "downsample_factor", "grid_tolerance", "normalization", "sampling",
                       "inference")
_SHARED_RECIPE_KEYS = ("seeds", "optimizer", "scheduler", "precision", "selection_rule")


def _bad(path: str) -> V5Error:
    return V5Error("UNIFIED_PLAN_INVALID", path)


def _keys(value: object, keys: tuple, path: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise _bad(path)
    return value


def member_recipe_v5(plan: dict, protocol_id: str, run_kind: str) -> dict:
    """The ordinary recipe of one member: shared values plus the member's own; smoke reads train only."""
    if run_kind not in RUN_KINDS:
        raise V5Error("INVALID_RUN_KIND", "$.run_kind")
    if protocol_id not in plan["members"]:
        raise V5Error("PROTOCOL_UNKNOWN", "$.protocol_id")
    shared, member, smoke = plan["shared"], plan["members"][protocol_id], plan["smoke"]
    recipe = {"recipe_id": f"{plan['plan_id']}.{protocol_id}" + (".smoke" if run_kind == "smoke" else ""),
              "model": {**{key: shared["model"][key] for key in SHARED_MODEL_KEYS if key != "stem_channels"},
                        "tasks": list(member["tasks"])},
              **{key: shared[key] for key in _SHARED_RECIPE_KEYS},
              **{key: member[key] for key in _MEMBER_RECIPE_KEYS}}
    if run_kind == "smoke":
        recipe.update(readable_roles=["train"], checkpoint_every=smoke["checkpoint_every"],
                      budget={"max_steps": smoke["steps"], "max_wall_hours": smoke["max_wall_hours"]})
    else:
        recipe.update(readable_roles=["train", "development_val"], checkpoint_every=shared["checkpoint_every"],
                      budget=shared["budget"])
    return copy.deepcopy(recipe)


def _check_member(plan: dict, pid: str, scope: ActiveScope, source: dict) -> None:
    path = f"$.members.{pid}"
    member = _keys(plan["members"][pid], MEMBER_KEYS, path)
    if not isinstance(scope, ActiveScope):
        raise V5Error("INVALID_TYPE", f"$.scopes.{pid}")
    if member["scope_id"] != scope.scope_id or member["scope_hash"] != scope.scope_hash:
        raise V5Error("UNIFIED_SCOPE_MISMATCH", f"{path}.scope_hash")
    if member["source"] != source:
        raise V5Error("UNIFIED_SOURCE_MISMATCH", f"{path}.source")
    if not _non_empty_str(member["lineage_id"]):
        raise _bad(f"{path}.lineage_id")
    tasks, supervision = member["tasks"], member["supervision"]
    if not (isinstance(tasks, list) and tasks and len(set(tasks)) == len(tasks)
            and set(tasks) <= set(trainable_tasks_v5(scope)) and isinstance(supervision, dict)
            and isinstance(supervision.get("tasks"), dict) and set(supervision["tasks"]) == set(tasks)):
        raise _bad(f"{path}.tasks")
    if not (_is_number(member["loss_weight"]) and member["loss_weight"] > 0):
        raise _bad(f"{path}.loss_weight")
    box = member["box_supervision"]
    if box is not None:
        if not (isinstance(box, dict) and set(box) - set(BOX_SPEC_OPTIONAL_KEYS) == set(BOX_SPEC_KEYS)
                and "U" in tasks
                and list(member["downsample_factor"]) == [1, 1, 1] and box["kind"] == "box_mil"
                and _is_number(box["topk_fraction"]) and 0 < box["topk_fraction"] <= 1
                and ("negative_topk_fraction" not in box
                     or (_is_number(box["negative_topk_fraction"]) and 0 < box["negative_topk_fraction"] <= 1))
                and all(_is_number(box[key]) and box[key] >= 0 for key in ("positive_weight", "negative_weight"))
                and box["positive_weight"] + box["negative_weight"] > 0
                and (_is_sha256(box["box_policy_sha256"]) or _non_empty_str(box["box_policy_sha256"]))):
            raise _bad(f"{path}.box_supervision")
    finding = _keys(member["finding_policy"], FINDING_POLICY_KEYS, f"{path}.finding_policy")
    if finding["host"] not in FINDING_HOSTS or (finding["host"] == "majority_accepted_anatomy" and "A" not in tasks):
        raise _bad(f"{path}.finding_policy.host")
    thresholds = finding["iobb_thresholds"]
    if thresholds is not None and not (isinstance(thresholds, list) and thresholds
                                       and all(_is_number(t) and 0 < t <= 1 for t in thresholds)):
        raise _bad(f"{path}.finding_policy.iobb_thresholds")
    for run_kind in RUN_KINDS:
        validate_recipe_v5(member_recipe_v5(plan, pid, run_kind))
    validate_supervision_v5(supervision, scope)
    validate_acceptance_policy_v5(member["acceptance_policy"], scope)
    metric = member["metric_policy"]
    if not isinstance(metric, dict) or validate_metric_policy_v5(metric, "development")["blocked_metrics"]:
        raise _bad(f"{path}.metric_policy")


def validate_unified_plan_v5(plan: dict, scopes: dict, sources: dict) -> dict:
    """A plan is valid when every member is a valid ordinary run configuration bound to its scope and source."""
    _keys(plan, PLAN_KEYS, "$")
    if not _non_empty_str(plan["plan_id"]):
        raise _bad("$.plan_id")
    shared = plan["shared"]
    if not isinstance(shared, dict) or set(shared) - set(SHARED_OPTIONAL_KEYS) != set(SHARED_KEYS):
        raise _bad("$.shared")
    model = _keys(shared["model"], SHARED_MODEL_KEYS, "$.shared.model")
    if not _positive_int(model["stem_channels"]):
        raise _bad("$.shared.model.stem_channels")
    members = plan["members"]
    if not (isinstance(members, dict) and members and isinstance(scopes, dict) and isinstance(sources, dict)
            and set(members) == set(scopes) == set(sources)
            and all(isinstance(pid, str) and _PROTOCOL_ID.fullmatch(pid) for pid in members)):
        raise _bad("$.members")
    smoke = _keys(plan["smoke"], SMOKE_KEYS, "$.smoke")
    for key in ("steps", "pause_at", "checkpoint_every"):
        if not _positive_int(smoke[key]):
            raise _bad(f"$.smoke.{key}")
    if not (smoke["pause_at"] < smoke["steps"] and smoke["pause_at"] % smoke["checkpoint_every"] == 0):
        raise _bad("$.smoke.pause_at")
    if smoke["steps"] % smoke["checkpoint_every"]:
        raise _bad("$.smoke.steps")
    counts = smoke["subset_count"]
    if not (isinstance(counts, dict) and set(counts) == set(members) and all(_positive_int(v) for v in counts.values())):
        raise _bad("$.smoke.subset_count")
    if not _non_empty_str(smoke["subset_salt"]):
        raise _bad("$.smoke.subset_salt")
    tolerances = smoke["tolerances"]
    if not (isinstance(tolerances, dict) and set(tolerances) == set(RESUME_TOLERANCE_KEYS)
            and all(_is_number(v) and v >= 0 for v in tolerances.values())):
        raise _bad("$.smoke.tolerances")
    if not (_is_number(smoke["max_wall_hours"]) and smoke["max_wall_hours"] > 0):
        raise _bad("$.smoke.max_wall_hours")
    for pid in sorted(members):
        _check_member(plan, pid, scopes[pid], sources[pid])
    if "stop_at_step" in shared:        # after the members: their recipes have validated budget.max_steps
        stop = shared["stop_at_step"]
        if not (_positive_int(stop) and stop < shared["budget"]["max_steps"]):
            raise _bad("$.shared.stop_at_step")
    budget = plan["budget"]
    if not (isinstance(budget, dict) and set(budget) == {"max_wall_hours"} and _is_number(budget["max_wall_hours"])
            and budget["max_wall_hours"] >= shared["budget"]["max_wall_hours"]):
        raise _bad("$.budget.max_wall_hours")
    return plan


def unified_plan_sha256_v5(plan: dict) -> str:
    return canonical_sha256(plan)


def stop_step_v5(plan: dict) -> int | None:
    """The step a baseline run of the plan stops at when that is before the end of its schedule, else None.

    `shared.stop_at_step` ends the run early and nothing else: the learning-rate schedule still decays over
    budget.max_steps, so the steps that are run are the first steps of the full schedule."""
    return plan["shared"].get("stop_at_step")


def register_unified_run_v5(*, run_id: str, run_kind: str, plan: dict, plan_sha256: str, member_runs: dict,
                            environment: dict, code: dict, created_at: str, fixture_mode: bool) -> dict:
    """Pure: the unified run record binding one registered member run per protocol to the plan."""
    if run_kind not in RUN_KINDS:
        raise V5Error("INVALID_RUN_KIND", "$.run_kind")
    if not _non_empty_str(run_id) or not _non_empty_str(created_at) or not isinstance(fixture_mode, bool):
        raise V5Error("INVALID_TYPE", "$.run_id")
    if unified_plan_sha256_v5(plan) != plan_sha256:
        raise V5Error("UNIFIED_PLAN_HASH_MISMATCH", "$.plan_sha256")
    if not isinstance(member_runs, dict) or set(member_runs) != set(plan["members"]):
        raise V5Error("UNIFIED_MEMBER_MISMATCH", "$.member_runs")
    members = {}
    for pid in sorted(member_runs):
        run, member = member_runs[pid], plan["members"][pid]
        expected = {"run_id": f"{run_id}.{pid}", "run_kind": run_kind, "lineage_id": member["lineage_id"],
                    "recipe_sha256": canonical_sha256(member_recipe_v5(plan, pid, run_kind)),
                    "supervision_sha256": canonical_sha256(member["supervision"]),
                    "scope": {"scope_id": member["scope_id"], "scope_hash": member["scope_hash"]},
                    "source": member["source"]}
        if not isinstance(run, dict) or any(run.get(key) != value for key, value in expected.items()):
            raise V5Error("UNIFIED_MEMBER_MISMATCH", f"$.member_runs.{pid}")
        members[pid] = {"run_id": run["run_id"], "run_sha256": run_sha256_v5(run), "scope": dict(run["scope"]),
                        "source": dict(run["source"]), "split_sha256": run["split"]["split_sha256"]}
    return {"schema_version": RUN_SCHEMA_VERSION, "run_id": run_id, "run_kind": f"unified_{run_kind}",
            "plan_id": plan["plan_id"], "plan_sha256": plan_sha256, "members": members, "code": dict(code),
            "environment": copy.deepcopy(environment), "fixture_mode": fixture_mode, "created_at": created_at}


def write_unified_run_v5(unified_run: dict, member_runs: dict, registries: dict, private_root: Path,
                         repo: Path) -> Path:
    """Write every member's run and exposure (write_run_v5), then runs/<run_id>/run.json; return that directory."""
    run_path = resolve_private_output(Path(private_root) / "runs" / unified_run["run_id"] / "run.json",
                                      private_root, repo)
    if run_path.exists():
        raise PrivatePathError("PRIVATE_FILE_EXISTS")
    if set(member_runs) != set(unified_run["members"]) or set(registries) != set(member_runs):
        raise V5Error("UNIFIED_MEMBER_MISMATCH", "$.member_runs")
    for pid in sorted(member_runs):
        if run_sha256_v5(member_runs[pid]) != unified_run["members"][pid]["run_sha256"]:
            raise V5Error("UNIFIED_MEMBER_MISMATCH", f"$.member_runs.{pid}")
    for pid in sorted(member_runs):
        write_run_v5(member_runs[pid], registries[pid], private_root, repo)
    write_private_json(run_path, unified_run)
    return run_path.parent
