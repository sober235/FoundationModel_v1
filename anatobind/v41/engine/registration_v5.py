"""v5.1 run registration, lineage inheritance and frozen-split access (Proposal_v5 §3.3, §6.3; plan_v5 T09a).

A run is registered before its first training step. The record binds the scope, the source, the frozen split,
the selected data, the recipe and the supervision policy by hash, and lists the gate decisions that allow it.
Exposure follows the weights: a run initialised from checkpoints lists every ancestor lineage (L6), and the split
it reads must have been frozen for that lineage and those ancestors. Samples are read only through the frozen
registry's assignment, never through a record's own split field (L4). Errors are V5Error(code, path) and never
echo input values.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import platform
import re
import subprocess
from pathlib import Path

from ..contracts_v5 import (UNKNOWN_KEY, ActiveScope, SampleV5, V5Error, _check_schema, canonical_sha256,
                            validate_sample_v5)
from ..eligibility import TRACKS_V5
from ..exposure_v5 import locked_test_blockers_v5
from ..gates_v5 import FIXTURE_APPROVERS, gate_task_v5
from ..paths import PrivatePathError, resolve_private_output, write_private_json
from ..splits_v5 import verify_split_registry_v5

RUN_SCHEMA_VERSION = "v5.1"
RUN_KINDS = ("smoke", "baseline")
READABLE_ROLES = ("train", "development_val")
EXPOSURE_ROLE_BY_SPLIT = {"train": "training", "development_val": "model_selection"}
TASK_KINDS = {"A": "softmax", "U": "sigmoid"}
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_HEX = set("0123456789abcdef")
_SEED_LIMIT = 2 ** 63


# ---------------------------------------------------------------------------
# Value checks


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _positive_int(value: object) -> bool:
    return _is_int(value) and value > 0


def _seed(value: object) -> bool:
    return _is_int(value) and 0 <= value < _SEED_LIMIT


def _vector3(value: object) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(_positive_int(v) for v in value)


def _unique_strings(value: object, allowed: tuple) -> bool:
    return (isinstance(value, list) and bool(value) and all(isinstance(v, str) and v in allowed for v in value)
            and len(set(value)) == len(value))


def _one_of(*allowed: str):
    return lambda value: isinstance(value, str) and value in allowed


_RECIPE_SPEC = {
    "recipe_id": _non_empty_str,
    "model": {
        "architecture": _one_of("resunet3d"),
        "tasks": lambda v: _unique_strings(v, ("A", "U")),
        "widths": lambda v: isinstance(v, list) and bool(v) and all(_positive_int(x) for x in v),
        "strides": lambda v: isinstance(v, list) and bool(v) and all(_vector3(x) for x in v),
        "blocks_per_stage": _positive_int,
    },
    "seeds": {"init": _seed, "sampler": _seed},
    "patch_size": _vector3,
    "batch_size": _positive_int,
    "precision": _one_of("fp32", "amp_bf16", "amp_fp16"),
    "optimizer": {
        "name": _one_of("adamw"),
        "lr": lambda v: _is_number(v) and v > 0,
        "weight_decay": lambda v: _is_number(v) and v >= 0,
    },
    "scheduler": {"name": _one_of("poly"), "exponent": lambda v: _is_number(v) and v >= 0},
    "downsample_factor": _vector3,
    "normalization": _one_of("zscore_volume"),
    "sampling": {"positive_fraction": lambda v: _is_number(v) and 0 <= v <= 1},
    "readable_roles": lambda v: _unique_strings(v, READABLE_ROLES) and "train" in v,
    "selection_rule": _one_of("last"),
    "inference": {"window": _vector3, "stride": _vector3},
    "checkpoint_every": _positive_int,
    "budget": {"max_steps": _positive_int, "max_wall_hours": lambda v: _is_number(v) and v > 0},
}

_SUPERVISION_SPEC = {
    "policy_id": _non_empty_str,
    "scope_id": _non_empty_str,
    "reference_policy_id": _non_empty_str,
    "track": lambda v: isinstance(v, str) and v in TRACKS_V5,
    "allow_unknown_visibility": lambda v: isinstance(v, bool),
    "channel": _one_of("expert"),
}

_TASK_SPEC = {
    "kind": lambda v: isinstance(v, str),
    "ce_weight": lambda v: _is_number(v) and v >= 0,
    "dice_weight": lambda v: _is_number(v) and v >= 0,
    "task_weight": lambda v: _is_number(v) and v > 0,
}


def _walk(value: object, spec, path: str, missing: str, invalid: str) -> None:
    """Every key of the spec must be present and non-null; keys outside the spec are unknown."""
    if value is None:
        raise V5Error(missing, path)
    if isinstance(spec, dict):
        if not isinstance(value, dict):
            raise V5Error(invalid, path)
        if any(key not in spec for key in value):
            raise V5Error("UNKNOWN_FIELD", f"{path}.{UNKNOWN_KEY}")
        for key, sub in spec.items():
            _walk(value.get(key), sub, f"{path}.{key}", missing, invalid)
    elif not spec(value):
        raise V5Error(invalid, path)


def trainable_tasks_v5(scope: ActiveScope) -> tuple[str, ...]:
    """Tasks a scope can supervise: A needs anatomy classes, U needs abnormality classes."""
    return tuple(task for task, classes in (("A", scope.anatomy_classes), ("U", scope.abnormality_classes))
                 if classes)


def validate_recipe_v5(recipe: dict) -> dict:
    if not isinstance(recipe, dict):
        raise V5Error("INVALID_TYPE", "$")
    _walk(recipe, _RECIPE_SPEC, "$", "RECIPE_VALUE_MISSING", "INVALID_RECIPE_VALUE")
    if len(recipe["model"]["strides"]) != len(recipe["model"]["widths"]):
        raise V5Error("INVALID_RECIPE_VALUE", "$.model.strides")
    if any(s > w for s, w in zip(recipe["inference"]["stride"], recipe["inference"]["window"])):
        raise V5Error("INVALID_RECIPE_VALUE", "$.inference.stride")
    return recipe


def validate_supervision_v5(policy: dict, scope: ActiveScope) -> dict:
    if not isinstance(scope, ActiveScope):
        raise V5Error("INVALID_TYPE", "$.scope")
    if not isinstance(policy, dict):
        raise V5Error("INVALID_TYPE", "$")
    if any(key not in _SUPERVISION_SPEC and key != "tasks" for key in policy):
        raise V5Error("UNKNOWN_FIELD", f"$.{UNKNOWN_KEY}")
    for key, spec in _SUPERVISION_SPEC.items():
        _walk(policy.get(key), spec, f"$.{key}", "SUPERVISION_VALUE_MISSING", "INVALID_SUPERVISION_VALUE")
    tasks = policy.get("tasks")
    if tasks is None:
        raise V5Error("SUPERVISION_VALUE_MISSING", "$.tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise V5Error("INVALID_SUPERVISION_VALUE", "$.tasks")
    if any(task not in TASK_KINDS for task in tasks):
        raise V5Error("UNKNOWN_FIELD", f"$.tasks.{UNKNOWN_KEY}")
    trainable = trainable_tasks_v5(scope)
    for task, entry in tasks.items():
        path = f"$.tasks.{task}"
        _walk(entry, _TASK_SPEC, path, "SUPERVISION_VALUE_MISSING", "INVALID_SUPERVISION_VALUE")
        if entry["kind"] != TASK_KINDS[task]:
            raise V5Error("INVALID_SUPERVISION_VALUE", f"{path}.kind")
        if entry["ce_weight"] + entry["dice_weight"] <= 0 or task not in trainable:
            raise V5Error("INVALID_SUPERVISION_VALUE", path)
    for key in ("scope_id", "reference_policy_id"):
        if policy[key] != getattr(scope, key):
            raise V5Error("SUPERVISION_SCOPE_MISMATCH", f"$.{key}")
    if policy["track"] == "observable_target" and policy["allow_unknown_visibility"]:
        raise V5Error("INVALID_SUPERVISION_VALUE", "$.allow_unknown_visibility")
    return policy


# ---------------------------------------------------------------------------
# Frozen-split access and lineage


def select_samples_v5(records: list[SampleV5], registry: dict, roles: list[str]) -> list[SampleV5]:
    """Records whose frozen assignment is one of `roles`; `record.split` is never read (L4)."""
    verify_split_registry_v5(registry)
    if not isinstance(roles, list) or not roles:
        raise V5Error("INVALID_ROLE", "$.roles")
    wanted: set[str] = set()
    for index, role in enumerate(roles):
        if role == "locked_test":
            raise V5Error("LOCKED_TEST_NOT_READABLE", f"$.roles[{index}]")
        if role not in READABLE_ROLES or role in wanted:
            raise V5Error("INVALID_ROLE", f"$.roles[{index}]")
        wanted.add(role)
    if not isinstance(records, list):
        raise V5Error("INVALID_TYPE", "$")
    assignment = registry["assignment"]
    seen: set[str] = set()
    selected = []
    for index, record in enumerate(records):
        if not isinstance(record, SampleV5):
            raise V5Error("INVALID_TYPE", f"$[{index}]")
        if record.sample_uid in seen:
            raise V5Error("DUPLICATE_SAMPLE", f"$[{index}]")
        seen.add(record.sample_uid)
        split = assignment.get(record.sample_uid)
        if split is None:
            raise V5Error("SPLIT_ASSIGNMENT_MISSING", f"$[{index}].sample_uid")
        if split in wanted:
            selected.append(record)
    return selected


def inherit_lineage_v5(parents: list[dict], lineage_id: str) -> list[str]:
    """Union of every parent's lineage and ancestors, minus `lineage_id` itself (Proposal_v5 §3.3)."""
    locked_test_blockers_v5([], lineage_id)
    if not isinstance(parents, list):
        raise V5Error("INVALID_TYPE", "$.parents")
    inherited: set[str] = set()
    for index, parent in enumerate(parents):
        path = f"$.parents[{index}]"
        if not isinstance(parent, dict):
            raise V5Error("INVALID_TYPE", path)
        run = parent.get("run")
        if not isinstance(run, dict):
            raise V5Error("PARENT_RUN_REQUIRED", f"{path}.run")
        _check_schema(run, "run", f"{path}.run")
        if run["run_kind"] == "smoke":
            raise V5Error("SMOKE_PARENT_NOT_ALLOWED", f"{path}.run.run_kind")
        if not _is_sha256(parent.get("checkpoint_sha256")):
            raise V5Error("INVALID_SHA256", f"{path}.checkpoint_sha256")
        for field, names in (("lineage_id", [run["lineage_id"]]), ("ancestor_lineages", run["ancestor_lineages"])):
            if any(not name.strip() for name in names):
                raise V5Error("INVALID_LINEAGE", f"{path}.run.{field}")
            inherited.update(name.strip() for name in names)
    inherited.discard(lineage_id.strip())
    return sorted(inherited)


def data_manifest_sha256_v5(records: list[SampleV5]) -> str:
    """Canonical hash of the records a run reads, independent of their order."""
    if not isinstance(records, list) or not all(isinstance(record, SampleV5) for record in records):
        raise V5Error("INVALID_TYPE", "$.records")
    return canonical_sha256(sorted((dataclasses.asdict(record) for record in records),
                                   key=lambda body: body["sample_uid"]))


# ---------------------------------------------------------------------------
# Environment and code state


def capture_environment_v5() -> dict:
    import numpy
    import torch

    devices = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
               if torch.cuda.is_available() else [])
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(numpy.__version__),
            "cuda": None if torch.version.cuda is None else str(torch.version.cuda), "devices": devices}


def capture_code_v5(repo: Path) -> dict:
    """Commit and dirty flag of the code that will run; untracked files count as dirty."""
    git = ["git", "-C", str(repo)]
    try:
        commit = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout
        status = subprocess.run(git + ["status", "--porcelain"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        raise V5Error("CODE_STATE_UNKNOWN", "$.code") from None
    if not commit.strip():
        raise V5Error("CODE_STATE_UNKNOWN", "$.code")
    return {"git_commit": commit.strip(), "dirty": bool(status.strip())}


# ---------------------------------------------------------------------------
# Registration


def _check_keys_are_strings(mapping: object, path: str) -> dict:
    if not isinstance(mapping, dict) or not all(isinstance(key, str) for key in mapping):
        raise V5Error("INVALID_TYPE", path)
    return mapping


def _gate_records(gate_decisions: dict, scope: ActiveScope, source: dict, required: list[str],
                  fixture_mode: bool) -> list[dict]:
    """Every presented decision must be bound to this scope and source and PASS; required ones must be present."""
    _check_keys_are_strings(gate_decisions, "$.gate_decisions")
    expected = {"scope_id": scope.scope_id, "scope_hash": scope.scope_hash, "source_id": source["source_id"],
                "source_version": source["source_version"], "task": gate_task_v5(scope)}
    records = []
    for gate_id in sorted(set(gate_decisions) | set(required)):
        path = f"$.gate_decisions.{gate_id}"
        decision = gate_decisions.get(gate_id)
        if decision is None:
            raise V5Error("GATE_NOT_PASSED", path)
        if not isinstance(decision, dict):
            raise V5Error("INVALID_TYPE", path)
        if decision.get("gate_id") != gate_id or any(decision.get(k) != v for k, v in expected.items()):
            raise V5Error("GATE_BINDING_MISMATCH", path)
        if not fixture_mode and (decision.get("fixture_mode") is not False or decision.get("fixture_approvals")):
            raise V5Error("GATE_FIXTURE_NOT_ALLOWED", path)
        if decision.get("engineering_status") != "PASS":
            raise V5Error("GATE_NOT_PASSED", path)
        records.append({"gate_id": gate_id, "decision_sha256": canonical_sha256(decision)})
    return records


def _check_approvals(approvals: dict, scope: ActiveScope, required: tuple, fixture_mode: bool) -> None:
    _check_keys_are_strings(approvals, "$.approvals")
    for key in sorted(set(approvals) | set(required)):
        path = f"$.approvals.{key}"
        item = approvals.get(key)
        if not (isinstance(item, dict) and _non_empty_str(item.get("approved_by"))
                and _non_empty_str(item.get("approved_at"))):
            raise V5Error("APPROVAL_MISSING", path)
        if item.get("scope_hash") != scope.scope_hash:
            raise V5Error("GATE_BINDING_MISMATCH", path)
        if not fixture_mode and item["approved_by"] in FIXTURE_APPROVERS:
            raise V5Error("GATE_FIXTURE_NOT_ALLOWED", path)


def register_run_v5(*, run_id: str, lineage_id: str, run_kind: str, recipe: dict, supervision: dict,
                    scope: ActiveScope, source: dict, split_registry: dict, records: list[SampleV5],
                    gate_decisions: dict, approvals: dict, parents: list[dict], environment: dict, code: dict,
                    created_at: str, fixture_mode: bool) -> dict:
    """Pure: build and validate the run record; nothing is written (write_run_v5 does that)."""
    if not isinstance(fixture_mode, bool):
        raise V5Error("INVALID_TYPE", "$.fixture_mode")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise V5Error("INVALID_RUN_ID", "$.run_id")
    if run_kind not in RUN_KINDS:
        raise V5Error("INVALID_RUN_KIND", "$.run_kind")
    if not isinstance(scope, ActiveScope):
        raise V5Error("INVALID_TYPE", "$.scope")
    if not (isinstance(source, dict) and set(source) == {"source_id", "source_version"}
            and all(_non_empty_str(value) for value in source.values())):
        raise V5Error("INVALID_SOURCE", "$.source")
    if not _non_empty_str(created_at):
        raise V5Error("INVALID_TYPE", "$.created_at")
    validate_recipe_v5(recipe)
    validate_supervision_v5(supervision, scope)
    if not (isinstance(code, dict) and set(code) == {"git_commit", "dirty"} and _non_empty_str(code["git_commit"])
            and isinstance(code["dirty"], bool)):
        raise V5Error("INVALID_TYPE", "$.code")
    if code["dirty"] and not fixture_mode:
        raise V5Error("CODE_DIRTY", "$.code.dirty")
    tasks = set(recipe["model"]["tasks"])
    if not tasks <= set(trainable_tasks_v5(scope)) or tasks != set(supervision["tasks"]):
        raise V5Error("MODEL_TASKS_INVALID", "$.recipe.model.tasks")
    roles = sorted(recipe["readable_roles"])
    if run_kind == "smoke" and roles != ["train"]:
        raise V5Error("SMOKE_ROLES_INVALID", "$.recipe.readable_roles")
    ancestors = inherit_lineage_v5(parents, lineage_id)
    verify_split_registry_v5(split_registry)
    if split_registry["lineage_id"] != lineage_id:
        raise V5Error("SPLIT_LINEAGE_MISMATCH", "$.split_registry.lineage_id")
    if not set(ancestors) <= set(split_registry["ancestor_lineages"]):
        raise V5Error("SPLIT_LINEAGE_MISMATCH", "$.split_registry.ancestor_lineages")
    selected = select_samples_v5(records, split_registry, roles)
    required = ["G-DATA"] if run_kind == "smoke" else [f"G-BASE-{task}" for task in sorted(tasks)]
    gates = _gate_records(gate_decisions, scope, source, required, fixture_mode)
    _check_approvals(approvals, scope, ("resources",) if run_kind == "smoke" else (), fixture_mode)
    run = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": run_kind,
        "lineage_id": lineage_id,
        "ancestor_lineages": ancestors,
        "parents": [{"run_id": p["run"]["run_id"], "run_sha256": run_sha256_v5(p["run"]),
                     "checkpoint_sha256": p["checkpoint_sha256"], "lineage_id": p["run"]["lineage_id"]}
                    for p in parents],
        "scope": {"scope_id": scope.scope_id, "scope_hash": scope.scope_hash},
        "source": {"source_id": source["source_id"], "source_version": source["source_version"]},
        "split": {"split_sha256": split_registry["split_sha256"], "readable_roles": roles},
        "data_manifest_sha256": data_manifest_sha256_v5(selected),
        "supervision_sha256": canonical_sha256(supervision),
        "recipe_sha256": canonical_sha256(recipe),
        "code": dict(code),
        "environment": copy.deepcopy(environment),
        "gate_decisions": gates,
        "approvals": copy.deepcopy(approvals),
        "fixture_mode": fixture_mode,
        "created_at": created_at,
    }
    _check_schema(run, "run")
    return run


def run_sha256_v5(run: dict) -> str:
    return canonical_sha256(run)


# ---------------------------------------------------------------------------
# Exposure written at registration (L6)


def run_exposure_entries_v5(run: dict, registry: dict) -> dict[str, list[dict]]:
    """T08a ledger entries for every sample the run may read: train → training, development_val → model_selection."""
    _check_schema(run, "run")
    verify_split_registry_v5(registry)
    if registry["split_sha256"] != run["split"]["split_sha256"]:
        raise V5Error("SPLIT_HASH_MISMATCH", "$.split.split_sha256")
    readable = set(run["split"]["readable_roles"])
    evidence = f"run_sha256:{run_sha256_v5(run)}"
    return {uid: [{"experiment_id": run["run_id"], "role": EXPOSURE_ROLE_BY_SPLIT[split],
                   "recorded_at": run["created_at"], "lineage_id": run["lineage_id"], "evidence": evidence}]
            for uid, split in sorted(registry["assignment"].items()) if split in readable}


def run_exposure_file_v5(run: dict, registry: dict) -> dict:
    """Content of runs/<run_id>/exposure.json; run_id lets a merge refuse the same run twice."""
    entries = run_exposure_entries_v5(run, registry)
    return {"schema_version": RUN_SCHEMA_VERSION, "run_id": run["run_id"], "run_sha256": run_sha256_v5(run),
            "entries": entries}


def merge_run_exposure_v5(records: list[SampleV5], exposure_files: list[dict]) -> list[SampleV5]:
    """Copies of `records` with each run's exposure appended, re-validated; inputs are not changed."""
    if not isinstance(records, list):
        raise V5Error("INVALID_TYPE", "$.records")
    if not isinstance(exposure_files, list):
        raise V5Error("INVALID_TYPE", "$")
    bodies: dict[str, dict] = {}
    for index, record in enumerate(records):
        if not isinstance(record, SampleV5):
            raise V5Error("INVALID_TYPE", f"$.records[{index}]")
        if record.sample_uid in bodies:
            raise V5Error("DUPLICATE_SAMPLE", f"$.records[{index}]")
        bodies[record.sample_uid] = dataclasses.asdict(record)
    runs: set[str] = set()
    for index, exposure in enumerate(exposure_files):
        path = f"$[{index}]"
        if not (isinstance(exposure, dict) and _non_empty_str(exposure.get("run_id"))
                and isinstance(exposure.get("entries"), dict)):
            raise V5Error("INVALID_EXPOSURE_FILE", path)
        if exposure["run_id"] in runs:
            raise V5Error("DUPLICATE_EXPOSURE_FILE", f"{path}.run_id")
        runs.add(exposure["run_id"])
        for uid, entries in exposure["entries"].items():
            if uid not in bodies:
                raise V5Error("EXPOSURE_SAMPLE_UNKNOWN", f"{path}.entries")
            if not isinstance(entries, list):
                raise V5Error("INVALID_EXPOSURE_FILE", f"{path}.entries")
            bodies[uid]["exposure_history"].extend(copy.deepcopy(entries))
    return [validate_sample_v5(bodies[record.sample_uid]) for record in records]


def write_run_v5(run: dict, registry: dict, private_root: Path, repo: Path) -> Path:
    """Write runs/<run_id>/exposure.json then run.json under the private root, never overwriting; return the dir.

    Both are written before any training step, so a run that fails midway has already recorded its exposure.
    """
    exposure = run_exposure_file_v5(run, registry)
    if not _RUN_ID.fullmatch(run["run_id"]):
        raise V5Error("INVALID_RUN_ID", "$.run_id")
    run_dir = Path(private_root) / "runs" / run["run_id"]
    run_path = resolve_private_output(run_dir / "run.json", private_root, repo)
    exposure_path = resolve_private_output(run_dir / "exposure.json", private_root, repo)
    if run_path.exists() or exposure_path.exists():
        raise PrivatePathError("PRIVATE_FILE_EXISTS")
    write_private_json(exposure_path, exposure)
    write_private_json(run_path, run)
    return run_path.parent
