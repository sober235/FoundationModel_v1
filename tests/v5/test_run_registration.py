"""v5.1 run registration, lineage inheritance and frozen-split access (plan_v5 T09a; Proposal_v5 §3.3, §6.3)."""
from __future__ import annotations

import copy
import json
import math
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from anatobind.v41.contracts_v5 import V5Error, canonical_sha256, validate_scope_v5
from anatobind.v41.engine.registration_v5 import (
    capture_code_v5, capture_environment_v5, data_manifest_sha256_v5, inherit_lineage_v5, merge_run_exposure_v5,
    run_exposure_entries_v5, run_exposure_file_v5, run_sha256_v5, select_samples_v5, validate_recipe_v5,
    validate_supervision_v5, write_run_v5)
from anatobind.v41.gates_v5 import evaluate_gate_v5
from anatobind.v41.paths import PrivatePathError
from anatobind.v41.splits_v5 import split_registry_sha256_v5, verify_split_registry_v5
from tests.v5.factories import (
    CONFIG_DIR_V5, FIXTURE_APPROVER, SYNTH_SOURCE, make_approvals, make_gate_decisions, make_module_scope,
    make_recipe, make_run, make_split_records, make_split_registry, make_supervision)

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = "c" * 64


def _scope(task="A", **overrides):
    return validate_scope_v5(make_module_scope(task, **overrides))


def _au_scope():
    return _scope("A", scope_id="synthetic_au", abnormality_classes=["target_a"])


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _real(decisions):
    """Decisions shaped as a non-fixture gate run issues them; no approver name is invented."""
    return {gate_id: dict(decision, fixture_mode=False, fixture_approvals=[]) for gate_id, decision in decisions.items()}


def _train_dev_run(records, registry, **overrides):
    return make_run(records=records, split_registry=registry,
                    recipe=make_recipe(readable_roles=["train", "development_val"]), **overrides)


def test_split_registry_hash_is_verified():
    records = make_split_records(["train", "development_val", "locked_test"])
    registry = make_split_registry(records, ancestor_lineages=["lin-0"])
    assert registry["split_sha256"] == split_registry_sha256_v5(registry["assignment"], "lin-a", ["lin-0"],
                                                                registry["policy"])
    assert verify_split_registry_v5(registry) == registry
    for key, value in (("assignment", dict(registry["assignment"], **{"s-0": "locked_test"})),
                       ("lineage_id", "lin-b"), ("ancestor_lineages", []),
                       ("policy", dict(registry["policy"], exposed_groups="train"))):
        _raises("SPLIT_HASH_MISMATCH", "$.split_sha256", verify_split_registry_v5, dict(registry, **{key: value}))
    for key in ("assignment", "lineage_id", "ancestor_lineages", "policy", "split_sha256"):
        _raises("INVALID_SPLIT_REGISTRY", f"$.{key}", verify_split_registry_v5,
                {k: v for k, v in registry.items() if k != key})
    _raises("INVALID_SPLIT_REGISTRY", "$.assignment", verify_split_registry_v5,
            dict(registry, assignment={"s-0": "holdout"}))


def test_selection_reads_assignment_not_sample_split():
    records = make_split_records(["train", "train", "development_val"])
    official = {"synthetic:p0000": "train", "synthetic:p0001": "test", "synthetic:p0002": "val"}
    registry = make_split_registry(records, official_split=official)
    assert records[1].split == "train" and registry["assignment"]["s-1"] == "locked_test"
    assert [r.sample_uid for r in select_samples_v5(records, registry, ["train"])] == ["s-0"]
    assert [r.sample_uid for r in select_samples_v5(records, registry, ["development_val", "train"])] == ["s-0", "s-2"]
    _raises("LOCKED_TEST_NOT_READABLE", "$.roles[1]", select_samples_v5, records, registry, ["train", "locked_test"])
    for roles, path in ((["unassigned"], "$.roles[0]"), (["excluded"], "$.roles[0]"),
                        (["train", "train"], "$.roles[1]"), ([], "$.roles"), ("train", "$.roles")):
        _raises("INVALID_ROLE", path, select_samples_v5, records, registry, roles)


def test_unassigned_record_rejected():
    records = make_split_records(["train", "train"])
    registry = make_split_registry(records[:1])
    _raises("SPLIT_ASSIGNMENT_MISSING", "$[1].sample_uid", select_samples_v5, records, registry, ["train"])
    tampered = dict(registry, assignment={"s-0": "train", "s-1": "train"})
    _raises("SPLIT_HASH_MISMATCH", "$.split_sha256", select_samples_v5, records, tampered, ["train"])


def test_lineage_inherits_all_ancestors():
    grand = make_run(run_id="run-g", lineage_id="lin-g")
    parent = make_run(run_id="run-p", lineage_id="lin-p", parents=[{"run": grand, "checkpoint_sha256": CHECKPOINT}])
    child = make_run(run_id="run-c", lineage_id="lin-c", parents=[{"run": parent, "checkpoint_sha256": CHECKPOINT}])
    assert (grand["ancestor_lineages"], parent["ancestor_lineages"], child["ancestor_lineages"]) == (
        [], ["lin-g"], ["lin-g", "lin-p"])
    assert child["parents"] == [{"run_id": "run-p", "run_sha256": run_sha256_v5(parent),
                                 "checkpoint_sha256": CHECKPOINT, "lineage_id": "lin-p"}]
    other = make_run(run_id="run-o", lineage_id="lin-o")
    merged = [{"run": parent, "checkpoint_sha256": CHECKPOINT}, {"run": other, "checkpoint_sha256": "d" * 64}]
    assert inherit_lineage_v5(merged, "lin-m") == ["lin-g", "lin-o", "lin-p"]
    assert inherit_lineage_v5(merged, "lin-p") == ["lin-g", "lin-o"]
    assert inherit_lineage_v5([], "lin-m") == []


def test_split_registry_must_cover_run_lineage():
    records = make_split_records(["train", "development_val", "locked_test"])
    parents = [{"run": make_run(run_id="run-g", lineage_id="lin-g", records=records), "checkpoint_sha256": CHECKPOINT}]
    _raises("SPLIT_LINEAGE_MISMATCH", "$.split_registry.ancestor_lineages", make_run, run_id="run-c",
            lineage_id="lin-c", parents=parents, records=records,
            split_registry=make_split_registry(records, lineage_id="lin-c"))
    _raises("SPLIT_LINEAGE_MISMATCH", "$.split_registry.lineage_id", make_run, run_id="run-c", lineage_id="lin-c",
            parents=parents, records=records,
            split_registry=make_split_registry(records, lineage_id="lin-x", ancestor_lineages=["lin-g"]))


def test_parent_without_run_record_or_smoke_parent_rejected():
    for parent in ({"run": None, "checkpoint_sha256": CHECKPOINT}, {"checkpoint_sha256": CHECKPOINT}):
        _raises("PARENT_RUN_REQUIRED", "$.parents[0].run", inherit_lineage_v5, [parent], "lin-c")
    smoke = make_run(run_id="run-s", lineage_id="lin-s", run_kind="smoke")
    _raises("SMOKE_PARENT_NOT_ALLOWED", "$.parents[0].run.run_kind", inherit_lineage_v5,
            [{"run": smoke, "checkpoint_sha256": CHECKPOINT}], "lin-c")
    baseline = make_run(run_id="run-b", lineage_id="lin-b")
    _raises("INVALID_SHA256", "$.parents[0].checkpoint_sha256", inherit_lineage_v5,
            [{"run": baseline, "checkpoint_sha256": "legacy.pt"}], "lin-c")
    broken = {key: value for key, value in baseline.items() if key != "lineage_id"}
    _raises("MISSING_FIELD", "$.parents[0].run.lineage_id", inherit_lineage_v5,
            [{"run": broken, "checkpoint_sha256": CHECKPOINT}], "lin-c")


def test_recipe_and_supervision_templates_fail_closed():
    recipe_template = yaml.safe_load((CONFIG_DIR_V5 / "run_recipe.template.yaml").read_text())
    _raises("RECIPE_VALUE_MISSING", "$.recipe_id", validate_recipe_v5, recipe_template)
    scope = _scope("A")
    supervision_template = yaml.safe_load((CONFIG_DIR_V5 / "supervision.template.yaml").read_text())
    _raises("SUPERVISION_VALUE_MISSING", "$.policy_id", validate_supervision_v5, supervision_template, scope)
    recipe = make_recipe()
    assert validate_recipe_v5(recipe) == recipe
    supervision = make_supervision(scope)
    assert validate_supervision_v5(supervision, scope) == supervision
    nested = copy.deepcopy(recipe)
    nested["model"]["tasks"] = None
    _raises("RECIPE_VALUE_MISSING", "$.model.tasks", validate_recipe_v5, nested)
    del nested["budget"]["max_steps"]
    nested["model"]["tasks"] = ["A"]
    _raises("RECIPE_VALUE_MISSING", "$.budget.max_steps", validate_recipe_v5, nested)
    nested = copy.deepcopy(supervision)
    nested["tasks"]["A"]["ce_weight"] = None
    _raises("SUPERVISION_VALUE_MISSING", "$.tasks.A.ce_weight", validate_supervision_v5, nested, scope)


@pytest.mark.parametrize("override, path", [
    ({"recipe_id": " "}, "$.recipe_id"),
    ({"model": {"architecture": "unet2d"}}, "$.model.architecture"),
    ({"model": {"tasks": []}}, "$.model.tasks"),
    ({"model": {"tasks": ["A", "A"]}}, "$.model.tasks"),
    ({"model": {"tasks": ["S"]}}, "$.model.tasks"),
    ({"model": {"widths": [4, 0]}}, "$.model.widths"),
    ({"model": {"strides": [[1, 1, 1]]}}, "$.model.strides"),
    ({"model": {"strides": [[1, 1, 1], [1, 2]]}}, "$.model.strides"),
    ({"model": {"blocks_per_stage": 0}}, "$.model.blocks_per_stage"),
    ({"seeds": {"init": -1}}, "$.seeds.init"),
    ({"seeds": {"sampler": True}}, "$.seeds.sampler"),
    ({"patch_size": [4, 8]}, "$.patch_size"),
    ({"batch_size": 0}, "$.batch_size"),
    ({"precision": "fp16"}, "$.precision"),
    ({"optimizer": {"name": "sgd"}}, "$.optimizer.name"),
    ({"optimizer": {"lr": 0}}, "$.optimizer.lr"),
    ({"optimizer": {"lr": math.nan}}, "$.optimizer.lr"),
    ({"optimizer": {"weight_decay": -0.1}}, "$.optimizer.weight_decay"),
    ({"scheduler": {"name": "cosine"}}, "$.scheduler.name"),
    ({"scheduler": {"exponent": -1}}, "$.scheduler.exponent"),
    ({"downsample_factor": [1, 0, 1]}, "$.downsample_factor"),
    ({"normalization": "minmax"}, "$.normalization"),
    ({"sampling": {"positive_fraction": 1.5}}, "$.sampling.positive_fraction"),
    ({"readable_roles": ["locked_test"]}, "$.readable_roles"),
    ({"readable_roles": ["development_val"]}, "$.readable_roles"),
    ({"readable_roles": ["train", "train"]}, "$.readable_roles"),
    ({"selection_rule": "best_dev"}, "$.selection_rule"),
    ({"inference": {"window": [4, 8]}}, "$.inference.window"),
    ({"inference": {"stride": [8, 8, 8]}}, "$.inference.stride"),
    ({"checkpoint_every": 0}, "$.checkpoint_every"),
    ({"budget": {"max_steps": 0}}, "$.budget.max_steps"),
    ({"budget": {"max_wall_hours": math.inf}}, "$.budget.max_wall_hours"),
    ({"model": ["resunet3d"]}, "$.model"),
])
def test_recipe_values_are_checked(override, path):
    _raises("INVALID_RECIPE_VALUE", path, validate_recipe_v5, make_recipe(**override))


def test_recipe_keys_are_fixed():
    _raises("UNKNOWN_FIELD", "$.model.<unknown>", validate_recipe_v5, make_recipe(model={"dropout": 0.1}))
    _raises("UNKNOWN_FIELD", "$.<unknown>", validate_recipe_v5, make_recipe(augmentation={"flip": True}))
    _raises("INVALID_TYPE", "$", validate_recipe_v5, ["recipe"])


@pytest.mark.parametrize("override, code, path", [
    ({"policy_id": ""}, "INVALID_SUPERVISION_VALUE", "$.policy_id"),
    ({"scope_id": "synthetic_u"}, "SUPERVISION_SCOPE_MISMATCH", "$.scope_id"),
    ({"reference_policy_id": "other_policy"}, "SUPERVISION_SCOPE_MISMATCH", "$.reference_policy_id"),
    ({"channel": "pseudo"}, "INVALID_SUPERVISION_VALUE", "$.channel"),
    ({"track": "any"}, "INVALID_SUPERVISION_VALUE", "$.track"),
    ({"track": "observable_target", "allow_unknown_visibility": True}, "INVALID_SUPERVISION_VALUE",
     "$.allow_unknown_visibility"),
    ({"allow_unknown_visibility": "no"}, "INVALID_SUPERVISION_VALUE", "$.allow_unknown_visibility"),
    ({"tasks": {"A": {"kind": "sigmoid"}}}, "INVALID_SUPERVISION_VALUE", "$.tasks.A.kind"),
    ({"tasks": {"A": {"ce_weight": -1.0}}}, "INVALID_SUPERVISION_VALUE", "$.tasks.A.ce_weight"),
    ({"tasks": {"A": {"ce_weight": 0.0, "dice_weight": 0.0}}}, "INVALID_SUPERVISION_VALUE", "$.tasks.A"),
    ({"tasks": {"A": {"task_weight": 0.0}}}, "INVALID_SUPERVISION_VALUE", "$.tasks.A.task_weight"),
    ({"tasks": {"U": {"kind": "sigmoid", "ce_weight": 1.0, "dice_weight": 1.0, "task_weight": 1.0}}},
     "INVALID_SUPERVISION_VALUE", "$.tasks.U"),
    ({"tasks": {"S": {}}}, "UNKNOWN_FIELD", "$.tasks.<unknown>"),
    ({"tasks": {"A": {"focal_weight": 1.0}}}, "UNKNOWN_FIELD", "$.tasks.A.<unknown>"),
    ({"tasks": []}, "INVALID_SUPERVISION_VALUE", "$.tasks"),
])
def test_supervision_values_are_checked(override, code, path):
    scope = _scope("A")
    _raises(code, path, validate_supervision_v5, make_supervision(scope, **override), scope)


def test_supervision_needs_at_least_one_task():
    scope = _scope("A")
    supervision = make_supervision(scope)
    supervision["tasks"] = {}
    _raises("INVALID_SUPERVISION_VALUE", "$.tasks", validate_supervision_v5, supervision, scope)


def test_registered_run_binds_scope_source_split_and_hashes():
    records = make_split_records(["train", "development_val", "locked_test"])
    registry = make_split_registry(records)
    scope = _scope("A")
    recipe, supervision = make_recipe(), make_supervision(scope)
    run = make_run(scope=scope, records=records, split_registry=registry, recipe=recipe, supervision=supervision)
    assert (run["schema_version"], run["run_kind"], run["fixture_mode"]) == ("v5.1", "baseline", True)
    assert run["scope"] == {"scope_id": "synthetic_a", "scope_hash": scope.scope_hash}
    assert run["source"] == SYNTH_SOURCE
    assert (run["recipe_sha256"], run["supervision_sha256"]) == (canonical_sha256(recipe),
                                                                 canonical_sha256(supervision))
    assert run["split"] == {"split_sha256": registry["split_sha256"], "readable_roles": ["train"]}
    assert run["data_manifest_sha256"] == data_manifest_sha256_v5(records[:1])
    assert run_sha256_v5(run) == canonical_sha256(run)
    for run_id in ("../escape", "a/b", "", ".hidden"):
        _raises("INVALID_RUN_ID", "$.run_id", make_run, run_id=run_id)
    _raises("INVALID_RUN_KIND", "$.run_kind", make_run, run_kind="final")
    _raises("INVALID_TYPE", "$.fixture_mode", make_run, fixture_mode="yes")


def test_registration_requires_passing_gates_per_task():
    for scope, gates in ((_scope("A"), ["G-BASE-A"]), (_scope("U"), ["G-BASE-U"]),
                         (_au_scope(), ["G-BASE-A", "G-BASE-U"])):
        decisions = make_gate_decisions(scope, "baseline")
        run = make_run(scope=scope)
        assert run["gate_decisions"] == [{"gate_id": gate_id, "decision_sha256": canonical_sha256(decisions[gate_id])}
                                         for gate_id in gates]
        for gate_id in gates:
            missing = {key: value for key, value in decisions.items() if key != gate_id}
            _raises("GATE_NOT_PASSED", f"$.gate_decisions.{gate_id}", make_run, scope=scope, gate_decisions=missing)
            blocked = dict(decisions, **{gate_id: evaluate_gate_v5(gate_id, scope, {}, {}, SYNTH_SOURCE,
                                                                   fixture_mode=True)})
            _raises("GATE_NOT_PASSED", f"$.gate_decisions.{gate_id}", make_run, scope=scope, gate_decisions=blocked)
    smoke = make_run(run_kind="smoke")
    assert [decision["gate_id"] for decision in smoke["gate_decisions"]] == ["G-DATA"]
    assert smoke["approvals"]["resources"]["approved_by"] == FIXTURE_APPROVER
    _raises("GATE_NOT_PASSED", "$.gate_decisions.G-DATA", make_run, run_kind="smoke",
            gate_decisions=make_gate_decisions(_scope("A"), "baseline"))
    _raises("APPROVAL_MISSING", "$.approvals.resources", make_run, run_kind="smoke", approvals={})
    _raises("GATE_BINDING_MISMATCH", "$.approvals.resources", make_run, run_kind="smoke",
            approvals=make_approvals(_scope("U"), ["resources"]))
    other_source = {"source_id": "other_source", "source_version": "v0"}
    for decisions in (make_gate_decisions(_scope("A", reference_policy_id="other_policy"), "baseline"),
                      make_gate_decisions(_scope("A"), "baseline", source=other_source)):
        _raises("GATE_BINDING_MISMATCH", "$.gate_decisions.G-BASE-A", make_run, gate_decisions=decisions)
    decisions = make_gate_decisions(_scope("A"), "baseline")
    _raises("GATE_BINDING_MISMATCH", "$.gate_decisions.G-BASE-U", make_run,
            gate_decisions=dict(decisions, **{"G-BASE-U": decisions["G-BASE-A"]}))
    _raises("GATE_FIXTURE_NOT_ALLOWED", "$.gate_decisions.G-BASE-A", make_run, fixture_mode=False)
    _raises("GATE_FIXTURE_NOT_ALLOWED", "$.approvals.resources", make_run, run_kind="smoke", fixture_mode=False,
            gate_decisions=_real(make_gate_decisions(_scope("A"), "smoke")))
    real = make_run(fixture_mode=False, gate_decisions=_real(decisions))
    assert real["fixture_mode"] is False


def test_model_tasks_must_match_scope_and_supervision():
    _raises("MODEL_TASKS_INVALID", "$.recipe.model.tasks", make_run, recipe=make_recipe(model={"tasks": ["A", "U"]}))
    scope = _au_scope()
    _raises("MODEL_TASKS_INVALID", "$.recipe.model.tasks", make_run, scope=scope,
            recipe=make_recipe(model={"tasks": ["A"]}))
    supervision = make_supervision(scope)
    del supervision["tasks"]["U"]
    run = make_run(scope=scope, recipe=make_recipe(model={"tasks": ["A"]}), supervision=supervision,
                   gate_decisions={"G-BASE-A": make_gate_decisions(scope, "baseline")["G-BASE-A"]})
    assert [decision["gate_id"] for decision in run["gate_decisions"]] == ["G-BASE-A"]


def test_smoke_run_reads_train_only():
    records = make_split_records(["train", "development_val", "locked_test", "train"])
    registry = make_split_registry(records)
    smoke = make_run(run_kind="smoke", records=records, split_registry=registry)
    assert smoke["split"] == {"split_sha256": registry["split_sha256"], "readable_roles": ["train"]}
    train = select_samples_v5(records, registry, ["train"])
    assert smoke["data_manifest_sha256"] == data_manifest_sha256_v5(train) == data_manifest_sha256_v5(train[::-1])
    baseline = _train_dev_run(records, registry)
    assert baseline["split"]["readable_roles"] == ["development_val", "train"]
    assert baseline["data_manifest_sha256"] != smoke["data_manifest_sha256"]
    _raises("SMOKE_ROLES_INVALID", "$.recipe.readable_roles", make_run, run_kind="smoke",
            recipe=make_recipe(readable_roles=["train", "development_val"]))


def test_dirty_code_rejected_for_real_runs():
    real = _real(make_gate_decisions(_scope("A"), "baseline"))
    dirty = {"git_commit": "0" * 40, "dirty": True}
    _raises("CODE_DIRTY", "$.code.dirty", make_run, fixture_mode=False, gate_decisions=real, code=dirty)
    assert make_run(fixture_mode=False, gate_decisions=real)["code"]["dirty"] is False
    assert make_run(code=dirty)["code"]["dirty"] is True


def test_run_exposure_entries_cover_read_roles_only():
    records = make_split_records(["train", "development_val", "locked_test", "unassigned"])
    registry = make_split_registry(records)
    run = _train_dev_run(records, registry)
    entry = {"experiment_id": run["run_id"], "recorded_at": run["created_at"], "lineage_id": "lin-a",
             "evidence": f"run_sha256:{run_sha256_v5(run)}"}
    assert run_exposure_entries_v5(run, registry) == {"s-0": [dict(entry, role="training")],
                                                      "s-1": [dict(entry, role="model_selection")]}
    smoke = make_run(run_kind="smoke", records=records, split_registry=registry)
    assert sorted(run_exposure_entries_v5(smoke, registry)) == ["s-0"]
    _raises("SPLIT_HASH_MISMATCH", "$.split.split_sha256", run_exposure_entries_v5, run,
            make_split_registry(records, official_split={}))


def test_merged_exposure_validates_and_rejects_unknown_samples():
    records = make_split_records(["train", "development_val", "locked_test"])
    registry = make_split_registry(records)
    run = _train_dev_run(records, registry)
    exposure = run_exposure_file_v5(run, registry)
    assert (exposure["run_id"], exposure["run_sha256"]) == (run["run_id"], run_sha256_v5(run))
    merged = merge_run_exposure_v5(records, [exposure])
    assert [[entry["role"] for entry in record.exposure_history] for record in merged] == [
        ["training"], ["model_selection"], []]
    assert all(record.exposure_history == [] for record in records)
    all_test = {f"synthetic:p{i:04d}": "test" for i in range(3)}
    refrozen = make_split_registry(merged, official_split=all_test)
    assert refrozen["assignment"] == {"s-0": "excluded", "s-1": "excluded", "s-2": "locked_test"}
    _raises("DUPLICATE_EXPOSURE_FILE", "$[1].run_id", merge_run_exposure_v5, records, [exposure, exposure])
    _raises("EXPOSURE_SAMPLE_UNKNOWN", "$[0].entries", merge_run_exposure_v5, records[1:], [exposure])
    bad = copy.deepcopy(exposure)
    bad["entries"]["s-0"][0]["role"] = "train"
    _raises("INVALID_ENUM", "$.exposure_history[0].role", merge_run_exposure_v5, records, [bad])


def test_run_record_written_privately_without_overwrite(tmp_path):
    records = make_split_records(["train", "development_val", "locked_test"])
    registry = make_split_registry(records)
    run = _train_dev_run(records, registry)
    root = tmp_path / "private"
    run_dir = write_run_v5(run, registry, root, REPO)
    assert run_dir == (root / "runs" / run["run_id"]).resolve()
    assert json.loads((run_dir / "run.json").read_text()) == run
    assert json.loads((run_dir / "exposure.json").read_text()) == run_exposure_file_v5(run, registry)
    for path in (run_dir / "run.json", run_dir / "exposure.json"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for directory in (root, root / "runs", run_dir):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    with pytest.raises(PrivatePathError) as err:
        write_run_v5(run, registry, root, REPO)
    assert err.value.code == "PRIVATE_FILE_EXISTS"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(PrivatePathError) as err:
        write_run_v5(run, registry, repo / "private", repo)
    assert err.value.code == "PRIVATE_PATH_IN_GIT"


def test_environment_and_code_capture(tmp_path):
    environment = capture_environment_v5()
    assert set(environment) == {"python", "torch", "numpy", "cuda", "devices"}
    assert isinstance(environment["devices"], list)
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.invalid",
           "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    subprocess.run(git + ["init", "-q"], check=True)
    (repo / "a.txt").write_text("a\n")
    subprocess.run(git + ["add", "a.txt"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "synthetic"], check=True)
    head = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    assert capture_code_v5(repo) == {"git_commit": head, "dirty": False}
    (repo / "a.txt").write_text("b\n")
    assert capture_code_v5(repo) == {"git_commit": head, "dirty": True}
    _raises("CODE_STATE_UNKNOWN", "$.code", capture_code_v5, tmp_path / "missing")
