"""Evaluate-only entry and sampling replay of a finished unified run (diagnostics of 2026-11-11): the checkpoint
stays fixed, only train and development_val are read, everything goes to a new directory. Synthetic volumes, CPU."""
from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from anatobind.v41 import real_unified_eval_v5 as eval_module
from anatobind.v41 import real_unified_v5 as real_unified_module
from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.real_unified_eval_v5 import (
    render_unified_evaluation_md_v5, render_unified_replay_md_v5, run_unified_evaluation_v5, run_unified_replay_v5)
from anatobind.v41.real_unified_v5 import run_unified_training_v5
from tests.v5.test_real_unified import env  # noqa: F401  (the synthetic two-member world)


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _tree(root) -> dict:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.fixture
def finished(env, tmp_path):  # noqa: F811
    """A completed synthetic training run (2 train scans, 1 development scan per member) and evaluation inputs."""
    summary = run_unified_training_v5(**env.train_kwargs)
    assert summary["status"] == "COMPLETED"
    root = env.train_kwargs["private_root"]
    common = {key: value for key, value in env.train_kwargs.items() if key != "private_root"}
    common.update(run_root=root, created_at="2026-11-12T00:00:00Z", max_wall_hours=0.5)
    return SimpleNamespace(
        summary=summary, root=root, before=_tree(root),
        eval_kwargs=dict(common, private_root=tmp_path / "evaluation", roles=["train", "development_val"]),
        replay_kwargs=dict(common, private_root=tmp_path / "replay", loss_steps=2))


def test_the_development_split_is_reproduced_from_the_outputs_the_run_saved(finished):
    result = run_unified_evaluation_v5(**finished.eval_kwargs)
    assert result["status"] == "COMPLETED" and result["stop_code"] is None, result
    assert set(result["members"]) == {"knee"}                          # the box-supervised member
    development, original = result["members"]["knee"]["development_val"], finished.summary["members"]["knee"]
    assert development["findings"] == original["findings"]
    assert development["anatomy_dice"] == original["anatomy_dice"] and development["sentences"] == original["sentences"]
    assert development["reuse"] == {"scans": 1, "reused": 1, "identical": 1}
    assert development["reproduces_original"] == {"findings": True, "anatomy_dice": True, "sentences": True}


def test_the_training_split_is_scored_against_all_of_its_own_primary_boxes(finished):
    result = run_unified_evaluation_v5(**finished.eval_kwargs)
    train = result["members"]["knee"]["train"]
    overall = train["findings"]["overall"]
    assert (overall["n_scans"], overall["n_patients"], overall["n_boxes"]) == (2, 2, 1)     # knee-0 has the one box
    assert train["diagnostics"]["funnel"]["overall"]["n_boxes"] == 1 and train["diagnostics"]["n_scans"] == 2
    assert sum(train["diagnostics"]["funnel"]["overall"]["stages"].values()) == 1            # a missed box is counted
    assert train["reuse"] == {"scans": 2, "reused": 0, "identical": 0} and train["reproduces_original"] is None
    assert set(train["anatomy_dice"]) == {"tissue_a", "tissue_b"}
    locations = train["diagnostics"]["locations"]["overall"]
    assert locations["total"] == overall["findings"]["total"]
    assert locations["same_class_primary"] == overall["findings"]["in_reference"]
    assert locations["other_class_primary"] + locations["outside_all"] == overall["findings"]["outside"]


def test_evaluation_changes_neither_the_weights_nor_the_finished_run(finished):
    result = run_unified_evaluation_v5(**finished.eval_kwargs)
    assert _tree(finished.root) == finished.before                     # the finished run is byte for byte as before
    checkpoint = result["checkpoint"]
    assert checkpoint["sha256"] == finished.summary["training"]["checkpoint_sha256"] and checkpoint["step"] == 4
    assert checkpoint["weights_unchanged"] is True and len(checkpoint["state_sha256"]) == 64
    out = finished.eval_kwargs["private_root"]
    assert json.loads((out / "summary.json").read_text())["status"] == "COMPLETED"
    rows = [json.loads(line) for line in (out / "rows" / "train.knee.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and set(rows[0]) == {"official", "diagnostic", "reused", "identical"}
    for path in out.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert result["run"]["run_sha256"] == finished.summary["run"]["run_sha256"]
    assert result["evaluation"]["roles"] == ["train", "development_val"]


def test_a_changed_weight_is_detected(finished, monkeypatch):
    original = eval_module.run_perception_v5

    def nudging(model, *args, **kwargs):
        next(iter(model.parameters())).data.add_(1.0)                  # anything that touches the weights
        return original(model, *args, **kwargs)

    monkeypatch.setattr(eval_module, "run_perception_v5", nudging)
    _raises("EVAL_WEIGHTS_CHANGED", "$.checkpoint", run_unified_evaluation_v5, **finished.eval_kwargs)
    assert not (finished.eval_kwargs["private_root"] / "summary.json").exists()


def test_only_train_and_development_val_are_read(finished, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("data was read for a role that may not be evaluated")

    with monkeypatch.context() as patch:
        patch.setattr(real_unified_module, "load_member_samples_v5", _fail_if_called)
        for roles, code, path in ((["locked_test"], "LOCKED_TEST_NOT_READABLE", "$.roles[0]"),
                                  (["train", "locked_test"], "LOCKED_TEST_NOT_READABLE", "$.roles[1]"),
                                  (["excluded"], "INVALID_ROLE", "$.roles[0]"),
                                  (["train", "train"], "INVALID_ROLE", "$.roles[1]"), ([], "INVALID_ROLE", "$.roles")):
            _raises(code, path, run_unified_evaluation_v5, **dict(finished.eval_kwargs, roles=roles))
    assert not finished.eval_kwargs["private_root"].exists()
    seen = []
    original = real_unified_module.load_box_references_v5

    def recording(path, registry, roles):
        seen.append(list(roles))
        return original(path, registry, roles)

    monkeypatch.setattr(real_unified_module, "load_box_references_v5", recording)
    run_unified_evaluation_v5(**finished.eval_kwargs)
    assert seen == [["train"], ["development_val"]]                    # each split holds its own boxes only


def test_evaluation_refuses_another_run_an_unfinished_run_and_a_bad_destination(finished, env, tmp_path, monkeypatch):  # noqa: F811
    kwargs = finished.eval_kwargs
    approvals = copy.deepcopy(kwargs["approvals"])
    approvals["brain"]["approved_at"] = "2026-11-11"                 # a valid approval, but not the registered one
    _raises("EVAL_RUN_MISMATCH", "$.run", run_unified_evaluation_v5, **dict(kwargs, approvals=approvals))
    _raises("EVAL_RUN_MISSING", "$.run", run_unified_evaluation_v5, **dict(kwargs, run_root=tmp_path / "nowhere"))
    _raises("EVAL_OUT_INSIDE_RUN", "$.out", run_unified_evaluation_v5,
            **dict(kwargs, private_root=finished.root / "evaluation"))
    _raises("EVAL_BUDGET_INVALID", "$.max_wall_hours", run_unified_evaluation_v5, **dict(kwargs, max_wall_hours=2.5))
    _raises("EVAL_BUDGET_INVALID", "$.max_wall_hours", run_unified_evaluation_v5, **dict(kwargs, max_wall_hours=0))
    _raises("CODE_DIRTY", "$.code.dirty", run_unified_evaluation_v5,
            **dict(kwargs, code={"git_commit": "0" * 40, "dirty": True}, fixture_mode=False))
    assert not kwargs["private_root"].exists()

    def killed(*args, **kwargs):
        raise RuntimeError("killed")

    unfinished = dict(env.train_kwargs, private_root=tmp_path / "unfinished")
    with monkeypatch.context() as patch:
        patch.setattr(real_unified_module, "train_unified_v5", killed)
        with pytest.raises(RuntimeError):
            run_unified_training_v5(**unfinished)
    _raises("EVAL_RUN_NOT_COMPLETED", "$.run", run_unified_evaluation_v5,
            **dict(kwargs, run_root=tmp_path / "unfinished"))
    run_unified_evaluation_v5(**kwargs)
    _raises("EVAL_OUT_EXISTS", "$.out", run_unified_evaluation_v5, **kwargs)        # never written over


def test_the_evaluation_report_carries_counts_and_no_identifier(finished):
    text = render_unified_evaluation_md_v5(run_unified_evaluation_v5(**finished.eval_kwargs))
    assert "knee-0" not in text and "synthetic:" not in text and str(finished.root) not in text
    for phrase in ("NOT_EVALUATED", "train", "development_val", "权重未被修改", "locked_test 未读",
                   "不能代替泛化评价", finished.summary["training"]["checkpoint_sha256"]):
        assert phrase in text


def test_the_sampling_replay_matches_the_log_and_recovers_the_loss_parts_at_checkpoints(finished):
    result = run_unified_replay_v5(**finished.replay_kwargs)
    assert result["status"] == "COMPLETED", result
    knee = result["members"]["knee"]
    assert knee["verification"] == {"steps": 4, "support_compared": 8, "support_mismatches": 0,
                                    "sampler_state_equal": True}
    # the weights are known at the seeded initialisation and at the step-2 checkpoint: steps 1 and 3 are recovered
    assert [(item["step"], item["weights_step"]) for item in knee["history"]] == [(1, 0), (3, 2)]
    log = [json.loads(line) for line in
           (finished.root / "runs" / "train-u" / "metrics.jsonl").read_text().splitlines()]
    for item in knee["history"]:
        logged = log[item["step"] - 1]["members"]["knee"]["tasks"]["U"]["loss"]
        assert item["logged_loss"] == logged and item["recomputed_loss"] == pytest.approx(logged, abs=1e-5)
        assert item["recomputed_loss"] == pytest.approx(
            sum(part for part in (item["positive"], item["negative"]) if part is not None), abs=1e-6)
    exposure = knee["exposure"]
    assert exposure["steps"] == 4 and exposure["bags_in_split"] == 1          # knee-0 holds the one training box
    assert 0 <= exposure["steps_with_a_bag"] <= 4
    assert set(knee["current"]) == {"train", "development_val"} and knee["current"]["train"]["batches"] == 2
    assert result["checkpoint"]["weights_unchanged"] is True
    assert _tree(finished.root) == finished.before
    text = render_unified_replay_md_v5(result)
    assert "knee-0" not in text and "逐步核对" in text and "没有记录" in text and "NOT_EVALUATED" in text
    assert "各检查点" in text


def test_every_stored_checkpoint_is_probed_on_the_same_batches_of_both_splits(finished):
    result = run_unified_replay_v5(**dict(finished.replay_kwargs, loss_steps=1))
    knee = result["members"]["knee"]
    trajectory = knee["trajectory"]
    assert [item["weights_step"] for item in trajectory] == [0, 2, 4]          # initialisation, checkpoint, final
    for item in trajectory:
        assert item["train"]["batches"] == item["development_val"]["batches"] == 1
    assert {role: trajectory[-1][role] for role in ("train", "development_val")} == knee["current"]
    # the probe's first training batch is the batch of step 1: under the initial weights it is the recovered history
    assert trajectory[0]["train"]["negative"]["mean"] == pytest.approx(knee["history"][0]["negative"], abs=1e-6)
    # the same batches under other weights give other values
    assert trajectory[0]["train"]["negative"]["mean"] != trajectory[-1]["train"]["negative"]["mean"]
    assert trajectory[0]["development_val"]["negative"]["mean"] != trajectory[-1]["development_val"]["negative"]["mean"]
    assert result["checkpoint"]["weights_unchanged"] is True and _tree(finished.root) == finished.before


def _command_files(tmp_path) -> list:
    """The files the evaluate / replay commands read, for the repository's own unified plan (no sample in them)."""
    import yaml

    from anatobind.v41.engine.unified_registration_v5 import unified_plan_sha256_v5
    from anatobind.v41.findings_v5 import load_protocols_v5
    from tests.v5.factories import make_split_registry

    config = Path(__file__).resolve().parents[2] / "configs" / "v5"
    plan = yaml.safe_load((config / "proposals" / "unified_knee_brain.yaml").read_text(encoding="utf-8"))
    table = load_protocols_v5(config / "protocols_v5.yaml", config)
    digest, approvals, argv = unified_plan_sha256_v5(plan), {}, []
    for pid in sorted(plan["members"]):
        scope, tasks = table[pid]["scope"], sorted(plan["members"][pid]["tasks"])
        resources = {"approved_by": "researcher", "approved_at": "2026-11-12", "scope_hash": scope.scope_hash,
                     "unified_plan_sha256": digest}
        approvals[scope.scope_id] = {f"G-BASE-{task}": {"resources": resources} for task in tasks}
        folder = tmp_path / pid
        folder.mkdir()
        (folder / "records.jsonl").write_text("")
        (folder / "registry.json").write_text(json.dumps(make_split_registry([], lineage_id=f"lin-{pid}")))
        (folder / "decision.json").write_text("{}")
        paths = {"records": folder / "records.jsonl", "split-registry": folder / "registry.json",
                 **{f"gbase-{task.lower()}": folder / "decision.json" for task in tasks}}
        argv += ["--member", f"{pid}:" + ",".join(f"{key}={path}" for key, path in paths.items())]
    (tmp_path / "approvals.yaml").write_text(yaml.safe_dump(approvals))
    return ["--plan", str(config / "proposals" / "unified_knee_brain.yaml"), "--protocols",
            str(config / "protocols_v5.yaml"), "--policies", str(config / "sources_v5.yaml"), "--raw",
            str(tmp_path / "raw"), "--approvals", str(tmp_path / "approvals.yaml"), "--run-id", "run-u", "--device",
            "cpu", "--run-root", str(tmp_path / "finished"), *argv]


def test_the_evaluate_and_replay_commands_hand_their_arguments_to_the_driver_and_keep_the_report(
        tmp_path, monkeypatch, capsys):
    calls = []

    def fake(status):
        def run(**kwargs):
            assert not kwargs["private_root"].exists()                 # the driver decides about the destination
            kwargs["private_root"].mkdir(mode=0o700)
            calls.append(kwargs)
            return {"status": status}
        return run

    shared = _command_files(tmp_path)
    monkeypatch.setattr(eval_module, "run_unified_evaluation_v5", fake("COMPLETED"))
    monkeypatch.setattr(eval_module, "render_unified_evaluation_md_v5", lambda summary: "evaluation report\n")
    out = tmp_path / "evaluation"
    assert real_unified_module.main(["evaluate", *shared, "--out", str(out), "--role", "train", "--role",
                                     "development_val", "--max-wall-hours", "1.5"]) == 0
    given = calls.pop()
    assert given["roles"] == ["train", "development_val"] and given["max_wall_hours"] == 1.5
    assert (given["run_root"], given["private_root"], given["run_id"]) == (tmp_path / "finished", out, "run-u")
    assert given["fixture_mode"] is False and given["device"] == "cpu" and set(given["members"]) == set(given["plan"]["members"])
    report = out / "unified_evaluation.md"
    assert report.read_text() == "evaluation report\n" and stat.S_IMODE(report.stat().st_mode) == 0o600
    assert capsys.readouterr().out == "evaluation report\n"

    monkeypatch.setattr(eval_module, "run_unified_replay_v5", fake("FAILED"))
    monkeypatch.setattr(eval_module, "render_unified_replay_md_v5", lambda summary: "replay report\n")
    out = tmp_path / "replay"
    assert real_unified_module.main(["replay", *shared, "--out", str(out), "--max-wall-hours", "2", "--loss-steps",
                                     "200"]) == 1                      # a run that did not complete exits non-zero
    given = calls.pop()
    assert given["loss_steps"] == 200 and given["max_wall_hours"] == 2.0 and "roles" not in given
    assert (out / "unified_replay.md").read_text() == "replay report\n"
