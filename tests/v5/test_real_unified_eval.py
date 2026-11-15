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
from tests.v5.test_real_unified import _world, env, pilot  # noqa: F401  (the synthetic two-member worlds)


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _tree(root) -> dict:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _finished(world, out):
    summary = run_unified_training_v5(**world.train_kwargs)
    assert summary["status"] == "COMPLETED"
    root = world.train_kwargs["private_root"]
    common = {key: value for key, value in world.train_kwargs.items() if key != "private_root"}
    common.update(run_root=root, created_at="2026-11-12T00:00:00Z", max_wall_hours=0.5)
    return SimpleNamespace(
        summary=summary, root=root, before=_tree(root),
        eval_kwargs=dict(common, private_root=out / "evaluation", roles=["train", "development_val"]),
        replay_kwargs=dict(common, private_root=out / "replay", loss_steps=2),
        probe_kwargs=dict(common, private_root=out / "probe", loss_steps=2))


@pytest.fixture
def finished(env, tmp_path):  # noqa: F811
    """A completed synthetic training run (2 train scans, 1 development scan per member) and evaluation inputs."""
    return _finished(env, tmp_path)


@pytest.fixture
def finished_pilot(pilot, tmp_path):  # noqa: F811
    """The same for a pilot plan: top quarter of the negatives, stopped after step 4 of an 8-step schedule."""
    return _finished(pilot, tmp_path)


def test_the_development_split_is_reproduced_from_the_outputs_the_run_saved(finished):
    result = run_unified_evaluation_v5(**finished.eval_kwargs)
    assert result["status"] == "COMPLETED" and result["stop_code"] is None, result
    assert set(result["members"]) == {"knee"}                          # the box-supervised member
    development, original = result["members"]["knee"]["development_val"], finished.summary["members"]["knee"]
    assert development["findings"] == original["findings"]
    assert development["anatomy_dice"] == original["anatomy_dice"] and development["sentences"] == original["sentences"]
    assert development["reuse"] == {"scans": 1, "reused": 1, "identical": 1, "other_checkpoint": 0}
    assert development["reproduces_original"] == {"findings": True, "anatomy_dice": True, "sentences": True}


def test_the_training_split_is_scored_against_all_of_its_own_primary_boxes(finished):
    result = run_unified_evaluation_v5(**finished.eval_kwargs)
    train = result["members"]["knee"]["train"]
    overall = train["findings"]["overall"]
    assert (overall["n_scans"], overall["n_patients"], overall["n_boxes"]) == (2, 2, 1)     # knee-0 has the one box
    assert train["diagnostics"]["funnel"]["overall"]["n_boxes"] == 1 and train["diagnostics"]["n_scans"] == 2
    assert sum(train["diagnostics"]["funnel"]["overall"]["stages"].values()) == 1            # a missed box is counted
    assert train["reuse"] == {"scans": 2, "reused": 0, "identical": 0, "other_checkpoint": 0}
    assert train["reproduces_original"] is None
    assert set(train["anatomy_dice"]) == {"tissue_a", "tissue_b"}
    locations = train["diagnostics"]["locations"]["overall"]
    assert locations["total"] == overall["findings"]["total"]
    assert locations["same_class_primary"] == overall["findings"]["in_reference"]
    assert locations["other_class_primary"] + locations["outside_all"] == overall["findings"]["outside"]


def test_every_split_carries_the_chance_control_of_its_hit_rate(finished):
    """Proposal of 2026-11-13: the findings of a scan meet the boxes of another patient's scan, rules unchanged."""
    result = run_unified_evaluation_v5(**finished.eval_kwargs)
    train = result["members"]["knee"]["train"]
    control, overall = train["chance_control"]["cross_patient"], train["findings"]["overall"]
    assert (control["n_scans"], control["n_patients"], control["n_cross_patient_pairs"]) == (2, 2, 2)
    assert control["overall"]["n_boxes"] == overall["n_boxes"]
    assert control["overall"]["hit"]["observed_rate"] == overall["box_sensitivity"]["pooled"]
    assert control["overall"]["host_correct"]["observed_rate"] == overall["host_accuracy"]["pooled"]
    assert control["overall"]["hit"]["chance_rate"] is not None
    moved = train["chance_control"]["relocation"]["overall"]       # the control to read: within the scan
    assert moved["n_boxes"] == overall["n_boxes"] == moved["n_usable"] + moved["excluded"]
    if moved["n_usable"] == moved["n_boxes"]:
        assert moved["hit"]["observed_rate"] == overall["box_sensitivity"]["pooled"]
        assert 0.0 <= moved["hit"]["chance_rate"] <= 1.0 and 0.0 < moved["hit"]["p_at_least_observed"] <= 1.0
    assert result["members"]["knee"]["development_val"]["chance_control"]["cross_patient"]["n_scans"] == 1
    rows = [json.loads(line) for line in (Path(finished.eval_kwargs["private_root"]) / "rows" / "train.knee.jsonl")
            .read_text().splitlines()]
    assert sum(len(row["relocation"]) for row in rows) == overall["n_boxes"]
    text = render_unified_evaluation_md_v5(result)
    assert "偶然命中水平的对照" in text and "knee-0" not in text and "synthetic:" not in text


def test_evaluation_changes_neither_the_weights_nor_the_finished_run(finished):
    result = run_unified_evaluation_v5(**finished.eval_kwargs)
    assert _tree(finished.root) == finished.before                     # the finished run is byte for byte as before
    checkpoint = result["checkpoint"]
    assert checkpoint["sha256"] == finished.summary["training"]["checkpoint_sha256"] and checkpoint["step"] == 4
    assert checkpoint["weights_unchanged"] is True and len(checkpoint["state_sha256"]) == 64
    out = finished.eval_kwargs["private_root"]
    assert json.loads((out / "summary.json").read_text())["status"] == "COMPLETED"
    rows = [json.loads(line) for line in (out / "rows" / "train.knee.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and set(rows[0]) == {"official", "diagnostic", "relocation", "reused", "identical"}
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
    check = knee["verification"]
    assert (check["steps"], check["support_compared"], check["support_mismatches"], check["sampler_state_equal"]) == (
        4, 8, 0, True)
    assert check["terms_compared"] > 0 and check["terms_mismatches"] == 0     # counts of the logged terms, every step
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
    assert "knee-0" not in text and "逐步核对" in text and "逐步记录" in text and "NOT_EVALUATED" in text
    assert "没有记录" not in text
    for item in knee["history"]:                               # the logged terms are the recomputed ones
        assert item["logged_terms"] is not None and item["logged_terms_abs_diff"] <= 1e-5
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
    assert (given["checkpoint"], given["reuse_inference"], given["module_development"]) == (None, "inference", False)
    assert (given["run_root"], given["private_root"], given["run_id"]) == (tmp_path / "finished", out, "run-u")
    assert given["fixture_mode"] is False and given["device"] == "cpu" and set(given["members"]) == set(given["plan"]["members"])
    report = out / "unified_evaluation.md"
    assert report.read_text() == "evaluation report\n" and stat.S_IMODE(report.stat().st_mode) == 0o600
    assert capsys.readouterr().out == "evaluation report\n"

    digest = "ab" * 32
    out = tmp_path / "evaluation_step"
    assert real_unified_module.main(["evaluate", *shared, "--out", str(out), "--role", "development_val",
                                     "--max-wall-hours", "1", "--checkpoint-step", "6000", "--checkpoint-sha256",
                                     digest, "--fresh-predictions", "--module-development"]) == 0
    given = calls.pop()
    assert (given["checkpoint"], given["reuse_inference"], given["module_development"]) == ((6000, digest), None, True)
    assert given["instance_mask_limit_gib"] is None
    capsys.readouterr()
    assert real_unified_module.main(["evaluate", *shared, "--out", str(tmp_path / "evaluation_budget"), "--role",
                                     "train", "--max-wall-hours", "1", "--instance-mask-limit-gib", "256"]) == 0
    assert calls.pop()["instance_mask_limit_gib"] == 256.0
    capsys.readouterr()
    with pytest.raises(V5Error) as err:                                # a step without its hash selects nothing
        real_unified_module.main(["evaluate", *shared, "--out", str(tmp_path / "half"), "--role", "train",
                                  "--max-wall-hours", "1", "--checkpoint-step", "6000"])
    assert (err.value.code, err.value.path) == ("EVAL_CHECKPOINT_INVALID", "$.checkpoint")
    assert not calls and not (tmp_path / "half").exists()

    monkeypatch.setattr(eval_module, "run_unified_probe_v5", fake("COMPLETED"))
    monkeypatch.setattr(eval_module, "render_unified_probe_md_v5", lambda summary: "probe report\n")
    out = tmp_path / "probe"
    assert real_unified_module.main(["probe", *shared, "--out", str(out), "--max-wall-hours", "0.5", "--loss-steps",
                                     "200", "--checkpoint", f"2000:{digest}", "--checkpoint", f"4000:{digest}",
                                     "--negative-topk-fraction", "0.01"]) == 0
    given = calls.pop()
    assert given["checkpoints"] == [(2000, digest), (4000, digest)] and given["negative_topk_fraction"] == 0.01
    assert (given["loss_steps"], given["max_wall_hours"], given["run_root"]) == (200, 0.5, tmp_path / "finished")
    assert (out / "unified_probe.md").read_text() == "probe report\n"
    capsys.readouterr()

    monkeypatch.setattr(eval_module, "run_unified_replay_v5", fake("FAILED"))
    monkeypatch.setattr(eval_module, "render_unified_replay_md_v5", lambda summary: "replay report\n")
    out = tmp_path / "replay"
    assert real_unified_module.main(["replay", *shared, "--out", str(out), "--max-wall-hours", "2", "--loss-steps",
                                     "200"]) == 1                      # a run that did not complete exits non-zero
    given = calls.pop()
    assert given["loss_steps"] == 200 and given["max_wall_hours"] == 2.0 and "roles" not in given
    assert (out / "unified_replay.md").read_text() == "replay report\n"


# ---------------------------------------------------------------------------
# Pilot of 2026-11-12: a checkpoint selected by step and hash, saved outputs reused only for the same hash, and the
# voxel-supervised member scored with the training driver's own module metrics


def _sidecar(root, step: int) -> dict:
    return json.loads((root / "runs" / "train-u" / "checkpoints" / f"step_{step:08d}.pt.json").read_text())


def test_a_checkpoint_selected_by_step_and_hash_is_the_one_evaluated(finished):
    earlier, final = _sidecar(finished.root, 2)["sha256"], finished.summary["training"]["checkpoint_sha256"]
    assert earlier != final
    result = run_unified_evaluation_v5(**finished.eval_kwargs, checkpoint=(2, earlier))
    assert result["status"] == "COMPLETED", result
    checkpoint = result["checkpoint"]
    assert (checkpoint["step"], checkpoint["sha256"], checkpoint["selection"], checkpoint["is_final"]) == (
        2, earlier, "explicit", False)
    assert checkpoint["run_final"] == {"step": 4, "sha256": final} and checkpoint["weights_unchanged"] is True
    assert _tree(finished.root) == finished.before                     # the run and its summary are as they were
    development = result["members"]["knee"]["development_val"]
    # the outputs the run saved belong to its final checkpoint: none of them enters this evaluation
    assert development["reuse"] == {"scans": 1, "reused": 0, "identical": 0, "other_checkpoint": 0}
    assert development["reproduces_original"] is None
    assert (result["evaluation"]["reuse_inference"], result["evaluation"]["reuse_refused"]) == (
        None, "NOT_THE_FINAL_CHECKPOINT")
    other = dict(finished.eval_kwargs, private_root=finished.eval_kwargs["private_root"].with_name("final"))
    final_result = run_unified_evaluation_v5(**other)
    assert final_result["checkpoint"]["state_sha256"] != checkpoint["state_sha256"]          # other weights
    assert (final_result["checkpoint"]["selection"], final_result["checkpoint"]["is_final"]) == ("final", True)
    assert final_result["evaluation"]["reuse_refused"] is None
    text = render_unified_evaluation_md_v5(result)
    assert earlier in text and "按步数与哈希指定" in text and final in text and "第 2 步" in text


def test_a_selection_that_does_not_name_a_checkpoint_of_the_run_is_refused_before_data_is_read(
        finished, monkeypatch, tmp_path):
    import shutil

    earlier, final = _sidecar(finished.root, 2)["sha256"], finished.summary["training"]["checkpoint_sha256"]
    kwargs = finished.eval_kwargs

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("data was read before the checkpoint was checked")

    with monkeypatch.context() as patch:
        patch.setattr(real_unified_module, "load_member_samples_v5", _fail_if_called)
        _raises("EVAL_CHECKPOINT_MISMATCH", "$.checkpoint", run_unified_evaluation_v5, **kwargs, checkpoint=(2, final))
        _raises("EVAL_CHECKPOINT_MISMATCH", "$.checkpoint", run_unified_evaluation_v5, **kwargs, checkpoint=(4, earlier))
        _raises("EVAL_CHECKPOINT_MISSING", "$.checkpoint", run_unified_evaluation_v5, **kwargs, checkpoint=(3, earlier))
        for bad in ((2, "xyz"), (0, earlier), (True, earlier), (2.0, earlier), [2, earlier], (2,)):
            _raises("EVAL_CHECKPOINT_INVALID", "$.checkpoint", run_unified_evaluation_v5, **kwargs, checkpoint=bad)
        copy_root = tmp_path / "copy"
        shutil.copytree(finished.root, copy_root)
        with open(copy_root / "runs" / "train-u" / "checkpoints" / "step_00000002.pt", "ab") as handle:
            handle.write(b"0")                                         # the file is no longer the hashed one
        _raises("CHECKPOINT_HASH_MISMATCH", "$.checkpoint", run_unified_evaluation_v5,
                **dict(kwargs, run_root=copy_root), checkpoint=(2, earlier))
    assert not kwargs["private_root"].exists() and _tree(finished.root) == finished.before


def test_fresh_predictions_of_the_final_checkpoint_reproduce_the_run_without_its_saved_outputs(finished):
    final = finished.summary["training"]["checkpoint_sha256"]
    result = run_unified_evaluation_v5(**dict(finished.eval_kwargs, reuse_inference=None), checkpoint=(4, final))
    assert (result["checkpoint"]["selection"], result["checkpoint"]["is_final"]) == ("explicit", True)
    development = result["members"]["knee"]["development_val"]
    assert development["reuse"] == {"scans": 1, "reused": 0, "identical": 0, "other_checkpoint": 0}
    assert development["reproduces_original"] == {"findings": True, "anatomy_dice": True, "sentences": True}
    assert development["findings"] == finished.summary["members"]["knee"]["findings"]
    assert result["evaluation"]["reuse_inference"] is None and result["evaluation"]["reuse_refused"] is None


def test_a_saved_output_of_another_checkpoint_is_never_reused(finished, tmp_path):
    import shutil

    copy_root = tmp_path / "copy"
    shutil.copytree(finished.root, copy_root)
    records = sorted((copy_root / "inference" / "knee").glob("*.record.json"))
    assert len(records) == 1
    body = json.loads(records[0].read_text())
    other = "c" * 64
    body["checkpoint_sha256"], body["provenance"]["model_sha256"] = other, other     # as if other weights wrote it
    records[0].chmod(0o600)
    records[0].write_text(json.dumps(body))
    result = run_unified_evaluation_v5(**dict(finished.eval_kwargs, run_root=copy_root))
    development = result["members"]["knee"]["development_val"]
    assert development["reuse"] == {"scans": 1, "reused": 0, "identical": 0, "other_checkpoint": 1}
    assert development["findings"] == finished.summary["members"]["knee"]["findings"]       # from the new inference


def test_the_voxel_supervised_member_is_scored_on_development_val_with_the_module_metrics(finished, monkeypatch):
    seen = []
    original = real_unified_module.load_member_samples_v5

    def recording(**kwargs):
        seen.append((kwargs["protocol_id"], list(kwargs["roles"])))
        return original(**kwargs)

    monkeypatch.setattr(real_unified_module, "load_member_samples_v5", recording)
    result = run_unified_evaluation_v5(**dict(finished.eval_kwargs, reuse_inference=None), module_development=True)
    assert result["status"] == "COMPLETED", result
    assert set(result["members"]) == {"brain", "knee"} and set(result["members"]["brain"]) == {"development_val"}
    assert ("brain", ["development_val"]) in seen and ("brain", ["train"]) not in seen
    brain = result["members"]["brain"]["development_val"]
    # the final checkpoint, new predictions: the module summary of the training driver, value for value
    assert brain["module"] == finished.summary["members"]["brain"]["module"] and brain["n_scans"] == 1
    assert brain["sentences"] == finished.summary["members"]["brain"]["sentences"]
    assert brain["reproduces_original"] == {"module": True, "sentences": True}
    assert _tree(finished.root) == finished.before and result["checkpoint"]["weights_unchanged"] is True
    assert result["evaluation"]["module_development"] is True
    text = render_unified_evaluation_md_v5(result)
    assert "## brain：development_val 的体素级评价" in text and "dice" in text and "brain-2" not in text
    earlier = _sidecar(finished.root, 2)["sha256"]
    other = dict(finished.eval_kwargs, private_root=finished.eval_kwargs["private_root"].with_name("earlier"))
    result = run_unified_evaluation_v5(**other, checkpoint=(2, earlier), module_development=True)
    brain = result["members"]["brain"]["development_val"]
    assert brain["module"]["checkpoint_sha256"] == earlier and brain["reproduces_original"] is None


# ---------------------------------------------------------------------------
# Probe: the terms of the box objective and the reach of the hard-negative selection under given checkpoints, on
# the first training batches of the run. Replay of a stopped run. Comparability of two runs.

from anatobind.v41.real_unified_eval_v5 import (  # noqa: E402
    compare_unified_runs_v5, render_unified_comparison_md_v5, render_unified_probe_md_v5, run_unified_probe_v5)


def test_the_probe_reads_the_given_checkpoints_on_the_same_training_batches(finished, monkeypatch):
    seen = []
    original = real_unified_module.load_member_samples_v5

    def recording(**kwargs):
        seen.append((kwargs["protocol_id"], list(kwargs["roles"])))
        return original(**kwargs)

    monkeypatch.setattr(real_unified_module, "load_member_samples_v5", recording)
    final = finished.summary["training"]["checkpoint_sha256"]
    checkpoints = [(2, _sidecar(finished.root, 2)["sha256"]), (4, final)]
    result = run_unified_probe_v5(**finished.probe_kwargs, checkpoints=checkpoints, negative_topk_fraction=0.25)
    assert result["status"] == "COMPLETED", result
    assert seen == [("knee", ["train"])]                               # training batches of the box member only
    assert result["run"]["finished"] is True and result["probe"]["roles"] == ["train"]
    knee = result["members"]["knee"]
    assert (knee["batches"], knee["negative_topk_fraction"], knee["plan_negative_topk_fraction"]) == (2, 0.25, None)
    assert knee["thresholds"] == {"target_a": 0.5, "target_b": 0.5}
    rows = knee["checkpoints"]
    assert [(row["weights_step"], row["sha256"]) for row in rows] == checkpoints
    for row in rows:
        assert row["weights_unchanged"] is True and row["batches"] == 2 and len(row["state_sha256"]) == 64
        assert row["negative"]["mean"] >= row["negative_all"]["mean"]         # top quarter against all
        for item in row["classes"].values():
            assert 0 < item["negative_selected"] < item["negative_eligible"]
            assert item["high_selected"] == min(item["negative_selected"], item["high"])
            assert item["batches_all_high_selected"] <= item["batches"] == 2
            assert item["coverage_pooled"] == (item["high_selected"] / item["high"] if item["high"] else None)
            assert item["cutoff_probability"]["n"] == 2
    # the same batches under other weights: the same voxels are eligible, the values differ
    assert [item["negative_eligible"] for item in rows[0]["classes"].values()] == [
        item["negative_eligible"] for item in rows[1]["classes"].values()]
    assert rows[0]["negative_all"]["mean"] != rows[1]["negative_all"]["mean"]
    assert _tree(finished.root) == finished.before
    out = finished.probe_kwargs["private_root"]
    assert json.loads((out / "summary.json").read_text())["status"] == "COMPLETED"
    text = render_unified_probe_md_v5(result)
    assert "knee-0" not in text and str(finished.root) not in text and "NOT_EVALUATED" in text
    for phrase in ("权重未被修改", "覆盖", "0.25", checkpoints[0][1], "locked_test 未读", "不更新权重"):
        assert phrase in text
    # without a fraction nothing is selected away, and the final weights give the replay's own diagnostic
    plain = run_unified_probe_v5(**dict(finished.probe_kwargs, private_root=out.with_name("plain")),
                                 checkpoints=[(4, final)])
    row = plain["members"]["knee"]["checkpoints"][0]
    assert plain["members"]["knee"]["negative_topk_fraction"] is None and row["negative"] == row["negative_all"]
    replay = run_unified_replay_v5(**finished.replay_kwargs)
    current = replay["members"]["knee"]["current"]["train"]
    assert row["negative_all"] == current["negative"] and row["positive"] == current["positive"]


def test_the_probe_of_a_pilot_run_uses_the_fraction_of_its_plan(finished_pilot):
    final = finished_pilot.summary["training"]["checkpoint_sha256"]
    result = run_unified_probe_v5(**finished_pilot.probe_kwargs, checkpoints=[(4, final)])
    knee = result["members"]["knee"]
    assert (knee["negative_topk_fraction"], knee["plan_negative_topk_fraction"]) == (0.25, 0.25)
    _raises("PROBE_FRACTION_MISMATCH", "$.negative_topk_fraction", run_unified_probe_v5,
            **dict(finished_pilot.probe_kwargs, private_root=finished_pilot.probe_kwargs["private_root"].with_name("x")),
            checkpoints=[(4, final)], negative_topk_fraction=0.5)


def test_a_checkpoint_of_a_run_that_has_not_finished_can_be_probed_but_not_evaluated(env, tmp_path, monkeypatch):  # noqa: F811
    original = real_unified_module.train_unified_v5

    def killed_after_step_2(*args, **kwargs):
        original(*args, **kwargs, pause_at=2)
        raise RuntimeError("killed")

    with monkeypatch.context() as patch:
        patch.setattr(real_unified_module, "train_unified_v5", killed_after_step_2)
        with pytest.raises(RuntimeError):
            run_unified_training_v5(**env.train_kwargs)
    root = env.train_kwargs["private_root"]
    assert not (root / "summary.json").exists()
    common = {key: value for key, value in env.train_kwargs.items() if key != "private_root"}
    common.update(run_root=root, created_at="2026-11-12T00:00:00Z", max_wall_hours=0.5)
    selected = (2, _sidecar(root, 2)["sha256"])
    before = _tree(root)
    result = run_unified_probe_v5(**common, private_root=tmp_path / "probe", loss_steps=1, checkpoints=[selected],
                                  negative_topk_fraction=0.25)
    assert result["status"] == "COMPLETED" and result["run"]["finished"] is False
    assert result["members"]["knee"]["checkpoints"][0]["weights_step"] == 2 and _tree(root) == before
    _raises("EVAL_RUN_NOT_COMPLETED", "$.run", run_unified_evaluation_v5, **common,
            private_root=tmp_path / "evaluation", roles=["train"], checkpoint=selected)


def test_the_probe_refuses_bad_selections_before_data_is_read(finished, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("data was read before the checks")

    earlier, final = _sidecar(finished.root, 2)["sha256"], finished.summary["training"]["checkpoint_sha256"]
    kwargs = finished.probe_kwargs
    with monkeypatch.context() as patch:
        patch.setattr(real_unified_module, "load_member_samples_v5", _fail_if_called)
        for bad in ([], [(2, earlier), (2, earlier)], [(2, "x")], (2, earlier), [(0, earlier)]):
            _raises("EVAL_CHECKPOINT_INVALID", "$.checkpoints", run_unified_probe_v5, **kwargs, checkpoints=bad)
        _raises("EVAL_CHECKPOINT_MISMATCH", "$.checkpoint", run_unified_probe_v5, **kwargs, checkpoints=[(2, final)])
        _raises("EVAL_CHECKPOINT_MISSING", "$.checkpoint", run_unified_probe_v5, **kwargs, checkpoints=[(3, final)])
        for fraction in (0, 1.5, True, "0.01"):
            _raises("PROBE_FRACTION_INVALID", "$.negative_topk_fraction", run_unified_probe_v5, **kwargs,
                    checkpoints=[(4, final)], negative_topk_fraction=fraction)
        for steps in (0, -1, True, 1.5):
            _raises("INVALID_TYPE", "$.loss_steps", run_unified_probe_v5, **dict(kwargs, loss_steps=steps),
                    checkpoints=[(4, final)])
        _raises("EVAL_BUDGET_INVALID", "$.max_wall_hours", run_unified_probe_v5, **dict(kwargs, max_wall_hours=99),
                checkpoints=[(4, final)])
        _raises("EVAL_OUT_INSIDE_RUN", "$.out", run_unified_probe_v5,
                **dict(kwargs, private_root=finished.root / "probe"), checkpoints=[(4, final)])
    assert not kwargs["private_root"].exists()


def test_the_probe_detects_a_changed_weight(finished, monkeypatch):
    original = eval_module._parts

    def nudging(view, *args, **kwargs):
        next(iter(view.parameters())).data.add_(1.0)
        return original(view, *args, **kwargs)

    monkeypatch.setattr(eval_module, "_parts", nudging)
    final = finished.summary["training"]["checkpoint_sha256"]
    _raises("EVAL_WEIGHTS_CHANGED", "$.checkpoint", run_unified_probe_v5, **finished.probe_kwargs,
            checkpoints=[(4, final)])
    assert not (finished.probe_kwargs["private_root"] / "summary.json").exists()


def test_the_replay_of_a_stopped_run_ends_at_its_stop_step_and_checks_the_logged_terms(finished_pilot):
    result = run_unified_replay_v5(**finished_pilot.replay_kwargs)
    assert result["status"] == "COMPLETED", result
    knee = result["members"]["knee"]
    check = knee["verification"]
    assert (check["steps"], check["schedule_steps"], check["support_mismatches"], check["sampler_state_equal"]) == (
        4, 8, 0, True)
    assert check["terms_compared"] > 0 and check["terms_mismatches"] == 0
    assert [(item["step"], item["weights_step"]) for item in knee["history"]] == [(1, 0), (3, 2)]
    for item in knee["history"]:
        assert item["logged_terms_abs_diff"] <= 1e-5 and item["abs_diff"] <= 1e-5
        assert item["negative"] >= item["negative_all"]
    assert [row["weights_step"] for row in knee["trajectory"]] == [0, 2, 4]
    assert knee["weights"]["negative_topk_fraction"] == 0.25 and knee["exposure"]["steps"] == 4
    text = render_unified_replay_md_v5(result)
    assert "逐步记录" in text and "没有记录" not in text and "前 0.25 比例" in text and "8 步" in text


def test_a_log_without_the_terms_is_still_replayed(finished, tmp_path):
    """A run of the code before 2026-11-12 logged one number per task; its replay recovers the terms as before."""
    import shutil

    copy_root = tmp_path / "copy"
    shutil.copytree(finished.root, copy_root)
    log = copy_root / "runs" / "train-u" / "metrics.jsonl"
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    for line in lines:
        for member in line.get("members", {}).values():
            for task in member["tasks"].values():
                task.pop("box", None)
    log.chmod(0o600)
    log.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines))
    result = run_unified_replay_v5(**dict(finished.replay_kwargs, run_root=copy_root))
    knee = result["members"]["knee"]
    assert knee["verification"]["terms_compared"] == 0 and knee["verification"]["support_mismatches"] == 0
    assert all(item["logged_terms"] is None and item["abs_diff"] <= 1e-5 for item in knee["history"])
    assert knee["weights"]["negative_topk_fraction"] is None
    text = render_unified_replay_md_v5(result)
    assert "没有记录" in text and "逐步记录" not in text


def _pair(tmp_path, **control_overrides):
    """Two finished runs on the same synthetic data: a control over an 8-step schedule and a pilot stopped at 4."""
    schedule = {"budget": {"max_steps": 8}}
    worlds = {}
    for name, overrides in (("control", {"shared": {**schedule, **control_overrides}}),
                            ("trial", {"shared": dict(schedule, stop_at_step=4),
                                       "members": {"knee": {"box_supervision": {"negative_topk_fraction": 0.25}}}})):
        (tmp_path / name).mkdir()
        worlds[name] = _world(tmp_path / name, **overrides)
        assert run_unified_training_v5(**worlds[name].train_kwargs)["status"] == "COMPLETED"
    return {"plan": worlds["trial"].plan, "control_plan": worlds["control"].plan,
            "protocols": worlds["trial"].train_kwargs["protocols"],
            "run_root": worlds["trial"].train_kwargs["private_root"], "run_id": "train-u",
            "control_run_root": worlds["control"].train_kwargs["private_root"], "control_run_id": "train-u",
            "steps": 4, "checkpoint_steps": [2, 4], "code": {"git_commit": "0" * 40, "dirty": False},
            "created_at": "2026-11-12T00:00:00Z"}


def test_two_runs_are_compared_on_initialisation_sampling_and_learning_rate(tmp_path):
    kwargs = _pair(tmp_path)
    before = {name: _tree(kwargs[name]) for name in ("run_root", "control_run_root")}
    result = compare_unified_runs_v5(**kwargs, private_root=tmp_path / "comparison")
    assert result["status"] == "COMPLETED" and result["steps"] == 4
    assert result["plan_differences"] == {
        "members.knee.box_supervision.negative_topk_fraction": [None, 0.25], "shared.stop_at_step": [None, 4]}
    assert result["initialisation"]["equal"] is True
    assert result["initialisation"]["control_state_sha256"] == result["initialisation"]["trial_state_sha256"]
    assert result["learning_rate"] == {"steps_compared": 4, "mismatches": 0, "first": 0.001,
                                       "last": pytest.approx(0.001 * (1 - 3 / 8) ** 0.9, rel=1e-9)}
    assert result["support"]["compared"] == 4 * 3 and result["support"]["mismatches"] == 0     # knee A, knee U, brain U
    assert [(item["step"], item["sampler_equal"], item["scheduler_equal"], item["data_sha256_equal"],
             item["rng_equal"]) for item in result["checkpoints"]] == [(2, True, True, True, True),
                                                                        (4, True, True, True, True)]
    assert all(item["control_sha256"] != item["trial_sha256"] for item in result["checkpoints"])
    first = result["first_step_losses"]
    # the same weights and the same first batch: every loss that does not depend on the changed term is equal
    assert first["brain"]["U"]["abs_diff"] == 0.0 and first["knee"]["A"]["abs_diff"] == 0.0
    assert first["knee"]["U"]["abs_diff"] > 0.0                         # the objective that was changed
    assert result["runs"]["control"]["finished_step"] == 8 and result["runs"]["trial"]["finished_step"] == 4
    assert {name: _tree(kwargs[name]) for name in ("run_root", "control_run_root")} == before
    assert json.loads((tmp_path / "comparison" / "summary.json").read_text())["status"] == "COMPLETED"
    text = render_unified_comparison_md_v5(result)
    assert "knee-0" not in text and str(tmp_path) not in text and "NOT_EVALUATED" in text
    for phrase in ("初始化", "采样", "学习率", "negative_topk_fraction", "stop_at_step"):
        assert phrase in text


def test_a_comparison_shows_where_two_runs_were_not_run_alike(tmp_path):
    kwargs = _pair(tmp_path, seeds={"sampler": 5})                      # the control drew other patches
    result = compare_unified_runs_v5(**kwargs, private_root=tmp_path / "comparison")
    assert result["plan_differences"]["shared.seeds.sampler"] == [5, 1]
    assert [item["sampler_equal"] for item in result["checkpoints"]] == [False, False]
    assert result["initialisation"]["equal"] is True and result["learning_rate"]["mismatches"] == 0
    _raises("COMPARE_LOG_MISSING", "$.steps", compare_unified_runs_v5, **dict(kwargs, steps=6),
            private_root=tmp_path / "too_long")
    _raises("EVAL_CHECKPOINT_MISSING", "$.checkpoint", compare_unified_runs_v5,
            **dict(kwargs, checkpoint_steps=[3]), private_root=tmp_path / "no_checkpoint")
    _raises("EVAL_OUT_EXISTS", "$.out", compare_unified_runs_v5, **kwargs, private_root=tmp_path / "comparison")


def test_the_compare_command_reads_both_plans_and_keeps_the_report(tmp_path, monkeypatch, capsys):
    import yaml

    config = Path(__file__).resolve().parents[2] / "configs" / "v5"
    calls = []

    def run(**kwargs):
        assert not kwargs["private_root"].exists()                     # the driver decides about the destination
        kwargs["private_root"].mkdir(mode=0o700)
        calls.append(kwargs)
        return {"status": "COMPLETED"}

    monkeypatch.setattr(eval_module, "compare_unified_runs_v5", run)
    monkeypatch.setattr(eval_module, "render_unified_comparison_md_v5", lambda summary: "comparison report\n")
    out = tmp_path / "comparison"
    control = config / "proposals" / "unified_knee_brain.yaml"
    argv = ["compare", "--plan", str(config / "proposals" / "unified_knee_brain_negtopk_pilot.yaml"), "--protocols",
            str(config / "protocols_v5.yaml"), "--policies", str(config / "sources_v5.yaml"), "--run-root",
            str(tmp_path / "trial"), "--run-id", "run-trial", "--control-run-root", str(tmp_path / "control"),
            "--control-run-id", "run-control", "--steps", "6000", "--checkpoint-step", "2000", "--checkpoint-step",
            "4000", "--checkpoint-step", "6000"]
    assert real_unified_module.main([*argv, "--control-plan", str(control), "--out", str(out)]) == 0
    given = calls.pop()
    assert set(given) == {"plan", "control_plan", "protocols", "run_root", "run_id", "control_run_root",
                          "control_run_id", "steps", "checkpoint_steps", "private_root", "code", "created_at"}
    assert (given["plan"]["plan_id"], given["control_plan"]["plan_id"]) == (
        "unified_knee_brain_negtopk_pilot_v1", "unified_knee_brain_v1")
    assert (given["run_root"], given["run_id"], given["control_run_root"], given["control_run_id"]) == (
        tmp_path / "trial", "run-trial", tmp_path / "control", "run-control")
    assert (given["steps"], given["checkpoint_steps"], given["private_root"]) == (6000, [2000, 4000, 6000], out)
    assert set(given["protocols"]) == set(given["plan"]["members"])
    report = out / "unified_comparison.md"
    assert report.read_text() == "comparison report\n" and stat.S_IMODE(report.stat().st_mode) == 0o600
    assert capsys.readouterr().out == "comparison report\n"

    broken = yaml.safe_load(control.read_text(encoding="utf-8"))       # the control plan is validated like the plan
    broken["shared"]["stop_at_step"] = 0
    (tmp_path / "broken.yaml").write_text(yaml.safe_dump(broken))
    with pytest.raises(V5Error) as err:
        real_unified_module.main([*argv, "--control-plan", str(tmp_path / "broken.yaml"), "--out",
                                  str(tmp_path / "refused")])
    assert (err.value.code, err.value.path) == ("UNIFIED_PLAN_INVALID", "$.shared.stop_at_step")
    assert not calls and not (tmp_path / "refused").exists()


def test_an_explicit_mask_budget_reaches_every_joint_inference_and_is_recorded(finished, monkeypatch):
    """2026-11-12: the memory bound of the instance masks may be named for one evaluation (GiB); it is handed to
    every joint inference, recorded, and changes no result."""
    from anatobind.v41 import perception

    seen = []
    original = eval_module.run_perception_v5

    def recording(*args, **kwargs):
        seen.append(kwargs.get("instance_mask_bytes_limit"))
        return original(*args, **kwargs)

    monkeypatch.setattr(eval_module, "run_perception_v5", recording)
    reference = run_unified_evaluation_v5(**dict(finished.eval_kwargs, reuse_inference=None))
    assert set(seen) == {None}
    assert reference["evaluation"]["instance_mask_bytes_limit"] == perception.INSTANCE_MASK_BYTES_LIMIT
    seen.clear()
    larger = dict(finished.eval_kwargs, reuse_inference=None,
                  private_root=finished.eval_kwargs["private_root"].with_name("larger"))
    result = run_unified_evaluation_v5(**larger, instance_mask_limit_gib=64)
    assert seen and set(seen) == {64 * 2 ** 30}
    assert result["evaluation"]["instance_mask_bytes_limit"] == 64 * 2 ** 30
    for role in ("train", "development_val"):
        assert result["members"]["knee"][role]["findings"] == reference["members"]["knee"][role]["findings"]
    text = render_unified_evaluation_md_v5(result)
    assert "实例掩膜的内存上限 64 GiB" in text and "不是评价规则" in text
    assert "实例掩膜的内存上限" not in render_unified_evaluation_md_v5(reference)       # the default is not announced

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("data was read")

    monkeypatch.setattr(real_unified_module, "load_member_samples_v5", _fail_if_called)
    for bad in (0, -1, True, "64", float("inf")):
        _raises("INVALID_TYPE", "$.instance_mask_limit_gib", run_unified_evaluation_v5,
                **dict(finished.eval_kwargs, private_root=finished.eval_kwargs["private_root"].with_name("bad")),
                instance_mask_limit_gib=bad)
    assert not finished.eval_kwargs["private_root"].with_name("bad").exists()


def test_a_run_whose_own_description_stopped_at_the_mask_bound_is_still_evaluated(env, tmp_path, monkeypatch):  # noqa: F811
    """2026-11-13: more kept components on a development scan than the instance-mask bound allows stops the
    driver's own description, not the training. The stop is recorded, the checkpoint stands, and the evaluate-only
    entry reads it (never reusing the run's partly saved outputs)."""
    real = real_unified_module._infer

    def bounded(**kwargs):
        if kwargs["pid"] == "knee":
            raise V5Error("INSTANCE_LIMIT_EXCEEDED", "$.abnormal_instances")
        return real(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(real_unified_module, "_infer", bounded)
        summary = run_unified_training_v5(**env.train_kwargs)
    assert (summary["status"], summary["stop_code"], summary["stop_phase"]) == (
        "FAILED", "INSTANCE_LIMIT_EXCEEDED", "evaluation")
    assert summary["training"]["status"] == "COMPLETED" and summary["members"] is None
    root = env.train_kwargs["private_root"]
    kwargs = {key: value for key, value in env.train_kwargs.items() if key != "private_root"}
    kwargs.update(run_root=root, created_at="2026-11-12T00:00:00Z", max_wall_hours=0.5,
                  private_root=tmp_path / "evaluation", roles=["train", "development_val"])
    result = run_unified_evaluation_v5(**kwargs)
    assert result["status"] == "COMPLETED"
    assert result["checkpoint"]["sha256"] == summary["training"]["checkpoint_sha256"]
    assert result["evaluation"]["run_description_stopped"] is True
    assert result["evaluation"]["reuse_inference"] is None
    assert result["evaluation"]["reuse_refused"] == "RUN_DESCRIPTION_STOPPED"
    development = result["members"]["knee"]["development_val"]
    assert development["reuse"]["reused"] == 0 and development["reproduces_original"] is None
    assert "INSTANCE_LIMIT_EXCEEDED" in render_unified_evaluation_md_v5(result)


def test_any_other_error_of_the_description_is_not_swallowed(env, monkeypatch):  # noqa: F811
    def broken(**kwargs):
        raise V5Error("GRID_MISMATCH", "$.abnormality_map")

    monkeypatch.setattr(real_unified_module, "_infer", broken)
    _raises("GRID_MISMATCH", "$.abnormality_map", run_unified_training_v5, **env.train_kwargs)


# ---------------------------------------------------------------------------
# Audit of 2026-11-14: the probabilities read without a threshold, and what the negative support is made of


def _rank_kwargs(finished, **overrides):
    kwargs = dict(finished.eval_kwargs, private_root=finished.eval_kwargs["private_root"].with_name("ranking"),
                  n_placements=50, n_bootstrap=20)
    kwargs.update(overrides)
    return kwargs


def test_the_ranking_reads_probabilities_without_threshold_component_or_anchor(finished, monkeypatch):
    from anatobind.v41 import perception
    from anatobind.v41.real_unified_eval_v5 import render_unified_ranking_md_v5, run_unified_ranking_v5

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the ranking went through the acceptance chain")

    for owner, name in ((eval_module, "run_perception_v5"), (perception, "apply_acceptance_v5"),
                        (perception, "extract_instances_v5"), (eval_module, "findings_from_output_v5")):
        monkeypatch.setattr(owner, name, _fail_if_called)
    result = run_unified_ranking_v5(**_rank_kwargs(finished))
    assert result["status"] == "COMPLETED" and result["stop_code"] is None, result
    assert result["kind"] == "unified_ranking" and result["scientific_conclusion"] == "NOT_EVALUATED"
    assert set(result["members"]) == {"knee"}                          # the box-supervised member
    train, development = result["members"]["knee"]["train"], result["members"]["knee"]["development_val"]
    assert (train["n_scans"], development["n_scans"]) == (2, 1)
    assert train["ranking"]["boxes"]["overall"]["n_boxes"] == 1        # knee-0 has the one training box
    assert development["ranking"]["boxes"]["overall"]["n_boxes"] == 1
    groups = train["ranking"]["scans"]["target_a"]["groups"]
    assert (groups["with_primary"]["n"], groups["no_box"]["n"]) == (1, 1)
    assert result["evaluation"]["n_placements"] == 50 and result["evaluation"]["roles"] == ["train", "development_val"]
    checkpoint = result["checkpoint"]
    assert checkpoint["sha256"] == finished.summary["training"]["checkpoint_sha256"]
    assert checkpoint["weights_unchanged"] is True
    assert _tree(finished.root) == finished.before                     # the finished run is byte for byte as before
    out = _rank_kwargs(finished)["private_root"]
    rows = [json.loads(line) for line in (out / "rows" / "train.knee.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and set(rows[0]) == {"ranking"} and "patient_uid" in rows[0]["ranking"]
    for path in out.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = render_unified_ranking_md_v5(result)
    assert "knee-0" not in text and "synthetic:" not in text and str(finished.root) not in text
    for phrase in ("NOT_EVALUATED", "不用阈值", "权重未被修改", "locked_test 未读", "不能代替泛化评价", "弱标注",
                   checkpoint["sha256"], "可比摆放的上限 50"):
        assert phrase in text, phrase


def test_the_ranking_reads_the_probabilities_the_evaluation_reads(finished, monkeypatch):
    import numpy as np

    from anatobind.v41.real_unified_eval_v5 import run_unified_ranking_v5

    seen = {"evaluation": [], "ranking": []}
    diagnose, rank = eval_module.diagnose_scan_v5, eval_module.ranking_scan_v5

    def recording_diagnose(**kwargs):
        seen["evaluation"].append(np.array(kwargs["probability"]))
        return diagnose(**kwargs)

    def recording_rank(**kwargs):
        seen["ranking"].append(np.array(kwargs["probability"]))
        return rank(**kwargs)

    monkeypatch.setattr(eval_module, "diagnose_scan_v5", recording_diagnose)
    monkeypatch.setattr(eval_module, "ranking_scan_v5", recording_rank)
    run_unified_evaluation_v5(**finished.eval_kwargs)
    run_unified_ranking_v5(**_rank_kwargs(finished))
    assert len(seen["evaluation"]) == len(seen["ranking"]) == 3
    for a, b in zip(seen["evaluation"], seen["ranking"]):
        assert a.dtype == b.dtype and np.array_equal(a, b)


def test_the_placements_drawn_for_a_box_do_not_depend_on_which_roles_one_command_reads(finished):
    from anatobind.v41.real_unified_eval_v5 import run_unified_ranking_v5

    def rows(out):
        return [json.loads(line)["ranking"]["boxes"] for line in
                (out / "rows" / "development_val.knee.jsonl").read_text().splitlines()]

    both = _rank_kwargs(finished, n_placements=3)
    alone = _rank_kwargs(finished, n_placements=3, roles=["development_val"],
                         private_root=both["private_root"].with_name("ranking_dev"))
    run_unified_ranking_v5(**both)
    run_unified_ranking_v5(**alone)
    assert rows(both["private_root"]) == rows(alone["private_root"])
    assert any(box["exhaustive"] is False for scan in rows(both["private_root"]) for box in scan)   # a draw was made


def test_the_ranking_reads_a_selected_checkpoint_and_refuses_what_the_evaluation_refuses(finished, monkeypatch):
    from anatobind.v41.real_unified_eval_v5 import run_unified_ranking_v5

    earlier, final = _sidecar(finished.root, 2)["sha256"], finished.summary["training"]["checkpoint_sha256"]

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("data was read before the request was checked")

    with monkeypatch.context() as patch:
        patch.setattr(real_unified_module, "load_member_samples_v5", _fail_if_called)
        _raises("LOCKED_TEST_NOT_READABLE", "$.roles[0]", run_unified_ranking_v5,
                **_rank_kwargs(finished, roles=["locked_test"]))
        _raises("EVAL_CHECKPOINT_MISMATCH", "$.checkpoint", run_unified_ranking_v5,
                **_rank_kwargs(finished), checkpoint=(2, final))
        _raises("INVALID_TYPE", "$.n_placements", run_unified_ranking_v5, **_rank_kwargs(finished, n_placements=0))
        _raises("EVAL_BUDGET_INVALID", "$.max_wall_hours", run_unified_ranking_v5,
                **_rank_kwargs(finished, max_wall_hours=2.5))
    assert not _rank_kwargs(finished)["private_root"].exists()
    result = run_unified_ranking_v5(**_rank_kwargs(finished), checkpoint=(2, earlier))
    checkpoint = result["checkpoint"]
    assert (checkpoint["step"], checkpoint["sha256"], checkpoint["selection"], checkpoint["is_final"]) == (
        2, earlier, "explicit", False)
    _raises("EVAL_OUT_EXISTS", "$.out", run_unified_ranking_v5, **_rank_kwargs(finished))      # never written over


def test_the_ranking_detects_a_changed_weight(finished, monkeypatch):
    from anatobind.v41.real_unified_eval_v5 import run_unified_ranking_v5

    original = eval_module.restored_probabilities_v5

    def nudging(model, *args, **kwargs):
        next(iter(model.parameters())).data.add_(1.0)
        return original(model, *args, **kwargs)

    monkeypatch.setattr(eval_module, "restored_probabilities_v5", nudging)
    _raises("EVAL_WEIGHTS_CHANGED", "$.checkpoint", run_unified_ranking_v5, **_rank_kwargs(finished))
    assert not (_rank_kwargs(finished)["private_root"] / "summary.json").exists()


def _census_kwargs(finished, **overrides):
    kwargs = {key: value for key, value in finished.replay_kwargs.items() if key not in ("loss_steps", "device")}
    kwargs.update(private_root=finished.replay_kwargs["private_root"].with_name("census"))
    kwargs.update(overrides)
    return kwargs


def test_the_census_counts_what_the_negative_support_of_the_run_was_made_of(finished, monkeypatch):
    import torch

    from anatobind.v41.real_unified_eval_v5 import render_unified_census_md_v5, run_unified_census_v5

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the census built a model")

    monkeypatch.setattr(eval_module, "build_unified_model_v5", _fail_if_called)
    monkeypatch.setattr(eval_module, "load_checkpoint_v5", _fail_if_called)
    result = run_unified_census_v5(**_census_kwargs(finished))
    assert result["status"] == "COMPLETED" and result["kind"] == "unified_census", result
    assert result["census"]["roles"] == ["train"] and result["scientific_conclusion"] == "NOT_EVALUATED"
    knee = result["members"]["knee"]
    patches = 4 * finished.eval_kwargs["plan"]["members"]["knee"]["batch_size"]
    assert knee["steps"] == 4 and knee["patches"] == patches
    assert knee["verification"]["support_mismatches"] == 0 and knee["verification"]["support_compared"] == 8
    # the same counts from the run's own log: every eligible negative voxel of every step is placed once
    log = [json.loads(line) for line in (finished.root / "runs" / "train-u" / "metrics.jsonl").read_text().splitlines()]
    steps = [entry["members"]["knee"]["tasks"]["U"]["box"]["classes"] for entry in log if entry.get("event") == "step"]
    for c, name in enumerate(("target_a", "target_b")):
        block = knee["classes"][name]
        assert block["negative"]["voxels"] == sum(step[c]["negative_eligible"] for step in steps)
        assert sum(item["voxels"] for item in block["negative"]["by_tissue"].values()) == block["negative"]["voxels"]
        assert sum(item["voxels"] for item in block["negative"]["by_scan_group"].values()) == block["negative"]["voxels"]
        assert sum(item["patches"] for item in block["negative"]["by_scan_group"].values()) == patches
        assert set(block["negative"]["by_tissue"]) == {"host", "other_tissue", "background"}
        assert block["bags"] == sum(step[c]["bags"] for step in steps)
        assert block["host_share_per_batch"]["n"] == 4
    a = knee["classes"]["target_a"]["negative"]["by_scan_group"]
    assert a["other_boxes_only"]["patches"] == 0 and a["with_primary"]["patches"] + a["no_box"]["patches"] == patches
    assert knee["classes"]["target_b"]["negative"]["by_scan_group"]["with_primary"]["patches"] == 0    # no box of it
    assert _tree(finished.root) == finished.before
    text = render_unified_census_md_v5(result)
    assert "knee-0" not in text and "synthetic:" not in text and str(finished.root) not in text
    for phrase in ("NOT_EVALUATED", "只读 train", "没有构建模型", "负监督", "宿主组织", "没有任何框", "不等于正常"):
        assert phrase in text, phrase
    assert torch.is_tensor(torch.zeros(1))                             # torch stays importable; nothing was trained


def test_the_census_of_the_first_steps_only_and_its_refusals(finished, monkeypatch):
    from anatobind.v41.real_unified_eval_v5 import run_unified_census_v5

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("data was read before the request was checked")

    with monkeypatch.context() as patch:
        patch.setattr(real_unified_module, "load_member_samples_v5", _fail_if_called)
        for bad in (0, 5, True, 2.0):
            _raises("INVALID_TYPE", "$.steps", run_unified_census_v5, **_census_kwargs(finished, steps=bad))
        _raises("EVAL_BUDGET_INVALID", "$.max_wall_hours", run_unified_census_v5,
                **_census_kwargs(finished, max_wall_hours=2.5))
    result = run_unified_census_v5(**_census_kwargs(finished, steps=2))
    batch_size = finished.eval_kwargs["plan"]["members"]["knee"]["batch_size"]
    assert result["members"]["knee"]["steps"] == 2 and result["members"]["knee"]["patches"] == 2 * batch_size
    assert result["members"]["knee"]["verification"]["support_compared"] == 4


def test_the_rank_and_census_commands_hand_their_arguments_to_the_driver(tmp_path, monkeypatch, capsys):
    calls = []

    def fake(**kwargs):
        kwargs["private_root"].mkdir(mode=0o700)
        calls.append(kwargs)
        return {"status": "COMPLETED"}

    shared = _command_files(tmp_path)
    digest = "ab" * 32
    monkeypatch.setattr(eval_module, "run_unified_ranking_v5", fake)
    monkeypatch.setattr(eval_module, "render_unified_ranking_md_v5", lambda summary: "ranking report\n")
    out = tmp_path / "ranking"
    assert real_unified_module.main(["rank", *shared, "--out", str(out), "--role", "train", "--role",
                                     "development_val", "--max-wall-hours", "1.5", "--checkpoint-step", "6000",
                                     "--checkpoint-sha256", digest, "--placements", "500"]) == 0
    given = calls.pop()
    assert (given["roles"], given["checkpoint"], given["n_placements"]) == (["train", "development_val"],
                                                                           (6000, digest), 500)
    assert (given["run_root"], given["private_root"], given["max_wall_hours"]) == (tmp_path / "finished", out, 1.5)
    assert (out / "unified_ranking.md").read_text() == "ranking report\n"
    capsys.readouterr()
    assert real_unified_module.main(["rank", *shared, "--out", str(tmp_path / "ranking_final"), "--role", "train",
                                     "--max-wall-hours", "1"]) == 0
    given = calls.pop()
    assert given["checkpoint"] is None and given["n_placements"] == 1000
    capsys.readouterr()

    monkeypatch.setattr(eval_module, "run_unified_census_v5", fake)
    monkeypatch.setattr(eval_module, "render_unified_census_md_v5", lambda summary: "census report\n")
    out = tmp_path / "census"
    assert real_unified_module.main(["census", *shared, "--out", str(out), "--max-wall-hours", "2", "--steps",
                                     "6000"]) == 0
    given = calls.pop()
    assert given["steps"] == 6000 and given["max_wall_hours"] == 2.0 and "roles" not in given and "device" not in given
    assert (out / "unified_census.md").read_text() == "census report\n"
    assert capsys.readouterr().out == "census report\n"


# ---------------------------------------------------------------------------
# Amendment of 2026-11-15: the preflight, what the rows keep, the policy check of the census, the family


def test_the_preflight_checks_the_chain_on_the_first_boxed_train_scans_and_reports_no_ranking(finished, monkeypatch):
    from anatobind.v41.real_unified_eval_v5 import render_unified_ranking_preflight_md_v5, run_unified_ranking_v5

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the preflight wrote or summarised a ranking")

    with monkeypatch.context() as patch:
        patch.setattr(real_unified_module, "load_member_samples_v5", _fail_if_called)
        _raises("INVALID_ROLE", "$.roles", run_unified_ranking_v5, **_rank_kwargs(finished, preflight=1))
        for bad in (0, True, 1.0):
            _raises("INVALID_TYPE", "$.preflight", run_unified_ranking_v5,
                    **_rank_kwargs(finished, roles=["train"], preflight=bad))
        _raises("PREFLIGHT_TOO_FEW_SCANS", "$.preflight", run_unified_ranking_v5,
                **_rank_kwargs(finished, roles=["train"], preflight=2))        # one train scan has a primary box
    monkeypatch.setattr(eval_module, "summarize_ranking_v5", _fail_if_called)
    monkeypatch.setattr(eval_module, "_append_jsonl", _fail_if_called)
    kwargs = _rank_kwargs(finished, roles=["train"], preflight=1)
    result = run_unified_ranking_v5(**kwargs)
    assert result["status"] == "COMPLETED" and result["kind"] == "unified_ranking_preflight", result
    assert result["evaluation"]["preflight_scans"] == 1 and result["checkpoint"]["weights_unchanged"] is True
    check = result["members"]["knee"]["preflight"]
    assert check["scans"] == 1
    for key in ("input_normalisation", "grid_restoration", "probabilities", "box_alignment"):
        assert check[key]["pass"] is True, (key, check[key])
    assert check["box_alignment"]["boxes"] == check["box_alignment"]["boxes_with_bag"] == 1
    assert check["grid_restoration"]["min_eligible_share"] == 1.0
    assert 0.0 <= check["probabilities"]["min"] <= check["probabilities"]["max"] <= 1.0
    assert check["anatomy_agreement"]["min_mean_dice"] is not None          # a four-step model: only that it is read
    assert check["passed"] == all(check[key]["pass"] for key in (
        "input_normalisation", "grid_restoration", "probabilities", "anatomy_agreement", "box_alignment"))
    assert check["resources"]["peak_rss_bytes"] > 0 and check["resources"]["peak_gpu_bytes"] is None
    dumped = json.dumps(result)
    for word in ("percentile", "p_at_least", "pool_values", "p_normal", "knee-0"):
        assert word not in dumped, word
    assert not (kwargs["private_root"] / "rows").exists()
    assert _tree(finished.root) == finished.before
    text = render_unified_ranking_preflight_md_v5(result)
    assert "knee-0" not in text and str(finished.root) not in text
    for phrase in ("预检", "只读 train", "不含任何排序结果", "输入归一化", "恢复到原始网格", "概率的取值", "框与解剖的对位",
                   "不按结果挑选", "权重未被修改", result["checkpoint"]["sha256"], "NOT_EVALUATED"):
        assert phrase in text, phrase


def test_the_rows_of_a_ranking_keep_the_pool_the_seed_key_and_the_grid(finished):
    from anatobind.v41.real_unified_eval_v5 import run_unified_ranking_v5

    kwargs = _rank_kwargs(finished, n_placements=3)
    result = run_unified_ranking_v5(**kwargs)
    assert result["evaluation"]["preflight_scans"] is None
    for role in ("train", "development_val"):
        rows = [json.loads(line)["ranking"] for line in
                (kwargs["private_root"] / "rows" / f"{role}.knee.jsonl").read_text().splitlines()]
        for row in rows:
            assert row["grid"]["shape"] == row["shape"] and row["n_placements"] == 3
            for box in row["boxes"]:
                assert len(box["seed_key"]) == 5 and box["seed_key"][0] == result["evaluation"]["seed"]
                if box["status"] == "ok":
                    assert len(box["stats"]["topk_mean"]["pool_values"]) == box["pool"]
    assert "pool_values" not in json.dumps(result)                    # the summary carries no pool


def test_the_census_checks_that_what_the_policy_excludes_holds_no_negative(finished):
    from anatobind.v41.real_unified_eval_v5 import render_unified_census_md_v5, run_unified_census_v5

    result = run_unified_census_v5(**_census_kwargs(finished))
    for name in ("target_a", "target_b"):
        check = result["members"]["knee"]["classes"][name]["policy_check"]
        assert check["passed"] is True and check["negative_in_ignore"] == 0 and check["negative_in_own_boxes"] == 0
        assert check["scans"] == 2
    assert result["members"]["knee"]["classes"]["target_a"]["policy_check"]["own_box_voxels"] > 0
    text = render_unified_census_md_v5(result)
    for phrase in ("策略核对", "成立", "**未测量**", "不能由体素计数推断", "批次里有该类袋的步数", "可用负体素"):
        assert phrase in text, phrase
    assert "不成立" not in text


def test_the_family_entry_reads_ranking_summaries_alike_and_asks_for_their_preflights(finished, tmp_path, capsys):
    from anatobind.v41 import ranking_family_v5 as family_module
    from anatobind.v41.real_unified_eval_v5 import run_unified_ranking_v5

    earlier = _sidecar(finished.root, 2)["sha256"]
    base = _rank_kwargs(finished)["private_root"]
    final = run_unified_ranking_v5(**_rank_kwargs(finished))
    early = run_unified_ranking_v5(**_rank_kwargs(finished, private_root=base.with_name("ranking_early")),
                                   checkpoint=(2, earlier))
    flight = run_unified_ranking_v5(**_rank_kwargs(finished, roles=["train"], preflight=1,
                                                   private_root=base.with_name("preflight_final")))
    family = family_module.build_ranking_family_v5({"final": final, "early": early}, {"final": flight})
    assert family["kind"] == "ranking_family" and family["scientific_conclusion"] == "NOT_EVALUATED"
    assert family["preflight_required"] is True and family["same_support"] is True
    problems = {entry["label"]: entry["problems"] for entry in family["entries"]}
    assert "PREFLIGHT_NOT_PASSED" in problems["early"]                # no preflight was given for it
    assert ("PREFLIGHT_NOT_PASSED" in problems["final"]) == (not flight["members"]["knee"]["preflight"]["passed"])
    rows = [row for row in family["comparisons"] if row["family"]]
    assert len(rows) == 4 and all(row["reading"] == "NOT_INTERPRETABLE" for row in rows)   # one box: no support
    assert all("FEW_PATIENTS" in row["problems"] for row in rows)
    free = family_module.build_ranking_family_v5({"final": final, "early": early}, None)
    assert free["preflight_required"] is False and all(entry["problems"] == [] for entry in free["entries"])
    # what makes summaries unfit for one family
    other = json.loads(json.dumps(early))
    other["evaluation"]["n_placements"] = 7
    _raises("FAMILY_SETTINGS_DIFFER", "$.entries", family_module.build_ranking_family_v5, {"a": final, "b": other})
    _raises("FAMILY_CHECKPOINT_REPEATED", "$.entries", family_module.build_ranking_family_v5, {"a": final, "b": final})
    _raises("FAMILY_ENTRY_INVALID", "$.entries", family_module.build_ranking_family_v5, {"a": flight})
    _raises("FAMILY_ENTRY_INVALID", "$.entries", family_module.build_ranking_family_v5, {})
    failed = json.loads(json.dumps(early))
    failed.update(status="FAILED", members=None)
    listed = family_module.build_ranking_family_v5({"final": final, "early": failed}, None)
    assert {entry["label"]: entry["problems"] for entry in listed["entries"]}["early"] == ["JOB_NOT_COMPLETED"]
    assert {row["label"] for row in listed["comparisons"]} == {"final"}
    # the command
    out = tmp_path / "family"
    arguments = ["--entry", f"final={base / 'summary.json'}", "--entry",
                 f"early={base.with_name('ranking_early') / 'summary.json'}", "--out", str(out)]
    assert family_module.main([*arguments, "--preflight",
                               f"final={base.with_name('preflight_final') / 'summary.json'}"]) == 0
    assert json.loads(capsys.readouterr().out)["comparisons"] == 4
    assert json.loads((out / "family.json").read_text())["comparisons"] == family["comparisons"]
    text = (out / "ranking_family.md").read_text()
    assert "knee-0" not in text and str(finished.root) not in text and "不可判读" in text and final["checkpoint"]["sha256"] in text
    for path in out.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert family_module.main([*arguments, "--no-preflight"]) == 1      # the directory exists: never written over
    assert family_module.main(["--entry", f"final={base / 'summary.json'}", "--no-preflight", "--preflight", "final=x",
                               "--out", str(tmp_path / "other")]) == 1
    assert not (tmp_path / "other").exists()


def test_the_rank_command_hands_a_preflight_to_the_driver_and_fails_when_a_check_fails(tmp_path, monkeypatch, capsys):
    calls, passed = [], [True]

    def fake(**kwargs):
        kwargs["private_root"].mkdir(mode=0o700)
        calls.append(kwargs)
        return {"status": "COMPLETED", "members": {"knee": {"preflight": {"passed": passed[0]}}}}

    shared = _command_files(tmp_path)
    monkeypatch.setattr(eval_module, "run_unified_ranking_v5", fake)
    monkeypatch.setattr(eval_module, "render_unified_ranking_preflight_md_v5", lambda summary: "preflight report\n")
    out = tmp_path / "preflight"
    command = ["rank", *shared, "--role", "train", "--max-wall-hours", "0.1", "--preflight", "2"]
    assert real_unified_module.main([*command, "--out", str(out)]) == 0
    assert calls.pop()["preflight"] == 2
    assert (out / "unified_ranking_preflight.md").read_text() == "preflight report\n"
    passed[0] = False
    assert real_unified_module.main([*command, "--out", str(tmp_path / "preflight_failed")]) != 0
    capsys.readouterr()
