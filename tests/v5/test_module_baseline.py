"""v5.1 GPU measurement and module baseline reporting (plan_v5 T10; Proposal §1.1, §6.1, §8.3). CPU, synthetic."""
from __future__ import annotations

import copy
import re

import numpy as np
import pytest
import torch

from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.engine.module_eval_v5 import (
    freeze_module_reference_v5, module_metrics_v5, render_module_baseline_md_v5, summarize_module_metrics_v5)
from anatobind.v41.engine.smoke_v5 import measure_candidates_v5, overfit_check_v5, render_measurement_md_v5
from anatobind.v41.models.interface_v5 import PerceptionModelV5, collate_model_inputs_v5
from anatobind.v41.models.resunet_v5 import build_model_v5
from tests.v5.factories import (
    make_acceptance_policy, make_metric_policy, make_model_input_v5, make_module_reference, make_module_scope,
    make_recipe, make_scored_output, make_supervision, make_training_sample_v5)

CANDIDATE = {"patch_size": [4, 8, 8], "batch_size": 2, "precision": "fp32"}
SHAPE = (6, 6, 6)
SECRETS = ("synthetic-case-0042", "/private/", ".nii.gz")


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _a_scope():
    return validate_scope_v5(make_module_scope("A"))


def _au_scope():
    return validate_scope_v5(make_module_scope("A", scope_id="synthetic_au", abnormality_classes=["target_a"]))


def _make_batch(scope):
    def make(candidate, device):
        items = [make_model_input_v5(scope, candidate["patch_size"]) for _ in range(candidate["batch_size"])]
        return collate_model_inputs_v5(items), lambda output: output.anatomy_logits.float().mean()
    return make


# ---------------------------------------------------------------------------
# GPU measurement and the small-sample overfit check


def test_measurement_needs_explicit_candidates():
    scope = _a_scope()
    built = []

    def build(candidate):
        built.append(candidate)
        return build_model_v5(make_recipe(), scope)

    make = _make_batch(scope)
    for empty in ([], None, {}):
        _raises("CANDIDATES_REQUIRED", "$.candidates", measure_candidates_v5, build, make, empty, "cpu")
    for key in CANDIDATE:
        partial = {k: v for k, v in CANDIDATE.items() if k != key}
        _raises("CANDIDATE_VALUE_MISSING", f"$.candidates[1].{key}", measure_candidates_v5, build, make,
                [CANDIDATE, partial], "cpu")
    _raises("UNKNOWN_FIELD", "$.candidates[0].<unknown>", measure_candidates_v5, build, make,
            [dict(CANDIDATE, lr=0.1)], "cpu")
    for key, bad in (("patch_size", [4, 8]), ("batch_size", 0), ("precision", "fp8")):
        _raises("INVALID_CANDIDATE_VALUE", f"$.candidates[0].{key}", measure_candidates_v5, build, make,
                [dict(CANDIDATE, **{key: bad})], "cpu")
    assert built == []  # nothing runs until every candidate is valid
    rows = measure_candidates_v5(build, make, [CANDIDATE], "cpu")
    assert len(built) == 1 and len(rows) == 1
    row = rows[0]
    assert {key: row[key] for key in CANDIDATE} == CANDIDATE and row["status"] == "OK"
    assert row["peak_memory_bytes"] is None  # no CUDA allocator on CPU
    assert row["seconds_per_step"] > 0 and row["samples_per_second"] == pytest.approx(2 / row["seconds_per_step"])


class _ExhaustedModel(PerceptionModelV5):
    """Every forward runs out of device memory; counts the attempts."""

    def __init__(self, scope):
        super().__init__(scope, ("A",))
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def _forward_v5(self, batch):
        self.calls += 1
        raise torch.cuda.OutOfMemoryError("synthetic")


def test_oom_candidate_recorded_not_retried():
    scope = _a_scope()
    big = dict(CANDIDATE, patch_size=[8, 16, 16])
    exhausted, built = _ExhaustedModel(scope), []

    def build(candidate):
        built.append(candidate["patch_size"])
        return exhausted if candidate["patch_size"] == big["patch_size"] else build_model_v5(make_recipe(), scope)

    rows = measure_candidates_v5(build, _make_batch(scope), [CANDIDATE, big, dict(CANDIDATE, batch_size=1)], "cpu")
    assert [row["status"] for row in rows] == ["OK", "OOM", "OK"]
    assert exhausted.calls == 1 and built == [[4, 8, 8], [8, 16, 16], [4, 8, 8]]
    assert rows[1] == dict(big, status="OOM", peak_memory_bytes=None, seconds_per_step=None, samples_per_second=None)

    def broken(candidate):
        raise RuntimeError("synthetic fault")

    with pytest.raises(RuntimeError):  # any other failure is a fault, not a measurement
        measure_candidates_v5(broken, _make_batch(scope), [CANDIDATE], "cpu")


def _measurement_summary() -> dict:
    rows = [dict(CANDIDATE, status="OK", peak_memory_bytes=3 * 2 ** 30, seconds_per_step=0.5, samples_per_second=4.0),
            dict(CANDIDATE, patch_size=[8, 16, 16], status="OOM", peak_memory_bytes=None, seconds_per_step=None,
                 samples_per_second=None)]
    overfit = {"steps": 3, "steps_completed": 3, "loss_curve": [1.0, 0.7, 0.4], "initial_loss": 1.0,
               "final_loss": 0.4, "loss_drop": 0.6, "dice": {"A": 0.8},
               "criteria": {"min_loss_drop": 0.5, "min_dice": 0.5}, "passed": True, "failure_code": None}
    return {"scope_id": "synthetic_a", "device": {"name": "Synthetic GPU", "count": 8}, "candidates": rows,
            "overfit": overfit, "case_id": "synthetic-case-0042", "run_dir": "/private/root/runs/smoke",
            "samples": ["synthetic-case-0042"], "image": "synthetic-case-0042_0000.nii.gz"}


def test_measurement_report_is_redacted():
    summary = _measurement_summary()
    text = render_measurement_md_v5(summary)
    assert not any(secret in text for secret in SECRETS)
    for shown in ("synthetic_a", "Synthetic GPU × 8", "4×8×8", "8×16×16", "OOM", "3.00", "0.500", "PASS"):
        assert shown in text
    _raises("UNSAFE_REPORT_VALUE", "$.device.name", render_measurement_md_v5,
            dict(summary, device={"name": "/private/gpu", "count": 1}))
    _raises("UNSAFE_REPORT_VALUE", "$.scope_id", render_measurement_md_v5,
            dict(summary, scope_id="synthetic-case-0042/x"))


def _learnable_samples(scope) -> dict:
    samples = {}
    for seed, uid in enumerate(("synthetic-sample-1", "synthetic-sample-2")):
        sample = make_training_sample_v5(scope, uid, seed=seed)
        intensity = sample.observation.channels[scope.input_modalities[0]]
        sample.targets["A"] = np.where(intensity > 0.5, 1, np.where(intensity < -0.5, 2, 0)).astype(np.int64)
        samples[uid] = sample
    return samples


def test_overfit_check_uses_explicit_criteria():
    scope = _a_scope()
    recipe = make_recipe(optimizer={"name": "adamw", "lr": 0.01, "weight_decay": 0.0},
                         budget={"max_steps": 30, "max_wall_hours": 0.1})
    samples, supervision = _learnable_samples(scope), make_supervision(scope)
    model = build_model_v5(recipe, scope)
    lenient = {"min_loss_drop": 0.0, "min_dice": 0.0}
    for criteria, path in (({}, "$.criteria.min_loss_drop"), ({"min_loss_drop": 0.1}, "$.criteria.min_dice")):
        _raises("CRITERIA_VALUE_MISSING", path, overfit_check_v5, model, samples, recipe, supervision, 5, criteria)
    _raises("INVALID_CRITERIA_VALUE", "$.criteria.min_dice", overfit_check_v5, model, samples, recipe, supervision,
            5, dict(lenient, min_dice=1.5))
    _raises("UNKNOWN_FIELD", "$.criteria.<unknown>", overfit_check_v5, model, samples, recipe, supervision, 5,
            dict(lenient, max_steps=3))
    for steps in (0, 31):  # past budget.max_steps the poly schedule has reached zero
        _raises("INVALID_STEPS", "$.steps", overfit_check_v5, model, samples, recipe, supervision, steps, lenient)
    result = overfit_check_v5(model, samples, recipe, supervision, 30, lenient)
    assert result["passed"] and result["failure_code"] is None and len(result["loss_curve"]) == 30
    assert result["loss_drop"] == pytest.approx(1 - result["final_loss"] / result["initial_loss"])
    assert result["loss_drop"] > 0
    assert set(result["dice"]) == {"A"} and 0 < result["dice"]["A"] <= 1
    assert "synthetic-sample" not in repr(result)  # nothing identifying enters what a report may show
    strict = overfit_check_v5(build_model_v5(recipe, scope), samples, recipe, supervision, 2,
                              {"min_loss_drop": 0.99, "min_dice": 1.0})
    assert strict["passed"] is False and strict["failure_code"] is None


# ---------------------------------------------------------------------------
# Module results on development_val


def _tissue() -> np.ndarray:
    truth = np.zeros(SHAPE, dtype=np.int16)
    truth[:2, :2, :2] = 1  # tissue_a, 8 voxels
    return truth


def _lesion() -> np.ndarray:
    truth = np.zeros((1, *SHAPE), dtype=bool)
    truth[0, 3:5, 3:5, 3:5] = True  # target_a, 8 voxels
    return truth


def test_module_reference_frozen_before_prediction():
    scope = _a_scope()
    references = {uid: make_module_reference(scope, patient, SHAPE, anatomy_truth=_tissue())
                  for uid, patient in (("obs-1", "synthetic-patient-a"), ("obs-2", "synthetic-patient-b"))}
    frozen = freeze_module_reference_v5(references, scope)
    assert re.fullmatch(r"[0-9a-f]{64}", frozen["reference_set_hash"])
    assert freeze_module_reference_v5(copy.deepcopy(references), scope)["reference_set_hash"] \
        == frozen["reference_set_hash"]
    changed = copy.deepcopy(references)
    changed["obs-2"]["tasks"]["A"]["eligible"][0, 0, 0, 0] = False
    other = freeze_module_reference_v5(changed, scope)
    assert other["reference_set_hash"] != frozen["reference_set_hash"]
    output = make_scored_output(scope, "obs-1", SHAPE, anatomy_map=_tissue())
    policy, acceptance = make_metric_policy(), make_acceptance_policy(scope)
    # Editing the caller's arrays after freezing reaches nothing that is evaluated.
    references["obs-1"]["tasks"]["A"]["truth"][:] = 0
    row = module_metrics_v5(output, frozen, policy, acceptance)
    assert row["segmentation"]["A"]["tissue_a"]["dice"] == 1.0
    assert row["reference_set_hash"] == frozen["reference_set_hash"]
    # An entry re-signed after freezing no longer matches the set hash.
    resigned = dict(frozen, entries=dict(frozen["entries"], **{"obs-2": other["entries"]["obs-2"]}),
                    entry_sha256=dict(frozen["entry_sha256"], **{"obs-2": other["entry_sha256"]["obs-2"]}))
    _raises("REFERENCE_HASH_MISMATCH", "$.reference_set_hash", module_metrics_v5,
            make_scored_output(scope, "obs-2", SHAPE, anatomy_map=_tissue()), resigned, policy, acceptance)
    # Frozen arrays are read-only, and a swapped evaluation region is refused at evaluation.
    entry = frozen["entries"]["obs-1"]
    with pytest.raises(ValueError):
        entry["tasks"]["A"]["eligible"][0, 0, 0, 0] = False
    entry["tasks"]["A"]["eligible"] = np.zeros_like(entry["tasks"]["A"]["eligible"])
    _raises("REFERENCE_HASH_MISMATCH", "$.reference", module_metrics_v5, output, frozen, policy, acceptance)


def test_module_evaluation_binds_reference_scope_and_policy():
    scope = _a_scope()
    reference = make_module_reference(scope, "synthetic-patient-a", SHAPE, anatomy_truth=_tissue())
    frozen = freeze_module_reference_v5({"obs-1": reference}, scope)
    policy, acceptance = make_metric_policy(), make_acceptance_policy(scope)
    output = make_scored_output(scope, "obs-1", SHAPE, anatomy_map=_tissue())
    _raises("SPLIT_NOT_ALLOWED", "$.references[0].split", freeze_module_reference_v5,
            {"obs-1": dict(reference, split="locked_test")}, scope)
    _raises("REFERENCES_REQUIRED", "$.references", freeze_module_reference_v5, {}, scope)
    _raises("REFERENCE_MISSING", "$.observation_uid", module_metrics_v5,
            make_scored_output(scope, "obs-9", SHAPE, anatomy_map=_tissue()), frozen, policy, acceptance)
    other = validate_scope_v5(make_module_scope("A", scope_id="synthetic_a_other"))
    _raises("SCOPE_MISMATCH", "$.scope_hash", module_metrics_v5,
            make_scored_output(other, "obs-1", SHAPE, anatomy_map=_tissue()), frozen, policy,
            make_acceptance_policy(other))
    _raises("POLICY_ROLE_NOT_ALLOWED", "$.acceptance_policy.policy_role", module_metrics_v5, output, frozen, policy,
            make_acceptance_policy(scope, policy_role="calibrated"))
    _raises("POLICY_MISMATCH", "$.acceptance_policy.policy_id", module_metrics_v5, output, frozen, policy,
            make_acceptance_policy(scope, policy_id="synthetic_acceptance_v2"))
    _raises("GRID_MISMATCH", "$.geometry", module_metrics_v5,
            make_scored_output(scope, "obs-1", SHAPE, anatomy_map=_tissue(), units_reliable=False), frozen, policy,
            acceptance)
    _raises("METRIC_POLICY_NULL", "$.metrics", module_metrics_v5, output, frozen, make_metric_policy(metrics=None),
            acceptance)


def test_rejected_voxels_stay_in_denominator():
    scope = _au_scope()
    truth_a = np.zeros(SHAPE, dtype=np.int16)
    truth_a[:2] = 1  # 72 voxels of tissue_a
    predicted_a = truth_a.copy()
    predicted_a[:2, 3:] = -1  # half of them rejected
    accepted_u = np.ones((1, *SHAPE), dtype=bool)
    accepted_u[0, 3] = False  # one slab of the lesion rejected
    reference = make_module_reference(scope, "synthetic-patient-a", SHAPE, anatomy_truth=truth_a,
                                      abnormality_truth=_lesion())
    frozen = freeze_module_reference_v5({"obs-1": reference}, scope)
    output = make_scored_output(scope, "obs-1", SHAPE, anatomy_map=predicted_a, abnormality_map=_lesion(),
                                abnormality_accepted=accepted_u)
    row = module_metrics_v5(output, frozen, make_metric_policy(), make_acceptance_policy(scope))
    assert row["segmentation"]["A"]["tissue_a"]["dice"] == pytest.approx(2 * 36 / (36 + 72))
    assert row["segmentation"]["A"]["tissue_b"]["dice"] is None  # absent on both sides
    assert row["segmentation"]["U"]["target_a"]["dice"] == pytest.approx(2 * 4 / (4 + 8))
    detection = row["detection"]
    assert (detection["tp"], detection["fp"], detection["fn"]) == (1, 0, 0)
    assert detection["matched_volume_error_voxels"] == [-4]


def test_module_mm_metrics_need_verified_geometry():
    scope = _au_scope()
    for reliable in (True, False):
        reference = make_module_reference(scope, "synthetic-patient-a", SHAPE, anatomy_truth=_tissue(),
                                          abnormality_truth=_lesion(), units_reliable=reliable)
        output = make_scored_output(scope, "obs-1", SHAPE, anatomy_map=np.roll(_tissue(), 1, axis=0),
                                    abnormality_map=np.roll(_lesion(), 1, axis=1), units_reliable=reliable)
        row = module_metrics_v5(output, freeze_module_reference_v5({"obs-1": reference}, scope),
                                make_metric_policy(), make_acceptance_policy(scope))
        tissue, lesion = row["segmentation"]["A"]["tissue_a"], row["segmentation"]["U"]["target_a"]
        assert tissue["dice"] == pytest.approx(0.5) and lesion["dice"] == pytest.approx(0.5)
        millimetres = (tissue["hd95_mm"], tissue["nsd"], lesion["hd95_mm"],
                       row["detection"]["matched_volume_error_mm3"])
        if reliable:
            assert all(value is not None for value in millimetres)
        else:
            assert all(value is None for value in millimetres)
            assert "PHYSICAL_GEOMETRY_UNKNOWN" in tissue["reasons"]


def test_reference_lesions_keep_every_component():
    scope = _au_scope()
    truth = _lesion()
    truth[0, 0, 0, 0] = True  # a one-voxel reference lesion
    reference = make_module_reference(scope, "synthetic-patient-a", SHAPE, anatomy_truth=_tissue(),
                                      abnormality_truth=truth)
    output = make_scored_output(scope, "obs-1", SHAPE, anatomy_map=_tissue(), abnormality_map=_lesion())
    acceptance = make_acceptance_policy(scope, instances={"connectivity": 26, "min_voxels": 2})
    row = module_metrics_v5(output, freeze_module_reference_v5({"obs-1": reference}, scope), make_metric_policy(),
                            acceptance)
    # The size filter shapes predictions only; the small reference lesion stays in the denominator as a miss.
    detection = row["detection"]
    assert (detection["tp"], detection["fn"], detection["sensitivity"]) == (1, 1, 0.5)


def _patient_rows() -> list:
    """Three perfect scans of one patient, one failed scan of another (tissue missed, lesion missed, 2 FP)."""
    scope = _au_scope()
    plan = (("obs-1", "synthetic-patient-a"), ("obs-2", "synthetic-patient-a"), ("obs-3", "synthetic-patient-a"),
            ("obs-4", "synthetic-patient-b"))
    frozen = freeze_module_reference_v5(
        {uid: make_module_reference(scope, patient, SHAPE, anatomy_truth=_tissue(), abnormality_truth=_lesion())
         for uid, patient in plan}, scope)
    missed = np.zeros((1, *SHAPE), dtype=bool)
    missed[0, 0, 5, 0] = missed[0, 5, 0, 5] = True
    rows = []
    for uid, patient in plan:
        if patient == "synthetic-patient-a":
            output = make_scored_output(scope, uid, SHAPE, anatomy_map=_tissue(), abnormality_map=_lesion())
        else:
            output = make_scored_output(scope, uid, SHAPE, anatomy_map=np.zeros(SHAPE, dtype=np.int16),
                                        abnormality_map=missed)
        rows.append(module_metrics_v5(output, frozen, make_metric_policy(), make_acceptance_policy(scope)))
    return rows


def test_module_summary_is_patient_weighted():
    rows = _patient_rows()
    summary = summarize_module_metrics_v5(rows)
    assert (summary["n_patients"], summary["n_scans"]) == (2, 4)
    assert summary["split"] == "development_val" and summary["scientific_conclusion"] == "NOT_EVALUATED"
    segmentation = summary["segmentation"]
    # Patient-equal weight gives 0.5; the scan mean would give 0.75.
    assert segmentation["A"]["tissue_a"]["dice"] == {"mean": 0.5, "n_patients": 2, "n_scans": 4, "n_none": 0}
    assert segmentation["U"]["target_a"]["dice"]["mean"] == pytest.approx(0.5)
    assert segmentation["A"]["tissue_b"]["dice"] == {"mean": None, "n_patients": 0, "n_scans": 0, "n_none": 4}
    assert segmentation["A"]["tissue_a"]["hd95_mm"] == {"mean": 0.0, "n_patients": 1, "n_scans": 3, "n_none": 1}
    # Lesion detection pools scans within a patient (T06c), then weights patients equally.
    detection = summary["detection"]
    assert detection["pooled_sensitivity"] == {"mean": 0.5, "n_patients": 2, "n_scans": 4, "n_none": 0}
    assert detection["fp_per_scan"] == {"mean": 1.0, "n_patients": 2, "n_scans": 4, "n_none": 0}
    mixed = copy.deepcopy(rows)
    mixed[2]["reference_set_hash"] = "0" * 64
    _raises("MIXED_ROWS", "$.rows[2].reference_set_hash", summarize_module_metrics_v5, mixed)
    _raises("DUPLICATE_OBSERVATION", "$.rows[4]", summarize_module_metrics_v5, rows + [rows[0]])
    _raises("ROWS_REQUIRED", "$.rows", summarize_module_metrics_v5, [])


def test_module_report_is_redacted_and_not_a_conclusion():
    summary = summarize_module_metrics_v5(_patient_rows())
    text = render_module_baseline_md_v5(dict(summary, case_id="synthetic-case-0042",
                                             source="/private/raw/synthetic-case-0042_0000.nii.gz"))
    assert not any(secret in text for secret in SECRETS + ("obs-", "synthetic-patient"))
    for shown in ("synthetic_au", "split: development_val", "scientific_conclusion: NOT_EVALUATED", "不相加",
                  "tissue_a", "target_a", "0.500", "1.000", summary["reference_set_hash"]):
        assert shown in text
    _raises("INVALID_SUMMARY", "$.split", render_module_baseline_md_v5, dict(summary, split="locked_test"))
    _raises("INVALID_SUMMARY", "$.scientific_conclusion", render_module_baseline_md_v5,
            dict(summary, scientific_conclusion="SUPPORTED"))
