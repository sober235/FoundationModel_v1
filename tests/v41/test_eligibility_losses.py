import numpy as np
import pytest
import torch

from anatobind.v41.eligibility import (
    annotation_eligibility,
    build_eligibility,
    supervision_counts,
)
from anatobind.v41.losses import aggregate_task_losses, masked_bce, masked_dice_loss


def _annotation(**overrides):
    base = {
        "annotation_uid": "ann-1",
        "task": "U",
        "class_scope": ["cartilage_lesion"],
        "representation": "voxel",
        "path": "annotations/synth/ann-1.nii.gz",
        "label_map_version": "synth-u-v1",
        "provenance_type": "manual",
        "creator_or_model": "synthetic_reader",
        "creator_version": "0",
        "parent_annotation_uid": None,
        "completeness": "target_complete",
        "review_region_path": None,
        "input_visibility": "visible",
        "quality_status": "accepted",
        "intended_role": "train",
        "reference_role": "expert_voxel",
        "adjudication_status": "single_reader",
        "reader_ids": ["reader_a"],
        "protocol_hash": "0" * 64,
        "visibility_evidence": "synthetic",
    }
    base.update(overrides)
    return base


def test_unreviewed_has_zero_gradient():
    logits = torch.zeros(2, requires_grad=True)
    loss = masked_bce(logits, torch.tensor([1., 0.]), torch.tensor([True, False]))
    loss.backward()
    assert logits.grad[1].item() == 0.0


def test_masked_bce_matches_manual_value():
    logits = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 1.0])
    eligible = torch.tensor([True, False, True])
    loss = masked_bce(logits, target, eligible)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[[0, 2]], target[[0, 2]])
    assert torch.isclose(loss, expected)


def test_all_ignored_returns_finite_zero():
    logits = torch.tensor([3.0, -2.0], requires_grad=True)
    target = torch.tensor([1.0, 1.0])
    eligible = torch.tensor([False, False])
    loss = masked_bce(logits, target, eligible)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert torch.all(logits.grad == 0)


def test_masked_dice_excludes_numerator_and_denominator():
    logits = torch.tensor([2.0, -2.0, 0.0, 0.0])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    eligible = torch.tensor([True, True, False, False])
    loss = masked_dice_loss(logits, target, eligible)
    p = torch.sigmoid(logits[:2])
    t = target[:2]
    expected = 1 - (2 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
    assert torch.isclose(loss, expected)


def test_changing_ineligible_logits_does_not_change_masked_dice():
    logits = torch.tensor([2.0, -2.0, 0.5])
    target = torch.tensor([1.0, 0.0, 1.0])
    eligible = torch.tensor([True, True, False])
    before = masked_dice_loss(logits, target, eligible).item()
    for wild in (50.0, -50.0, 1e4):
        changed = logits.clone()
        changed[2] = wild
        assert masked_dice_loss(changed, target, eligible).item() == pytest.approx(before)


def test_zero_support_task_does_not_dilute_mean():
    active = torch.tensor(0.5)
    zero = torch.zeros((), requires_grad=True)
    losses = {"A": active, "U": zero}
    supports = {"A": 100, "U": 0}
    weights = {"A": 1.0, "U": 1.0}
    total = aggregate_task_losses(losses, supports, weights)
    assert total.item() == pytest.approx(0.5)
    total2 = aggregate_task_losses({"A": active}, {"A": 0}, {"A": 1.0})
    assert torch.isfinite(total2) and total2.item() == 0.0


def test_weights_scale_tasks_but_support_gates_them():
    a = torch.tensor(1.0)
    b = torch.tensor(3.0)
    total = aggregate_task_losses({"A": a, "U": b}, {"A": 10, "U": 10}, {"A": 1.0, "U": 3.0})
    assert total.item() == pytest.approx((1.0 * 1.0 + 3.0 * 3.0) / 4.0)
    total = aggregate_task_losses({"A": a, "U": b}, {"A": 10, "U": 0}, {"A": 1.0, "U": 3.0})
    assert total.item() == pytest.approx(1.0)


def test_complete_class_annotation_gives_no_negative_for_other_classes():
    ann = _annotation(class_scope=["cartilage_lesion"], completeness="target_complete")
    decision = annotation_eligibility(ann, task="U", class_scope=["cartilage_lesion", "meniscal_tear"])
    assert decision["classes"]["cartilage_lesion"]["positive"] is True
    assert decision["classes"]["cartilage_lesion"]["negative"] is True
    assert decision["classes"]["meniscal_tear"]["positive"] is False
    assert decision["classes"]["meniscal_tear"]["negative"] is False
    assert decision["role"] == "qualified"


def test_removed_low_confidence_box_is_not_background():
    ann = _annotation(representation="bbox", completeness="positive_only",
                      provenance_type="manual", quality_status="rejected")
    decision = annotation_eligibility(ann, task="U", class_scope=["cartilage_lesion"])
    assert decision["role"] == "none"
    assert decision["classes"]["cartilage_lesion"]["negative"] is False
    ann_ok = _annotation(representation="bbox", completeness="positive_only")
    decision = annotation_eligibility(ann_ok, task="U", class_scope=["cartilage_lesion"])
    assert decision["classes"]["cartilage_lesion"]["positive"] is False
    assert decision["classes"]["cartilage_lesion"]["negative"] is False
    assert decision["localization_only"] is True


def test_box_fill_and_automated_never_become_qualified():
    for provenance in ("box_fill", "automated", "geometry", "unknown"):
        ann = _annotation(provenance_type=provenance)
        assert annotation_eligibility(ann, "U", ["cartilage_lesion"])["role"] in ("pseudo", "none")
    assert annotation_eligibility(_annotation(provenance_type="box_fill"), "U", ["cartilage_lesion"])["role"] == "none"
    assert annotation_eligibility(_annotation(provenance_type="automated", task="A"), "A", ["patellar_cartilage"])["role"] == "pseudo"
    assert annotation_eligibility(_annotation(provenance_type="expert_corrected"), "U", ["cartilage_lesion"])["role"] == "qualified"


def test_build_eligibility_region_and_positive_only_masks():
    record = {
        "annotation_records": [
            _annotation(annotation_uid="full", completeness="target_complete", class_scope=["lesion"]),
        ]
    }
    eligible = build_eligibility(record, "U", ["lesion"], shape=(4, 4))
    assert eligible.shape == (1, 4, 4) and eligible.all()

    region = np.zeros((4, 4), dtype=bool)
    region[:2, :2] = True
    record["annotation_records"] = [
        _annotation(annotation_uid="reg", completeness="region_complete", class_scope=["lesion"],
                    review_region_path="regions/reg.nii.gz"),
    ]
    eligible = build_eligibility(record, "U", ["lesion"], shape=(4, 4),
                                 review_regions={"regions/reg.nii.gz": region})
    assert eligible[0].tolist() == region.tolist()

    record["annotation_records"] = [
        _annotation(annotation_uid="pos", completeness="positive_only", class_scope=["lesion"]),
    ]
    target = np.zeros((4, 4), dtype=np.int16)
    target[0, 0] = 1
    eligible = build_eligibility(record, "U", ["lesion"], shape=(4, 4),
                                 targets={"pos": target})
    assert eligible[0].sum() == 1 and eligible[0, 0, 0]


def test_unreviewed_and_missing_scope_have_no_eligibility():
    record = {"annotation_records": [_annotation(completeness="unreviewed")]}
    eligible = build_eligibility(record, "U", ["lesion"], shape=(2, 2))
    assert not eligible.any()
    record = {"annotation_records": []}
    assert not build_eligibility(record, "U", ["lesion"], shape=(2, 2)).any()


def test_supervision_counts_separate_roles_and_signs():
    record = {
        "sample_uid": "synth",
        "source_dataset": "synthetic",
        "annotation_records": [
            _annotation(annotation_uid="m", completeness="target_complete", class_scope=["lesion"]),
            _annotation(annotation_uid="a", provenance_type="automated", task="A",
                        class_scope=["patellar_cartilage"], reference_role="pseudo_only"),
        ],
    }
    eligible_u = build_eligibility(record, "U", ["lesion"], shape=(3, 3))
    target_u = np.zeros((1, 3, 3), dtype=np.int16)
    target_u[0, 0, 0] = 1
    rows = supervision_counts(record, {"U": (eligible_u, target_u)})
    row = rows[0]
    assert row["task"] == "U" and row["role"] == "qualified"
    assert row["positive_voxels"] == 1 and row["negative_voxels"] == 8
    assert row["pseudo_voxels"] == 0
