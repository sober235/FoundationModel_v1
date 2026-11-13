"""Replay of one member's patch sampling and the parts of the box objective (diagnostics, 2026-11-11).

Training logs one number per task and step. What the model was actually shown is recovered here by drawing the
same patches again: a member's sampler is a seeded PCG64 stream that depends only on the member's samples, never on
the weights, so the batches of a finished run can be rebuilt step by step. The replay is checked against the
run's own log (the support of every task at every step) and against the sampler state kept in a checkpoint.

`box_objective_parts_v5` gives the terms that box_mil_objective_v5 adds, through the function training itself
uses (losses.box_mil_terms_v5). Runs before 2026-11-12 never logged them apart; for those they can be recomputed
only where the weights are known (a checkpoint, or the seeded initialisation) and are then diagnostics of those
weights. Nothing here updates a model.
"""
from __future__ import annotations

import torch

from ..losses import box_mil_terms_v5, box_terms_record_v5
from .box_data_v5 import box_positive_centers_v5
from .data_v5 import PatchSamplerV5, positive_centers_v5
from .train_unified_v5 import _bag_count
from .train_v5 import _support


def box_objective_parts_v5(logits: torch.Tensor, bags: torch.Tensor, e_pos: torch.Tensor, e_neg: torch.Tensor,
                           spec: dict) -> dict:
    """The terms of box_mil_objective_v5 as numbers: box_terms_record_v5 of box_mil_terms_v5, plus
    "negative_voxels" (the e_neg voxels of all classes). The same record a training log line keeps.

    positive: mean over bags of the mean softplus(-logit) of each bag's top ceil(topk_fraction·n) logits;
    negative: the optimised negative term (every e_neg voxel of a class, or its top negative_topk_fraction);
    negative_all: the mean over every e_neg voxel whatever the plan keeps; classes: the same per class with the
    eligible and selected voxel counts. The objective is positive_weight·positive + negative_weight·negative; a
    part without support is None."""
    record = box_terms_record_v5(box_mil_terms_v5(logits, bags, e_pos, e_neg, spec))
    return {**record, "negative_voxels": sum(entry["negative_eligible"] for entry in record["classes"])}


class RecordingSamplerV5(PatchSamplerV5):
    """PatchSamplerV5 that also remembers which samples it handed out, in order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drawn: list = []

    def next_uid(self) -> str:
        uid = super().next_uid()
        self.drawn.append(uid)
        return uid


def member_sampler_v5(samples: dict, recipe: dict, seed: int, box_supervision: dict | None) -> RecordingSamplerV5:
    """The sampler train_unified_v5 builds for a member: same samples, centres, positive fraction and seed."""
    centers = box_positive_centers_v5 if box_supervision is not None else positive_centers_v5
    return RecordingSamplerV5(sorted(samples), {uid: centers(sample) for uid, sample in samples.items()},
                              recipe["sampling"]["positive_fraction"], seed)


def full_bag_voxels_v5(samples: dict) -> dict:
    """{(sample uid, class index, bag id): supervised voxels of the whole bag} of box-supervised samples."""
    sizes = {}
    for uid, sample in samples.items():
        bags, e_pos = sample.targets["U"], sample.e_pos["U"]
        for c in range(bags.shape[0]):
            values = bags[c][e_pos[c] & (bags[c] > 0)]
            for bag_id in sorted({int(v) for v in values}):
                sizes[(uid, c, bag_id)] = int((values == bag_id).sum())
    return sizes


def batch_exposure_v5(uids: list, targets: dict, eligibility: dict) -> dict:
    """What one batch showed: the support train_unified_v5 logs per task, and the bags in the patches.

    {"support": {task: int}, "bags": [(uid, class index, bag id, supervised voxels inside the patch), ...]}"""
    support, found = {}, []
    for task in sorted(targets):
        e_pos = eligibility[task]["e_pos"].to(torch.bool)
        e_neg = eligibility[task]["e_neg"].to(torch.bool)
        if task != "U":
            support[task] = _support(task, targets[task], e_pos, e_neg)
            continue
        bags = targets[task]
        support[task] = _bag_count(bags, e_pos) + int(e_neg.sum())
        for b, uid in enumerate(uids):
            for c in range(bags.shape[1]):
                values = bags[b, c][e_pos[b, c] & (bags[b, c] > 0)]
                for bag_id in torch.unique(values).tolist():
                    found.append((uid, c, int(bag_id), int((values == bag_id).sum())))
    return {"support": support, "bags": found}
