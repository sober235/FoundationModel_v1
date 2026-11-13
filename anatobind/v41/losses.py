"""Masked training objectives. Ineligible positions produce no gradient and
tasks without support never dilute the aggregated mean.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def masked_bce(logits: torch.Tensor, target: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    eligible = eligible.to(dtype=torch.bool)
    while eligible.ndim < logits.ndim:
        eligible = eligible.unsqueeze(0)
    eligible = eligible.expand_as(logits)
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none")
    return losses[eligible].mean()


def masked_dice_loss(logits: torch.Tensor, target: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    eligible = eligible.to(dtype=torch.bool)
    while eligible.ndim < logits.ndim:
        eligible = eligible.unsqueeze(0)
    eligible = eligible.expand_as(logits)
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits)[eligible]
    truth = target.to(logits.dtype)[eligible]
    intersection = (probs * truth).sum()
    return 1.0 - (2.0 * intersection + EPS) / (probs.sum() + truth.sum() + EPS)


def aggregate_task_losses(losses: dict[str, torch.Tensor], supports: dict[str, int],
                          weights: dict[str, float]) -> torch.Tensor:
    terms = []
    total_weight = 0.0
    for task, loss in losses.items():
        if supports.get(task, 0) <= 0:
            continue
        weight = float(weights.get(task, 1.0))
        terms.append(weight * loss)
        total_weight += weight
    if not terms:
        any_loss = next(iter(losses.values()), None)
        if any_loss is None:
            return torch.zeros(())
        return any_loss.sum() * 0.0
    return sum(terms) / total_weight


# ---------------------------------------------------------------------------
# v5.1 signed masked objective (Proposal_v5 §4.3). v41 functions above are unchanged.

import math  # noqa: E402
from fractions import Fraction  # noqa: E402

from .contracts_v5 import V5Error  # noqa: E402

LOSS_KINDS_V5 = ("sigmoid", "softmax")


def _check_weights_v5(task_spec: dict) -> tuple[float, float]:
    weights = (task_spec.get("ce_weight"), task_spec.get("dice_weight"))
    valid = all(isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w) and w >= 0
                for w in weights)
    if not valid or sum(weights) <= 0:
        raise V5Error("INVALID_LOSS_WEIGHTS", "$.task_spec")
    return float(weights[0]), float(weights[1])


def _soft_dice_v5(probs: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    """Mean over channels of 1 - Dice; probs/onehot are [N, channels] over supervised positions only."""
    intersection = (probs * onehot).sum(dim=0)
    denominator = probs.sum(dim=0) + onehot.sum(dim=0)
    return (1.0 - (2.0 * intersection + EPS) / (denominator + EPS)).mean()


def masked_objective_v5(logits: torch.Tensor, target: torch.Tensor, e_pos: torch.Tensor, e_neg: torch.Tensor,
                        task_spec: dict) -> torch.Tensor:
    kind = task_spec.get("kind")
    if kind not in LOSS_KINDS_V5:
        raise V5Error("UNKNOWN_LOSS_KIND", "$.task_spec.kind")
    ce_weight, dice_weight = _check_weights_v5(task_spec)
    e_pos, e_neg = e_pos.to(torch.bool), e_neg.to(torch.bool)
    if kind == "sigmoid":
        if not (target.shape == logits.shape == e_pos.shape == e_neg.shape):
            raise V5Error("GRID_MISMATCH", "$.logits")
        mask = ((target == 1) & e_pos) | ((target == 0) & e_neg)
        full_mask = mask
    else:
        batch, spatial, n_classes = logits.shape[0], tuple(logits.shape[2:]), logits.shape[1] - 1
        if tuple(target.shape) != (batch, *spatial) or e_pos.shape != e_neg.shape \
                or tuple(e_pos.shape) != (batch, n_classes, *spatial):
            raise V5Error("GRID_MISMATCH", "$.logits")
        mask = e_pos.all(dim=1) & e_neg.all(dim=1)
        full_mask = mask.unsqueeze(1).expand_as(logits)
    if not bool(mask.any()):
        return logits.masked_fill(~full_mask, 0).sum() * 0.0
    if kind == "sigmoid":
        selected = logits[mask]
        if not bool(torch.isfinite(selected).all()):
            raise V5Error("NON_FINITE_LOGITS", "$.logits")
        # Proposal §4.3: normalize per class over its own support, then average classes with support.
        per_class_ce, per_class_dice = [], []
        for c in range(logits.shape[1]):
            class_mask = mask[:, c]
            if bool(class_mask.any()):
                class_logits = logits[:, c][class_mask]
                class_labels = target[:, c][class_mask].to(logits.dtype)
                per_class_ce.append(F.binary_cross_entropy_with_logits(class_logits, class_labels))
                per_class_dice.append(_soft_dice_v5(torch.sigmoid(class_logits).unsqueeze(1), class_labels.unsqueeze(1)))
        ce = torch.stack(per_class_ce).mean()
        dice = torch.stack(per_class_dice).mean()
    else:
        selected = logits.movedim(1, -1)[mask]
        if not bool(torch.isfinite(selected).all()):
            raise V5Error("NON_FINITE_LOGITS", "$.logits")
        labels = target[mask].long()
        ce = F.cross_entropy(selected, labels)
        probs = torch.softmax(selected, dim=1)
        onehot = F.one_hot(labels, num_classes=logits.shape[1]).to(logits.dtype)
        dice = _soft_dice_v5(probs[:, 1:], onehot[:, 1:])
    return ce_weight * ce + dice_weight * dice


# ---------------------------------------------------------------------------
# Box multiple-instance objective (Proposal_v5 §0.6, §5.0; design 2026-11-10 §5.2)

BOX_SPEC_KEYS = ("kind", "topk_fraction", "positive_weight", "negative_weight", "box_policy_sha256")
BOX_SPEC_OPTIONAL_KEYS = ("negative_topk_fraction",)        # absent: every eligible negative voxel is kept


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fraction(value: object) -> bool:
    return _number(value) and 0 < value <= 1


def negative_top_count_v5(fraction: float, eligible: int) -> int:
    """ceil(fraction · eligible), the fraction read as the decimal the plan writes (0.01 · 700 is 7, not 8)."""
    return math.ceil(Fraction(str(fraction)) * eligible)


def box_mil_terms_v5(logits: torch.Tensor, bags: torch.Tensor, e_pos: torch.Tensor, e_neg: torch.Tensor,
                     spec: dict) -> dict:
    """The terms box_mil_objective_v5 adds, as tensors, with their supports. Training, replay and probes all read
    the objective through this one function.

    positive      mean over bags of the mean softplus(-logit) of each bag's top ceil(topk_fraction·n) logits;
    negative      the term that is optimised: mean over the classes with support of each class's mean
                  softplus(logit) over the voxels it keeps. A class keeps all N of its e_neg voxels, or with
                  `negative_topk_fraction` the ceil(fraction·N) highest of them, pooled over the whole batch;
    negative_all  the same average with every class keeping all of its e_neg voxels (no gradient: it is recorded,
                  not optimised, unless every class keeps everything, where it is `negative` itself);
    classes       per class: bags, its positive mean, eligible and selected negative voxels, its two negative means.
    A term without support is None. Only e_neg says what a negative is; it is never rebuilt from the bags."""
    if not isinstance(spec, dict) or spec.get("kind") != "box_mil":
        raise V5Error("UNKNOWN_LOSS_KIND", "$.spec.kind")
    fraction, positive_weight, negative_weight = (spec.get(key) for key in BOX_SPEC_KEYS[1:4])
    if not (_fraction(fraction) and _number(positive_weight) and _number(negative_weight)
            and positive_weight >= 0 and negative_weight >= 0 and positive_weight + negative_weight > 0):
        raise V5Error("INVALID_LOSS_WEIGHTS", "$.spec")
    keep = spec["negative_topk_fraction"] if "negative_topk_fraction" in spec else None
    if "negative_topk_fraction" in spec and not _fraction(keep):
        raise V5Error("INVALID_LOSS_WEIGHTS", "$.spec")
    if not (logits.shape == bags.shape == e_pos.shape == e_neg.shape) or logits.ndim < 3:
        raise V5Error("GRID_MISMATCH", "$.logits")
    e_pos, e_neg = e_pos.to(torch.bool), e_neg.to(torch.bool)
    positive, negative, negative_all, classes = [], [], [], []
    for c in range(logits.shape[1]):
        class_logits, class_negative = logits[:, c], e_neg[:, c]
        entry = {"bags": 0, "positive": None, "negative": None, "negative_all": None, "negative_eligible": 0,
                 "negative_selected": 0}
        if bool(class_negative.any()):
            selected = class_logits[class_negative]
            if not bool(torch.isfinite(selected).all()):
                raise V5Error("NON_FINITE_LOGITS", "$.logits")
            eligible = int(selected.numel())
            kept = eligible if keep is None else negative_top_count_v5(keep, eligible)
            if kept < eligible:
                with torch.no_grad():
                    everything = F.softplus(selected).mean()
                term = F.softplus(torch.topk(selected, kept, sorted=False).values).mean()
            else:
                term = F.softplus(selected).mean()
                everything = term.detach()
            entry.update(negative=term, negative_all=everything, negative_eligible=eligible, negative_selected=kept)
            negative.append(term)
            negative_all.append(everything)
        of_class = []
        for b in range(logits.shape[0]):
            supported = bags[b, c][e_pos[b, c]]
            for bag_id in torch.unique(supported[supported > 0]).tolist():
                selected = logits[b, c][(bags[b, c] == bag_id) & e_pos[b, c]]
                if not bool(torch.isfinite(selected).all()):
                    raise V5Error("NON_FINITE_LOGITS", "$.logits")
                top = torch.topk(selected, max(1, math.ceil(fraction * selected.numel()))).values
                of_class.append(F.softplus(-top).mean())
        if of_class:
            entry.update(bags=len(of_class), positive=torch.stack(of_class).mean())
        positive += of_class
        classes.append(entry)
    return {"positive": torch.stack(positive).mean() if positive else None,
            "negative": torch.stack(negative).mean() if negative else None,
            "negative_all": torch.stack(negative_all).mean() if negative_all else None,
            "bags": len(positive), "classes": classes}


def box_mil_total_v5(terms: dict, spec: dict, logits: torch.Tensor) -> torch.Tensor:
    """positive_weight·positive + negative_weight·negative of box_mil_terms_v5; a term without support adds zero
    and without any support the result is a differentiable zero."""
    if terms["positive"] is None and terms["negative"] is None:
        return logits.masked_fill(torch.ones_like(logits, dtype=torch.bool), 0).sum() * 0.0
    total = logits.new_zeros(())
    if terms["positive"] is not None:
        total = total + float(spec["positive_weight"]) * terms["positive"]
    if terms["negative"] is not None:
        total = total + float(spec["negative_weight"]) * terms["negative"]
    return total


def box_terms_record_v5(terms: dict) -> dict:
    """box_mil_terms_v5 as plain numbers (None without support): what a training log line and a replay keep."""
    def value(term):
        return None if term is None else float(term.detach())

    return {"positive": value(terms["positive"]), "negative": value(terms["negative"]),
            "negative_all": value(terms["negative_all"]), "bags": terms["bags"],
            "classes": [{**entry, **{key: value(entry[key]) for key in ("positive", "negative", "negative_all")}}
                        for entry in terms["classes"]]}


def box_mil_objective_v5(logits: torch.Tensor, bags: torch.Tensor, e_pos: torch.Tensor, e_neg: torch.Tensor,
                         spec: dict) -> torch.Tensor:
    """Box-level supervision without filling boxes into voxel truth.

    Positive part: for each bag (a released box on its host tissue, id > 0 in `bags`) the mean of softplus(-logit)
    over its top ceil(topk_fraction·n) logits, so a box only has to contain *some* marked tissue; averaged over
    bags. Negative part: mean softplus(logit) over each class's e_neg voxels, averaged over classes with support;
    with `negative_topk_fraction` each class keeps only its ceil(fraction·N) highest e_neg voxels of the batch.
    A part without support adds zero; without any support the result is a differentiable zero.
    """
    return box_mil_total_v5(box_mil_terms_v5(logits, bags, e_pos, e_neg, spec), spec, logits)
