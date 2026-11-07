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
