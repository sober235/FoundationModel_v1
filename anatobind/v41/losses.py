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
