"""Task-scoped metric policies (Proposal_v5 §8.3, §9).

Only the metrics a policy lists are checked; a null value blocks the metric
that needs it, never unrelated metrics. A locked evaluation refuses any block.
"""
from __future__ import annotations

from ..contracts_v5 import V5Error

METRIC_REQUIREMENTS = {
    "dice": (),
    "hd95": (),
    "nsd": ("surface_tolerance_mm",),
    "lesion_detection": ("iou_match_threshold",),
    "fp_constraint": ("fp_budget_per_scan",),
}
STAGES = ("development", "locked")


def validate_metric_policy_v5(policy: dict, stage: str) -> dict:
    if stage not in STAGES:
        raise V5Error("UNKNOWN_STAGE", "$.stage")
    metrics = policy.get("metrics")
    if metrics is None:
        if stage == "locked":
            raise V5Error("METRIC_POLICY_NULL", "$.metrics")
        return {"ok_metrics": [], "blocked_metrics": []}
    if not isinstance(metrics, list):
        raise V5Error("INVALID_TYPE", "$.metrics")
    ok, blocked = [], []
    for name in metrics:
        if name not in METRIC_REQUIREMENTS:
            raise V5Error("UNKNOWN_METRIC", "$.metrics")
        if any(policy.get(key) is None for key in METRIC_REQUIREMENTS[name]):
            blocked.append(name)
        else:
            ok.append(name)
    if stage == "locked" and blocked:
        raise V5Error("METRIC_POLICY_NULL", f"$.metrics.{blocked[0]}")
    return {"ok_metrics": ok, "blocked_metrics": blocked}
