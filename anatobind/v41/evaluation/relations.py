"""Relation set metrics. Lesions without a relation reference lower coverage, never raise F1."""
from __future__ import annotations


def relation_scores(predicted: list[tuple], truth: list[tuple], lesions_without_reference: int = 0) -> dict:
    predicted_set, truth_set = set(predicted), set(truth)
    per_predicate = {}
    for _, predicate, *_ in sorted(predicted_set | truth_set, key=str):
        tp_items = {x for x in predicted_set & truth_set if x[1] == predicate}
        p_items = {x for x in predicted_set if x[1] == predicate}
        t_items = {x for x in truth_set if x[1] == predicate}
        precision = len(tp_items) / len(p_items) if p_items else 0.0
        recall = len(tp_items) / len(t_items) if t_items else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_predicate[predicate] = {"precision": precision, "recall": recall, "f1": f1,
                                    "tp": len(tp_items), "pred": len(p_items), "truth": len(t_items)}
    tp = len(predicted_set & truth_set)
    precision = tp / len(predicted_set) if predicted_set else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0
    lesions_with_reference = len({x[0] for x in truth_set})
    denominator = lesions_with_reference + lesions_without_reference
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "per_predicate": per_predicate,
        "lesions_without_reference": lesions_without_reference,
        "reference_coverage_denominator": denominator,
        "lesions_with_reference": lesions_with_reference,
        "reference_coverage": (lesions_with_reference / denominator) if denominator else None,
    }


# ---------------------------------------------------------------------------
# v5.1 relation scores on a frozen reference set (Proposal_v5 §8.3). v41 relation_scores above is unchanged.

from ..contracts_v5 import V5Error  # noqa: E402
from .reference import verify_reference_set_v5  # noqa: E402


def _ratio_v5(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def relation_scores_v5(frozen: dict, predicted_sets: dict) -> dict:
    """Exact tissue-set match and per-tissue F1 over R-evaluable reference lesions.

    predicted_sets[uid] is None (or absent) for "no prediction", which never matches;
    [] is a predicted empty tissue set and matches an empty truth set.
    """
    lesions = verify_reference_set_v5(frozen)
    for uid, predicted in predicted_sets.items():
        if uid not in lesions:
            raise V5Error("UNKNOWN_REFERENCE_INSTANCE", "$.predicted_sets")
        if predicted is not None and (not isinstance(predicted, (list, tuple))
                                      or not all(isinstance(t, str) for t in predicted)):
            raise V5Error("INVALID_PREDICTED_SET", "$.predicted_sets")
    evaluable = [lesion for lesion in frozen["lesions"]
                 if lesion["r_reference_available"] and lesion["eligible"] and lesion["visibility_qualified"]]
    matches = n_empty = n_missing = 0
    counts = {}
    for lesion in evaluable:
        truth = set(lesion["truth_relation_set"])
        predicted = predicted_sets.get(lesion["instance_uid"])
        n_empty += not truth
        if predicted is None:
            n_missing += 1
            predicted = set()
        else:
            predicted = set(predicted)
            matches += predicted == truth
        for tissue in truth | predicted:
            row = counts.setdefault(tissue, {"tp": 0, "fp": 0, "fn": 0})
            row["tp" if tissue in truth and tissue in predicted else "fn" if tissue in truth else "fp"] += 1
    per_tissue = {}
    for tissue in sorted(counts):
        tp, fp, fn = counts[tissue]["tp"], counts[tissue]["fp"], counts[tissue]["fn"]
        precision, recall = _ratio_v5(tp, tp + fp), _ratio_v5(tp, tp + fn)
        # A listed tissue always has tp + fp + fn > 0; a never-predicted tissue is F1 0, not NA.
        f1 = _ratio_v5(2 * tp, 2 * tp + fp + fn)
        per_tissue[tissue] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    return {
        "exact_match_rate": _ratio_v5(matches, len(evaluable)),
        "n_with_reference": len(evaluable),
        "n_empty_truth_sets": n_empty,
        "n_missing_prediction": n_missing,
        "per_tissue": per_tissue,
    }
