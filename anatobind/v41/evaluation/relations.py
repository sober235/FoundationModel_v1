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
    denominator = len(truth_set) + lesions_without_reference
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "per_predicate": per_predicate,
        "lesions_without_reference": lesions_without_reference,
        "reference_coverage_denominator": denominator,
        "reference_coverage": (len(truth_set) / denominator) if denominator else None,
    }
