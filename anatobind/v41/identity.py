"""Patient identity, derivative lineage and alias-based leakage groups.

Groups are connected components over sample records: samples sharing a
patient_uid or a parent scan join, and alias edges join patients only when
they carry written evidence. Unresolved edges merge conservatively and are
disclosed, never silently split.
"""
from __future__ import annotations

from collections.abc import Iterable

from anatobind.v41.schema import SampleRecord

ALIAS_KINDS = ("same_patient", "same_scan", "followup", "cross_source")
ALIAS_STATUSES = ("confirmed", "unresolved")


class AliasError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def validate_alias(edge: dict) -> dict:
    if not isinstance(edge, dict) or not edge.get("a") or not edge.get("b"):
        raise AliasError("ALIAS_MISSING_ENDPOINT")
    if edge.get("kind") not in ALIAS_KINDS:
        raise AliasError("ALIAS_UNKNOWN_KIND")
    if not str(edge.get("evidence") or "").strip():
        raise AliasError("ALIAS_WITHOUT_EVIDENCE")
    status = edge.get("status", "confirmed")
    if status not in ALIAS_STATUSES:
        raise AliasError("ALIAS_UNKNOWN_STATUS")
    return {**edge, "status": status}


def resolve_groups(records: list[SampleRecord], aliases: list[dict]) -> dict[str, str]:
    edges = [validate_alias(e) for e in aliases]
    uf = _UnionFind()
    for record in records:
        patient = f"patient:{record.patient_uid}"
        uf.union(f"sample:{record.sample_uid}", patient)
        if record.parent_scan_uid:
            uf.union(patient, f"scan:{record.parent_scan_uid}")
    for edge in edges:
        uf.union(f"patient:{edge['a']}", f"patient:{edge['b']}")
    members: dict[str, list[str]] = {}
    for record in records:
        members.setdefault(uf.find(f"sample:{record.sample_uid}"), []).append(record.patient_uid)
    names = {root: f"group:{min(patients)}" for root, patients in members.items()}
    return {record.sample_uid: names[uf.find(f"sample:{record.sample_uid}")] for record in records}


def alias_report_rows(records: list[SampleRecord], aliases: list[dict], groups: dict[str, str]) -> list[dict]:
    patient_group = {r.patient_uid: groups[r.sample_uid] for r in records}
    rows = []
    for edge in (validate_alias(e) for e in aliases):
        rows.append({
            "a": edge["a"], "b": edge["b"], "kind": edge["kind"], "status": edge["status"],
            "evidence": edge["evidence"],
            "group": patient_group.get(edge["a"]) or patient_group.get(edge["b"]),
            "a_in_records": edge["a"] in patient_group, "b_in_records": edge["b"] in patient_group,
        })
    return rows


def group_members(records: Iterable[SampleRecord], groups: dict[str, str]) -> dict[str, list[SampleRecord]]:
    out: dict[str, list[SampleRecord]] = {}
    for record in records:
        out.setdefault(groups[record.sample_uid], []).append(record)
    return out
