"""Audit manifest publication and versioned gate decisions.

A manifest entry is admitted only when its references are qualified and its
source has every required audit. Gates combine audit status with human
sign-offs; the Agent records decisions, it never fills approvals.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from anatobind.v41.paths import write_private_json
from anatobind.v41.schema import GateDecision, SampleRecord, SCHEMA_VERSION

REQUIRED_AUDITS = (
    "ontology_validation_report",
    "patient_alias_report",
    "split_leakage_report",
    "geometry_qc",
    "annotation_provenance_report",
    "eligibility_coverage",
    "image_label_pairing",
    "input_visibility_report",
)
QUALIFIED_PROVENANCE = ("manual", "expert_corrected")
AUDIT_STATUSES = ("pass", "fail", "missing")
EXPERT_REPRESENTATIONS = ("voxel", "surface")
EXPERT_COMPLETENESS = ("target_complete", "region_complete")
GATES = ("G0-U", "G0-A", "G0-joint")
GATE_SIGNOFFS = {
    "G0-U": ("target_abnormality_definition",),
    "G0-A": ("target_abnormality_definition",),
    "G0-joint": ("target_abnormality_definition",),
}


class ManifestError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def check_reference_qualification(record: SampleRecord) -> None:
    for annotation in record.annotation_records:
        if annotation.reference_role != "expert_voxel":
            continue
        if (annotation.provenance_type not in QUALIFIED_PROVENANCE
                or annotation.representation not in EXPERT_REPRESENTATIONS
                or annotation.quality_status != "accepted"
                or annotation.completeness not in EXPERT_COMPLETENESS):
            raise ManifestError("UNQUALIFIED_REFERENCE")


def _check_status(status) -> str:
    if status not in AUDIT_STATUSES:
        raise ManifestError("INVALID_AUDIT_STATUS")
    return status


def _audit_index(audit_reports: list[dict], source: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for report in audit_reports:
        status = _check_status(report.get("status", "missing"))
        if report.get("scope") not in (source, "all"):
            continue
        previous = index.get(report["name"])
        if previous is None or (previous != "fail" and status == "fail") or (previous == "missing" and status != "missing"):
            index[report["name"]] = status
    return index


def publish_manifest(records: list[SampleRecord], audit_reports: list[dict], out: Path) -> dict:
    out = Path(out)
    seen = set()
    for record in records:
        if record.sample_uid in seen:
            raise ManifestError("DUPLICATE_SAMPLE_UID")
        seen.add(record.sample_uid)
    entries = []
    for record in records:
        check_reference_qualification(record)
        audits = _audit_index(audit_reports, record.source_dataset)
        reasons = []
        missing = [name for name in REQUIRED_AUDITS if audits.get(name, "missing") == "missing"]
        failed = [name for name in REQUIRED_AUDITS if audits.get(name) == "fail"]
        if missing:
            reasons.append("MISSING_AUDIT")
        if failed:
            reasons.append("FAILED_AUDIT")
        if record.split == "locked_test" and any(a.reference_role == "unverified" for a in record.annotation_records):
            reasons.append("UNVERIFIED_REFERENCE_IN_TEST")
        entries.append({
            "sample_uid": record.sample_uid,
            "source_dataset": record.source_dataset,
            "split": record.split,
            "status": "blocked" if reasons else "admitted",
            "reasons": reasons,
            "missing_audits": missing,
            "failed_audits": failed,
            "roles": {a.annotation_uid: a.reference_role for a in record.annotation_records},
        })
    sources = {}
    for entry in entries:
        source = sources.setdefault(entry["source_dataset"], {"admitted": 0, "blocked": 0})
        source[entry["status"]] += 1
    for source, counts in sources.items():
        counts["status"] = "admitted" if counts["blocked"] == 0 else "blocked"
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    try:
        fd = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ManifestError("PRIVATE_FILE_EXISTS") from None
    digest = hashlib.sha256()
    with os.fdopen(fd, "w") as handle:
        for entry in entries:
            line = json.dumps(entry, sort_keys=True) + "\n"
            digest.update(line.encode())
            handle.write(line)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": digest.hexdigest(),
        "entries": len(entries),
        "admitted": sum(1 for e in entries if e["status"] == "admitted"),
        "blocked": sum(1 for e in entries if e["status"] == "blocked"),
        "audit_reports": {f'{a["name"]}@{a["scope"]}': {"status": a.get("status"), "sha256": a.get("sha256")}
                          for a in audit_reports},
        "required_audits": list(REQUIRED_AUDITS),
    }
    write_private_json(out / "manifest.lock.json", lock)
    return {**lock, "entries": entries, "sources": sources,
            "manifest_sha256": lock["manifest_sha256"]}


def evaluate_gate(gate_id: str, reports: dict[str, str] | list[dict], approvals: dict,
                  joint_evidence: dict | None = None, source: str | None = None) -> GateDecision:
    if gate_id not in GATES:
        raise ManifestError("UNKNOWN_GATE")
    if isinstance(reports, list):
        if source is None:
            raise ManifestError("GATE_SOURCE_REQUIRED")
        evidence = sorted({e["sha256"] for e in reports
                           if e.get("scope") in (source, "all") and e.get("sha256")})
        reports = _audit_index(reports, source)
    else:
        for status in reports.values():
            _check_status(status)
        evidence = [hashlib.sha256(json.dumps(reports, sort_keys=True).encode()).hexdigest()]
    blockers = []
    failed = [name for name in REQUIRED_AUDITS if reports.get(name) == "fail"]
    missing = [name for name in REQUIRED_AUDITS if reports.get(name, "missing") == "missing"]
    for field in GATE_SIGNOFFS[gate_id]:
        approval = approvals.get(field)
        if not approval or not approval.get("approved_by") or not approval.get("approved_at"):
            blockers.append(f"missing sign-off: {field}")
    if gate_id == "G0-joint":
        evidence = joint_evidence or {}
        if not evidence.get("same_case_expert_A_and_U"):
            blockers.append("no same-case expert A and U reference cohort")
    if failed:
        engineering = "FAIL"
    elif missing or blockers:
        engineering = "BLOCKED"
    else:
        engineering = "PASS"
    if missing:
        blockers = [f"missing audits: {len(missing)}"] + blockers
    if failed:
        blockers = [f"failed audits: {', '.join(failed)}"] + blockers
    approvers = sorted({a["approved_by"] for a in approvals.values()
                        if isinstance(a, dict) and a.get("approved_by")})
    return GateDecision(
        schema_version=SCHEMA_VERSION,
        gate_id=gate_id,
        scope_hash=hashlib.sha256(json.dumps(reports, sort_keys=True).encode()).hexdigest(),
        engineering_status=engineering,
        scientific_conclusion="NOT_EVALUATED",
        evidence_hashes=evidence,
        blockers=blockers,
        approved_by=", ".join(approvers) if approvers else None,
        approved_at=None if not approvers else max(a["approved_at"] for a in approvals.values()
                                                   if isinstance(a, dict) and a.get("approved_at")),
    )

def assemble_audit_index(entries: list[dict]) -> list[dict]:
    assembled = []
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_file():
            raise ManifestError("AUDIT_FILE_MISSING")
        if not entry.get("scope"):
            raise ManifestError("AUDIT_SCOPE_MISSING")
        assembled.append({
            "name": entry["name"],
            "scope": entry["scope"],
            "status": _check_status(entry.get("status")),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    names = [(e["name"], e["scope"]) for e in assembled]
    if len(set(names)) != len(names):
        raise ManifestError("DUPLICATE_AUDIT_NAME")
    return assembled


def _load_records(paths: list[str]) -> list[SampleRecord]:
    from anatobind.v41.schema import validate_sample

    records = []
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    records.append(validate_sample(json.loads(line)))
    return records


def _cli_audit(args) -> int:
    from anatobind.v41.audits import write_scope_audit
    from anatobind.v41.cli import package_repo_root
    from anatobind.v41.inventory import read_jsonl
    from anatobind.v41.paths import resolve_private_output

    out = resolve_private_output(Path(args.out), Path(args.private_root), package_repo_root())
    summary = write_scope_audit(args.scope, read_jsonl(Path(args.assets)), out,
                                Path(args.annotations_dir) if args.annotations_dir else None)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _cli_build_manifest(args) -> int:
    from anatobind.v41.cli import package_repo_root
    from anatobind.v41.paths import resolve_private_output

    out = resolve_private_output(Path(args.out), Path(args.private_root), package_repo_root())
    records = _load_records(args.records)
    audits = assemble_audit_index(json.loads(Path(args.audit_index).read_text()))
    result = publish_manifest(records, audits, out)
    print(json.dumps({"entries": result["entries"] and len(result["entries"]),
                      "admitted": result["admitted"], "blocked": result["blocked"],
                      "sources": result["sources"], "manifest_sha256": result["manifest_sha256"]}))
    return 0


def _cli_gate(args) -> int:
    from anatobind.v41.cli import package_repo_root
    from anatobind.v41.paths import resolve_private_output, write_private_json

    out = resolve_private_output(Path(args.out), Path(args.private_root), package_repo_root())
    if args.audit_index:
        reports = assemble_audit_index(json.loads(Path(args.audit_index).read_text()))
    elif args.reports:
        reports = json.loads(Path(args.reports).read_text())
    else:
        raise ValueError("GATE_INPUT_REQUIRED")
    approvals = json.loads(Path(args.approvals).read_text()) if args.approvals else {}
    joint = json.loads(Path(args.joint_evidence).read_text()) if args.joint_evidence else None
    decision = evaluate_gate(args.gate, reports, approvals, joint_evidence=joint, source=args.source)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = f"{args.gate}.{args.source}.json" if args.source else f"{args.gate}.json"
    write_private_json(out / name, dict(decision.__dict__))
    print(json.dumps({"gate_id": decision.gate_id, "engineering_status": decision.engineering_status,
                      "scientific_conclusion": decision.scientific_conclusion,
                      "blockers": len(decision.blockers)}))
    return 0


def register_cli(subparsers) -> None:
    audit = subparsers.add_parser("audit")
    audit.add_argument("--scope", required=True, choices=("isles_u", "skm_a"))
    audit.add_argument("--assets", required=True)
    audit.add_argument("--annotations-dir")
    audit.add_argument("--out", required=True)
    audit.add_argument("--private-root", required=True)
    audit.set_defaults(func=_cli_audit)
    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--records", action="append", required=True)
    manifest.add_argument("--audit-index", required=True)
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--private-root", required=True)
    manifest.set_defaults(func=_cli_build_manifest)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--gate", required=True, choices=list(GATES))
    gate.add_argument("--reports")
    gate.add_argument("--audit-index")
    gate.add_argument("--source")
    gate.add_argument("--approvals")
    gate.add_argument("--joint-evidence")
    gate.add_argument("--out", required=True)
    gate.add_argument("--private-root", required=True)
    gate.set_defaults(func=_cli_gate)
