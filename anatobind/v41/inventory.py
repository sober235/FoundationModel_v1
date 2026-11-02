"""Controlled asset discovery and incremental integrity checks.

Discovery never follows symbolic links that leave the raw root, never reads
credential-like files, and counts cases by configured case patterns rather
than by file count. Verification levels are recorded alongside results so a
header-only check is never reported as full validation.
"""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from anatobind.v41.paths import resolve_private_output, write_private_json

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "v41" / "sources.yaml"
VERIFY_LEVELS = ("metadata", "header", "full_decode")
QUARANTINE_SUFFIXES = (".truncated", ".partial", ".part")
IMAGE_SUFFIXES = (".nii", ".nii.gz", ".h5", ".dcm", ".mha", ".nrrd")
METADATA_SUFFIXES = (".json", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml", ".xlsx", ".log")


def load_sources_config(path: Path | None = None) -> dict:
    config = yaml.safe_load(Path(path or CONFIG_PATH).read_text())
    if not isinstance(config, dict) or "sources" not in config:
        raise ValueError("INVALID_SOURCES_CONFIG")
    return config


def _suffix(name: str) -> str:
    lower = name.lower()
    for compound in (".nii.gz", ".tar.gz", ".h5.truncated", ".nii.gz.truncated"):
        if lower.endswith(compound):
            return compound
    return PurePosixPath(lower).suffix


def _matches(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(rel, p) for p in patterns)


def _case_key(source: str, rel_in_source: str, patterns: list[str]) -> tuple[str | None, str | None]:
    parts = PurePosixPath(rel_in_source).parts
    for depth in range(1, len(parts) + 1):
        prefix = "/".join(parts[:depth])
        if _matches(prefix, patterns):
            leaf = parts[depth - 1]
            case_id = leaf[: -len(_suffix(leaf))] if _suffix(leaf) else leaf
            return f"{source}/{prefix}", case_id
    return None, None


def _classify(rel_in_source: str, suffix: str, spec: dict, archive_suffixes: list[str]) -> str:
    if suffix.endswith(QUARANTINE_SUFFIXES):
        return "quarantined"
    if suffix in archive_suffixes:
        return "archive"
    if _matches(rel_in_source, spec.get("label_patterns", [])):
        return "label"
    if _matches(rel_in_source, spec.get("image_patterns", [])):
        return "image"
    if suffix in IMAGE_SUFFIXES:
        return "image_unassigned"
    if suffix in METADATA_SUFFIXES:
        return "metadata"
    return "other"


def _asset_id(relpath: str) -> str:
    return hashlib.sha256(relpath.encode()).hexdigest()


def discover_assets(root: Path, source_config: dict) -> list[dict]:
    root = Path(root).resolve()
    cred_patterns = source_config.get("credential_like_patterns", [])
    archive_suffixes = source_config.get("archive_suffixes", [])
    assets: list[dict] = []
    for source, spec in source_config["sources"].items():
        source_dir = root / source
        exclusions = {"excluded_symlinks": 0, "excluded_credential_like": 0, "unreadable": 0}
        if source_dir.is_dir() and not source_dir.is_symlink():
            for current, dirnames, filenames in os.walk(source_dir, followlinks=False):
                current_path = Path(current)
                kept = []
                for name in sorted(dirnames):
                    child = current_path / name
                    if child.is_symlink():
                        try:
                            inside = child.resolve().is_relative_to(root)
                        except OSError:
                            inside = False
                        if not inside:
                            exclusions["excluded_symlinks"] += 1
                            continue
                    kept.append(name)
                dirnames[:] = kept
                for name in sorted(filenames):
                    child = current_path / name
                    if _matches(name, cred_patterns) or _matches(name.lower(), cred_patterns):
                        exclusions["excluded_credential_like"] += 1
                        continue
                    if child.is_symlink():
                        try:
                            if not child.resolve().is_relative_to(root):
                                exclusions["excluded_symlinks"] += 1
                                continue
                        except OSError:
                            exclusions["excluded_symlinks"] += 1
                            continue
                    try:
                        size = child.lstat().st_size
                    except OSError:
                        exclusions["unreadable"] += 1
                        continue
                    rel_in_source = child.relative_to(source_dir).as_posix()
                    suffix = _suffix(name)
                    kind = _classify(rel_in_source, suffix, spec, archive_suffixes)
                    case_key = case_id = None
                    if kind not in ("archive", "quarantined"):
                        case_key, case_id = _case_key(source, rel_in_source, spec.get("case_patterns", []))
                    relpath = f"{source}/{rel_in_source}"
                    assets.append({
                        "asset_id": _asset_id(relpath),
                        "relpath": relpath,
                        "source": source,
                        "parent_source": source,
                        "priority": spec.get("priority", "P2"),
                        "kind": kind,
                        "suffix": suffix,
                        "size_bytes": size,
                        "case_key": case_key,
                        "case_id": case_id,
                        "status": "observed",
                        "_root": str(root),
                    })
        assets.append({
            "asset_id": _asset_id(f"{source}/__source_summary__"),
            "relpath": f"{source}/__source_summary__",
            "source": source,
            "parent_source": source,
            "priority": spec.get("priority", "P2"),
            "kind": "source_summary",
            "suffix": "",
            "size_bytes": 0,
            "case_key": None,
            "case_id": None,
            "status": "observed",
            "source_present": source_dir.is_dir(),
            **exclusions,
        })
    return assets


def summarize_assets(assets: list[dict]) -> dict:
    sources: dict[str, dict[str, Any]] = {}
    for asset in assets:
        entry = sources.setdefault(asset["source"], {
            "priority": asset["priority"], "files": 0, "bytes": 0, "images": 0, "labels": 0,
            "archives": 0, "quarantined": 0, "metadata": 0, "other": 0, "image_unassigned": 0,
            "cases_observed": 0, "_cases": set(), "excluded_symlinks": 0,
            "excluded_credential_like": 0, "unreadable": 0, "source_present": False,
        })
        if asset["kind"] == "source_summary":
            for key in ("excluded_symlinks", "excluded_credential_like", "unreadable", "source_present"):
                entry[key] = asset[key]
            continue
        entry["files"] += 1
        entry["bytes"] += asset["size_bytes"]
        plural = {"image": "images", "label": "labels", "archive": "archives", "quarantined": "quarantined",
                  "metadata": "metadata", "other": "other", "image_unassigned": "image_unassigned"}[asset["kind"]]
        entry[plural] += 1
        if asset["case_id"] and asset["kind"] in ("image", "label"):
            entry["_cases"].add(asset["case_id"])
    for entry in sources.values():
        entry["cases_observed"] = len(entry.pop("_cases"))
    return {
        "note": "cases_observed counts distinct case ids (image or label tracks merged) within one source; it is not a patient count",
        "sources": sources,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _header_nifti(path: Path, full: bool) -> dict:
    import nibabel as nib

    image = nib.load(str(path))
    header = {
        "shape": [int(v) for v in image.shape],
        "dtype": str(image.get_data_dtype()),
        "spacing_mm": [float(v) for v in image.header.get_zooms()[:3]],
        "axis_codes": list(nib.aff2axcodes(image.affine)),
    }
    if full:
        data = image.dataobj[...]
        header["finite"] = bool(__import__("numpy").isfinite(data).all())
    return header


def _header_h5(path: Path, full: bool) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys())
        header = {"keys": keys, "shapes": {}}
        for key in keys:
            node = handle[key]
            if isinstance(node, h5py.Dataset):
                header["shapes"][key] = [int(v) for v in node.shape]
                if full:
                    node[()]
    return header


def verify_asset(asset: dict, level: str, root: Path | None = None) -> dict:
    if level not in VERIFY_LEVELS:
        raise ValueError("INVALID_VERIFY_LEVEL")
    report = {
        "asset_id": asset["asset_id"], "level": level, "status": "not_checked",
        "training_eligible": None, "reason": None, "size_bytes": None,
        "header": None, "content_sha256": None,
    }
    if asset["kind"] == "quarantined" or asset["suffix"].endswith(QUARANTINE_SUFFIXES):
        report.update(status="quarantined", training_eligible=False, reason="TRUNCATED_ASSET")
        return report
    base = root or asset.get("_root")
    if base is None:
        report.update(status="not_checked", reason="NO_ROOT")
        return report
    path = Path(base) / asset["relpath"]
    try:
        report["size_bytes"] = path.stat().st_size
    except OSError:
        report.update(status="failed", training_eligible=False, reason="MISSING_ASSET")
        return report
    if level == "metadata":
        report["status"] = "observed"
        return report
    full = level == "full_decode"
    try:
        if asset["suffix"] in (".nii", ".nii.gz"):
            report["header"] = _header_nifti(path, full)
        elif asset["suffix"] == ".h5":
            report["header"] = _header_h5(path, full)
        else:
            report.update(status="not_checked", reason="UNSUPPORTED_FORMAT")
            return report
    except Exception as error:  # noqa: BLE001 - error text may contain the path
        code = "H5_OPEN_ERROR" if asset["suffix"] == ".h5" and not full else "DECODE_ERROR"
        report.update(status="failed", training_eligible=False, reason=code)
        return report
    if full:
        report["content_sha256"] = _sha256_file(path)
        report.update(status="verified", training_eligible=True)
    else:
        report["status"] = "header_only"
    return report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("PRIVATE_FILE_EXISTS") from None
    with os.fdopen(fd, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def integrity_summary(reports: list[dict], level: str) -> dict:
    return {
        "level": level,
        "counts": {status: sum(1 for r in reports if r["status"] == status)
                   for status in ("observed", "header_only", "verified", "quarantined", "failed", "not_checked")},
        "training_eligible_true": sum(1 for r in reports if r["training_eligible"] is True),
    }


def write_inventory(root: Path, config: dict, out: Path, private_root: Path, repo: Path) -> dict:
    out = resolve_private_output(out, private_root, repo)
    assets = discover_assets(root, config)
    public = [{k: v for k, v in a.items() if k != "_root"} for a in assets]
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_jsonl(out / "assets.jsonl", public)
    summary = summarize_assets(assets)
    write_private_json(out / "inventory_summary.json", summary)
    return summary


def verify_inventory(root: Path, assets_path: Path, level: str, out: Path, private_root: Path, repo: Path,
                     priorities: tuple[str, ...] = ("P0",)) -> dict:
    out = resolve_private_output(out, private_root, repo)
    assets = [a for a in read_jsonl(assets_path) if a["kind"] not in ("source_summary",)]
    chosen = [a for a in assets if a["priority"] in priorities]
    previous_path = out / "asset_integrity.jsonl"
    previous = {r["asset_id"]: r for r in read_jsonl(previous_path)} if previous_path.exists() else {}
    rank = {name: i for i, name in enumerate(VERIFY_LEVELS)}
    reports = []
    for asset in chosen:
        old = previous.get(asset["asset_id"])
        if old and rank[old["level"]] >= rank[level] and old["status"] in ("verified", "header_only", "quarantined"):
            reports.append(old)
            continue
        reports.append(verify_asset(asset, level, root=Path(root)))
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    if previous_path.exists():
        previous_path.rename(out / f"asset_integrity.previous.{time.time_ns()}.jsonl")
    _write_jsonl(previous_path, reports)
    summary = integrity_summary(reports, level)
    summary["assets_selected"] = len(chosen)
    summary["priorities"] = list(priorities)
    write_private_json(out / f"integrity_summary_{level}_{time.time_ns()}.json", summary)
    return summary


def _cli_inventory(args) -> int:
    from anatobind.v41.cli import package_repo_root

    config = load_sources_config(args.config)
    summary = write_inventory(Path(args.root), config, Path(args.out), Path(args.private_root), package_repo_root())
    print(json.dumps({"sources": len(summary["sources"]),
                      "cases_observed": {k: v["cases_observed"] for k, v in summary["sources"].items()}}))
    return 0


def _cli_verify(args) -> int:
    from anatobind.v41.cli import package_repo_root

    priorities = tuple(dict.fromkeys(args.priority or ["P0"]))
    summary = verify_inventory(Path(args.root), Path(args.assets), args.level, Path(args.out),
                               Path(args.private_root), package_repo_root(), priorities)
    print(json.dumps(summary))
    return 0


def register_cli(subparsers) -> None:
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--config", default=str(CONFIG_PATH))
    inventory.add_argument("--root", required=True)
    inventory.add_argument("--out", required=True)
    inventory.add_argument("--private-root", required=True)
    inventory.set_defaults(func=_cli_inventory)
    verify = subparsers.add_parser("verify-assets")
    verify.add_argument("--root", required=True)
    verify.add_argument("--assets", required=True)
    verify.add_argument("--level", choices=VERIFY_LEVELS, default="header")
    verify.add_argument("--priority", action="append", default=None)
    verify.add_argument("--out", required=True)
    verify.add_argument("--private-root", required=True)
    verify.set_defaults(func=_cli_verify)
