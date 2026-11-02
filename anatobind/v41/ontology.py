"""Explicit A/U/R ontology access. Unknown integers are errors, not background."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ONTOLOGY_PATH = Path(__file__).resolve().parents[2] / "configs" / "v41" / "ontology.yaml"
SCOPE_STATUSES = ("in_scope", "in_scope_laterality_unknown", "out_of_anatomy_scope", "no_host", "unknown")
LATERALITIES = ("none", "medial", "lateral")

_CACHE: dict | None = None


class OntologyError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def load_ontology(path: Path | None = None) -> dict:
    global _CACHE
    if path is not None:
        return yaml.safe_load(Path(path).read_text())
    if _CACHE is None:
        _CACHE = yaml.safe_load(ONTOLOGY_PATH.read_text())
    return copy.deepcopy(_CACHE)


def _a6(config: dict | None = None) -> dict:
    return (config or load_ontology())["anatomy"]["skm_a6_v1"]


def map_skm_a_to_semantic(label: int, config: dict | None = None) -> dict:
    table = _a6(config)
    if isinstance(label, bool) or not isinstance(label, int):
        raise OntologyError("UNKNOWN_A_LABEL")
    if label == table["background"]:
        return {"name": "background", "host": "background", "laterality": "none"}
    entry = table["classes"].get(str(label))
    if entry is None:
        raise OntologyError("UNKNOWN_A_LABEL")
    return dict(entry)


def map_skm_a_to_host(label: int) -> str:
    return map_skm_a_to_semantic(label)["host"]


def resolve_relation_scope(tissue_id: int, config: dict | None = None) -> dict:
    table = (config or load_ontology())["skm_tissue_table_v1"]
    entry = table.get(str(tissue_id)) if not isinstance(tissue_id, bool) else None
    if not isinstance(entry, dict):
        return {"status": "unknown", "a_classes": [], "tissue_name": None, "tissue_id": tissue_id}
    return {"status": entry["status"], "a_classes": list(entry["a_classes"]),
            "tissue_name": entry["name"], "tissue_id": tissue_id}


def validate_ontology(config: dict) -> dict:
    errors: list[str] = []
    a6 = config["anatomy"]["skm_a6_v1"]
    classes = a6["classes"]
    names = [c["name"] for c in classes.values()]
    if len(set(names)) != len(names):
        errors.append("DUPLICATE_CLASS_NAME")
    if sorted(int(k) for k in classes) != [1, 2, 3, 4, 5, 6]:
        errors.append("A6_LABEL_SET")
    for entry in classes.values():
        if entry.get("laterality") not in LATERALITIES:
            errors.append("INVALID_LATERALITY")
            break
    hosts = {c["host"] for c in classes.values()}
    tissue = config["skm_tissue_table_v1"]
    for key, entry in tissue.items():
        if key == "description":
            continue
        if entry.get("status") not in SCOPE_STATUSES or entry["status"] == "unknown":
            errors.append("INVALID_SCOPE_STATUS")
            break
        mapped = [classes[str(a)] for a in entry["a_classes"] if str(a) in classes]
        if len(mapped) != len(entry["a_classes"]):
            errors.append("TISSUE_MAPS_TO_UNKNOWN_A")
        if len(mapped) > 1 and entry["status"] != "in_scope_laterality_unknown":
            errors.append("LATERALITY_INFERRED")
        if len(mapped) > 1 and len({m["host"] for m in mapped}) != 1:
            errors.append("TISSUE_SPANS_HOSTS")
    runnable = {}
    for task, spec in config["tasks"].items():
        if spec.get("phase") == "deferred":
            continue
        ok = True
        for requirement in spec.get("formal_requires", []):
            if requirement in spec:
                ok = ok and spec[requirement] is not None
            else:
                target = config.get("abnormality", {}).get(spec.get("ontology"), None) or config.get("relations", {}).get(spec.get("ontology"), {})
                if "classes" in target:
                    ok = ok and all(c.get(requirement) is not None for c in target["classes"].values())
                else:
                    ok = ok and target.get(requirement) is not None
        runnable[task] = ok
    report = {
        "pass": not errors,
        "errors": errors,
        "versions": {
            "anatomy": "skm_a6_v1",
            "anatomy_legacy": "synthseg_a32_v1",
            "abnormality": "u_v1",
            "relation_geo": "r_geo_v1",
            "relation_clin": "r_clin_v1",
            "tissue_table": "skm_tissue_table_v1",
        },
        "a6_hosts": sorted(hosts),
        "h5_channel_order_status": a6["source_encodings"]["skm_h5_seg_channels"]["status"],
        "formal_runnable": runnable,
        "pending_value_ranges": sorted(k for k, v in config.get("pending_value_ranges", {}).items()
                                       if v.get("status") == "pending_verification"),
    }
    return report
