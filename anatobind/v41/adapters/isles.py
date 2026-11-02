"""ISLES-2022 pairing of per-case mask with DWI/ADC/FLAIR assets (read-only)."""
from __future__ import annotations

SOURCE = "ISLES_ltr"
ROLES = {"dwi": "_dwi.nii.gz", "adc": "_adc.nii.gz", "flair": "_flair.nii.gz"}


def pair_assets(assets: list[dict]) -> list[dict]:
    cases: dict[str, dict] = {}
    for asset in assets:
        if asset.get("source") != SOURCE or asset.get("kind") not in ("image", "label") or not asset.get("case_id"):
            continue
        entry = cases.setdefault(asset["case_id"], {"case_id": asset["case_id"], "source": SOURCE,
                                                     "label_relpath": None, "images": {}})
        name = asset["relpath"].lower()
        if asset["kind"] == "label":
            entry["label_relpath"] = asset["relpath"]
            continue
        for role, suffix in ROLES.items():
            if name.endswith(suffix):
                entry["images"][role] = asset["relpath"]
    return [cases[k] for k in sorted(cases)]
