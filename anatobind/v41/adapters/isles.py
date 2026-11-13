"""ISLES-2022 pairing of per-case mask with DWI/ADC assets (read-only).

FLAIR is not registered as a model input (researcher decision 2026-11-02):
it is not on the mask grid in 250/250 cases and the target U class is the
DWI-defined acute/sub-acute infarct.
"""
from __future__ import annotations

SOURCE = "ISLES_ltr"
ROLES = {"dwi": "_dwi.nii.gz", "adc": "_adc.nii.gz"}


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


def pairing_report(assets: list[dict]) -> dict:
    """Counts only: complete cases (DWI, ADC and mask), what is missing, and assets that would overwrite another.

    `pair_assets` keeps the last asset of a role; a second mask or a second image of one role is therefore
    counted here as a duplicate instead of passing unnoticed. FLAIR and other images are not model inputs and
    are only counted.
    """
    seen: dict[tuple, int] = {}
    other_images = 0
    for asset in assets:
        if asset.get("source") != SOURCE or asset.get("kind") not in ("image", "label") or not asset.get("case_id"):
            continue
        if asset["kind"] == "label":
            role = "mask"
        else:
            name = asset["relpath"].lower()
            role = next((r for r, suffix in ROLES.items() if name.endswith(suffix)), None)
            if role is None:
                other_images += 1
                continue
        seen[(asset["case_id"], role)] = seen.get((asset["case_id"], role), 0) + 1
    pairs = pair_assets(assets)
    complete = [p for p in pairs if p["label_relpath"] and all(r in p["images"] for r in ROLES)]
    return {"cases": len(pairs), "paired": len(complete),
            "missing_mask": sum(1 for p in pairs if not p["label_relpath"]),
            **{f"missing_{role}": sum(1 for p in pairs if role not in p["images"]) for role in ROLES},
            "duplicate_role_assets": sum(n - 1 for n in seen.values() if n > 1),
            "cases_with_duplicate_assets": len({case for (case, _), n in seen.items() if n > 1}),
            "other_images_not_registered": other_images}
