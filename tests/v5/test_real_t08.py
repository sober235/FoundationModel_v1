"""Real T08 split freezing (plan_v5 §2B T08b; §1.6 L8). Synthetic records only."""
from __future__ import annotations

import json

import pytest

from anatobind.v41.contracts_v5 import V5Error, validate_sample_v5
from anatobind.v41.real_t08_v5 import (
    official_split_from_release_v5,
    render_split_md_v5,
    run_freeze_v5,
    volume_tertile_strata_v5,
)
from anatobind.v41.splits_v5 import verify_split_registry_v5
from tests.v5.factories import make_sample_v5

UNKNOWN = {"experiment_id": "legacy", "role": "unknown", "recorded_at": "2026-11-02T00:00:00Z",
           "lineage_id": None, "evidence": "migrated_from_v41"}


def _record(i: int, patient: int | None = None, release: str | None = None, exposure=None):
    patient = i if patient is None else patient
    body = make_sample_v5(sample_uid=f"s-{i}", patient_uid=f"p-{patient}", leakage_group_uid=f"p-{patient}",
                          exposure_history=list(exposure or []))
    body["provenance"] = {**body["provenance"], "release_split": release}
    return validate_sample_v5(body)


def test_official_split_comes_from_the_release_per_patient():
    records = [_record(0, 0, "train"), _record(1, 0, "train"), _record(2, 2, "val"), _record(3, 3, "test"),
               _record(4, 4, None)]
    assert official_split_from_release_v5(records) == {"p-0": "train", "p-2": "val", "p-3": "test"}
    with pytest.raises(V5Error) as error:
        official_split_from_release_v5([_record(0, 0, "train"), _record(1, 0, "test")])
    assert (error.value.code, error.value.path) == ("OFFICIAL_SPLIT_CONFLICT", "$.records[1].provenance.release_split")
    with pytest.raises(V5Error) as error:
        official_split_from_release_v5([_record(0, 0, "holdout")])
    assert error.value.code == "INVALID_RELEASE_SPLIT"


def test_volume_tertiles_cut_patient_volumes():
    volumes = {f"p-{i}": float(v) for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8])}
    strata, thresholds = volume_tertile_strata_v5(volumes)
    assert [strata[f"p-{i}"] for i in range(9)] == ["volume_t1"] * 3 + ["volume_t2"] * 3 + ["volume_t3"] * 3
    assert thresholds == pytest.approx([8 / 3, 16 / 3])
    with pytest.raises(V5Error) as error:
        volume_tertile_strata_v5({"p-0": -1.0})
    assert error.value.code == "INVALID_VOLUME"


def test_freeze_writes_private_registry_and_redacted_summary(tmp_path):
    records = [_record(i, exposure=[UNKNOWN] if i == 0 else []) for i in range(30)]
    volumes = {f"p-{i}": float(i) for i in range(30)}
    policy = {"kind": "stratified", "seed": 20261108,
              "fractions": {"train": 0.7, "development_val": 0.15, "locked_test": 0.15},
              "min_stratum_groups": 5, "small_stratum_rule": "merge", "exposed_groups": "excluded"}
    out = tmp_path / "t08"
    summary = run_freeze_v5(records, policy, "v51_synthetic_b2", out, volumes=volumes)
    registry = verify_split_registry_v5(json.loads((out / "split_registry.json").read_text()))
    assert oct(out.stat().st_mode & 0o777) == "0o700"
    assert oct((out / "split_registry.json").stat().st_mode & 0o777) == "0o600"
    assert registry["lineage_id"] == "v51_synthetic_b2" and registry["ancestor_lineages"] == []
    assert set(registry["policy"]["strata"].values()) == {"volume_t1", "volume_t2", "volume_t3"}
    assert summary["split_sha256"] == registry["split_sha256"]
    assert sum(summary["counts"].values()) == 30 and summary["strata"] == {
        "volume_t1": 10, "volume_t2": 10, "volume_t3": 10}
    md = render_split_md_v5(summary)
    assert "s-0" not in md and "p-0" not in md and registry["split_sha256"] in md
    if registry["assignment"]["s-0"] != "train" and registry["assignment"]["s-0"] != "development_val":
        assert registry["assignment"]["s-0"] == "excluded"                   # unknown exposure never reaches locked_test
    with pytest.raises(Exception):
        run_freeze_v5(records, policy, "v51_synthetic_b2", out, volumes=volumes)


def test_official_freeze_uses_the_release(tmp_path):
    records = [_record(0, release="train"), _record(1, release="val"), _record(2, release="test"),
               _record(3, release="test", exposure=[UNKNOWN])]
    policy = {"kind": "official", "exposed_groups": "excluded"}
    summary = run_freeze_v5(records, policy, "v51_synthetic_b2", tmp_path / "t08")
    registry = json.loads((tmp_path / "t08/split_registry.json").read_text())
    assert registry["assignment"] == {"s-0": "train", "s-1": "development_val", "s-2": "locked_test",
                                      "s-3": "excluded"}
    assert summary["exposed_groups"] == 1 and summary["strata"] is None


def test_lesion_volume_is_measured_in_ml_from_the_released_mask(tmp_path):
    import nibabel as nib
    import numpy as np

    from anatobind.v41.real_t08_v5 import lesion_volumes_ml_v5
    from anatobind.v41.contracts_v5 import validate_scope_v5
    from tests.v5.factories import make_module_scope

    mask = np.zeros((8, 8, 4), dtype=np.uint8)
    mask[:5, :5, :2] = 1                                                  # 50 voxels of 2 mm cubes = 0.4 mL
    nib.save(nib.Nifti1Image(mask, np.diag([2.0, 2.0, 2.0, 1.0])), str(tmp_path / "m.nii.gz"))
    scope = validate_scope_v5(make_module_scope("U"))
    annotation = {"task": "U", "class_scope": ["target_a"], "representation": "voxel", "path": "m.nii.gz",
                  "label_map_version": "synthetic_values_v1"}
    body = make_sample_v5(annotation=annotation, sample_uid="s-0", patient_uid="p-0", leakage_group_uid="p-0")
    body["image_geometry"] = {**body["image_geometry"], "spacing_mm": [2.0, 2.0, 2.0]}
    record = validate_sample_v5(body)
    maps = {"synthetic_values_v1": {"encoding": "values", "classes": {"target_a": 1}}}
    assert lesion_volumes_ml_v5(tmp_path, [record, ], scope, maps) == {"p-0": pytest.approx(0.4)}
    body["image_geometry"] = None
    body.update(index_alignment_status="unknown", physical_geometry_status="unknown",
                geometry_evidence={"index_alignment": None, "physical_geometry": None})
    with pytest.raises(V5Error) as error:
        lesion_volumes_ml_v5(tmp_path, [validate_sample_v5(body)], scope, maps)
    assert (error.value.code, error.value.path) == ("SPACING_UNKNOWN", "$.records[0].image_geometry")
