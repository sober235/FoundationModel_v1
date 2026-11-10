"""Protocol table and findings: box, anchor, host tissue group and fixed wording (design 2026-11-10 §5.4)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.findings_v5 import (
    findings_from_output_v5, host_relation_fn_v5, load_protocols_v5, render_findings_zh_v5)

CONFIG = Path("configs/v5")
PROTOCOLS = load_protocols_v5(CONFIG / "protocols_v5.yaml", CONFIG)
KNEE, BRAIN = PROTOCOLS["knee_qdess"], PROTOCOLS["brain_dwi_adc"]
SCOPE, GROUPS, GROUPS_BY_CLASS, DISPLAY = KNEE["scope"], KNEE["groups"], KNEE["groups_by_class"], KNEE["display"]
PLAN = yaml.safe_load((CONFIG / "proposals" / "unified_knee_brain.yaml").read_text())
POLICY = PLAN["members"]["knee_qdess"]["acceptance_policy"]
FEMORAL = SCOPE.anatomy_classes.index("femoral_cartilage") + 1
MENISCUS = SCOPE.anatomy_classes.index("meniscus_medial") + 1
SHAPE = (8, 4, 4)
FINDING = {"finding_uid": "inst-0001", "abnormality_class": "cartilage_lesion",
           "box": {"start": [1, 1, 0], "size": [2, 3, 1]}, "anchor": [1, 2, 0], "voxels": 6,
           "host_group": "femoral_cartilage", "host_status": "bound"}


def _instance(mask, class_name="cartilage_lesion", uid="inst-0001"):
    where = np.argwhere(mask)
    bbox = [int(v) for v in where.min(axis=0)] + [int(v) + 1 for v in where.max(axis=0)]
    return {"instance_uid": uid, "class_name": class_name, "mask": mask, "bbox": bbox,
            "candidate_set_hash": "c" * 64, "accepted": True}


def test_protocol_table_resolves_scopes_groups_and_wording():
    assert KNEE["scope"].scope_id == "skm_joint_early" and BRAIN["scope"].scope_id == "isles_u"
    assert (KNEE["source_id"], BRAIN["source_id"]) == ("skm_tea_joint_early", "isles_2022")
    assert GROUPS["tibial_cartilage"] == ["tibial_cartilage_medial", "tibial_cartilage_lateral"]
    assert GROUPS_BY_CLASS == {"cartilage_lesion": ["femoral_cartilage", "patellar_cartilage", "tibial_cartilage"],
                               "meniscal_tear": ["meniscus"]}
    assert BRAIN["groups"] is None and BRAIN["display"] == {"ischemic_infarct": "缺血性梗死"}
    assert set(PLAN["members"]) == set(PROTOCOLS)                    # the proposed plan's members are protocols


def test_host_is_the_allowed_group_with_most_accepted_voxels():
    anatomy_map = np.full(SHAPE, MENISCUS, dtype=np.int64)
    anatomy_map[:4] = FEMORAL
    mask = np.zeros(SHAPE, dtype=bool)
    mask[3, 0:3, 0:2] = True                                          # 6 femoral cartilage voxels
    mask[4, 0:4, 0:2] = True
    mask[5, 0, 0:2] = True                                            # 10 meniscus voxels in all
    accepted = np.ones(SHAPE, dtype=bool)
    relation_fn = host_relation_fn_v5(GROUPS, GROUPS_BY_CLASS)
    relations, reasons = relation_fn([_instance(mask)], anatomy_map, accepted, SCOPE)
    assert relations[0]["status"] == "bound" and relations[0]["tissue_ids"] == [FEMORAL]   # meniscus is not allowed
    assert relations[0]["predicate"] == "located_in_tissue_group" and relations[0]["accepted"] is True
    assert relations[0]["completion_role"] == "auxiliary_spatial" and reasons == {"inst-0001": 0}
    relations, _ = relation_fn([_instance(mask, "meniscal_tear")], anatomy_map, accepted, SCOPE)
    assert relations[0]["tissue_ids"] == [MENISCUS]
    relations, _ = relation_fn([_instance(mask)], anatomy_map, np.zeros(SHAPE, dtype=bool), SCOPE)
    assert relations[0]["status"] == "unknown" and relations[0]["tissue_ids"] == []


def test_a_finding_carries_box_anchor_host_and_a_fixed_sentence():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[1:3, 1:4, 0] = True                                          # x 1..2, y 1..3, z 0: 6 voxels
    output = SimpleNamespace(
        output_kind="joint", abnormal_instances=[_instance(mask)],
        tissue_relations=[{"instance_uid": "inst-0001", "tissue_ids": [FEMORAL], "status": "bound"}])
    findings = findings_from_output_v5(output, SCOPE, POLICY, GROUPS)
    assert findings == [FINDING]                                      # centroid (1.5, 2, 0): the lower voxel wins
    assert render_findings_zh_v5(findings, SCOPE, DISPLAY) == ["在股骨软骨上，疑似软骨病变"]


def test_undetermined_host_and_no_findings_have_their_own_wording():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[1:3, 1:4, 0] = True
    output = SimpleNamespace(
        output_kind="joint", abnormal_instances=[_instance(mask)],
        tissue_relations=[{"instance_uid": "inst-0001", "tissue_ids": [], "status": "unknown"}])
    undetermined = findings_from_output_v5(output, SCOPE, POLICY, GROUPS)
    assert undetermined == [dict(FINDING, host_group=None, host_status="undetermined")]
    assert render_findings_zh_v5(undetermined, SCOPE, DISPLAY) == ["疑似软骨病变，所在结构未能确定"]
    assert render_findings_zh_v5([], SCOPE, DISPLAY) == [
        "未见所列异常（仅限：软骨病变、半月板撕裂）。医生没有标注不等于正常。"]
    with pytest.raises(V5Error) as err:
        render_findings_zh_v5([FINDING], SCOPE, {"cartilage_lesion": "软骨病变"})
    assert (err.value.code, err.value.path) == ("PROTOCOL_DISPLAY_MISSING", "$.display")


def test_a_module_output_gives_findings_without_a_host():
    scope, policy = BRAIN["scope"], PLAN["members"]["brain_dwi_adc"]["acceptance_policy"]
    abnormality = np.zeros((1, *SHAPE), dtype=bool)
    abnormality[0, 2:4, 0:2, 0:2] = True                              # one 8-voxel component
    output = SimpleNamespace(output_kind="module", abnormality_map=abnormality,
                             acceptance={"abnormality": np.ones_like(abnormality)})
    findings = findings_from_output_v5(output, scope, policy, None)
    assert [(f["host_group"], f["host_status"], f["voxels"]) for f in findings] == [(None, "not_reported", 8)]
    assert findings[0]["box"] == {"start": [2, 0, 0], "size": [2, 2, 2]}
    assert render_findings_zh_v5(findings, scope, BRAIN["display"]) == ["疑似缺血性梗死"]
    assert render_findings_zh_v5([], scope, BRAIN["display"]) == ["未见所列异常（仅限：缺血性梗死）。医生没有标注不等于正常。"]
