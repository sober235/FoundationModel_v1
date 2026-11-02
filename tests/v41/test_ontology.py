import pytest

from anatobind.v41.ontology import (
    OntologyError,
    load_ontology,
    map_skm_a_to_host,
    map_skm_a_to_semantic,
    resolve_relation_scope,
    validate_ontology,
)


def test_skm_semantics_are_not_integer_identity():
    assert map_skm_a_to_host(1) == "patellar_cartilage"
    assert map_skm_a_to_host(5) == "meniscus"
    assert resolve_relation_scope(2)["status"] == "out_of_anatomy_scope"


def test_a6_table_matches_contract():
    expected = {
        1: ("patellar_cartilage", "patellar_cartilage", "none"),
        2: ("femoral_cartilage", "femoral_cartilage", "none"),
        3: ("tibial_cartilage_medial", "tibial_cartilage", "medial"),
        4: ("tibial_cartilage_lateral", "tibial_cartilage", "lateral"),
        5: ("meniscus_medial", "meniscus", "medial"),
        6: ("meniscus_lateral", "meniscus", "lateral"),
    }
    for label, (name, host, side) in expected.items():
        semantic = map_skm_a_to_semantic(label)
        assert (semantic["name"], semantic["host"], semantic["laterality"]) == (name, host, side)
    assert map_skm_a_to_host(0) == "background"


def test_unknown_integer_is_never_background():
    for bad in (7, 13, -1, 255):
        with pytest.raises(OntologyError, match="UNKNOWN_A_LABEL"):
            map_skm_a_to_host(bad)
    assert resolve_relation_scope(42)["status"] == "unknown"
    assert resolve_relation_scope(42)["a_classes"] == []


def test_ligaments_are_out_of_scope_not_background():
    for tissue in (2, 3):
        scope = resolve_relation_scope(tissue)
        assert scope["status"] == "out_of_anatomy_scope"
        assert scope["a_classes"] == []
        assert scope["tissue_name"] in ("ACL", "PCL")


def test_laterality_is_not_inferred_from_tissue_id():
    meniscus = resolve_relation_scope(1)
    assert meniscus["status"] == "in_scope_laterality_unknown"
    assert meniscus["a_classes"] == [5, 6]
    tibial = resolve_relation_scope(6)
    assert tibial["a_classes"] == [3, 4]
    assert resolve_relation_scope(-1)["status"] == "no_host"
    assert resolve_relation_scope(4)["a_classes"] == [2]
    assert resolve_relation_scope(5)["a_classes"] == [1]


def test_validate_ontology_reports_unsigned_clinical_tasks():
    config = load_ontology()
    report = validate_ontology(config)
    assert report["pass"] is True
    assert report["versions"]["anatomy"] == "skm_a6_v1"
    assert report["versions"]["anatomy_legacy"] == "synthseg_a32_v1"
    assert report["h5_channel_order_status"] == "pending_verification"
    assert report["formal_runnable"]["R_clin"] is False
    assert report["formal_runnable"]["U"] is False
    assert report["formal_runnable"]["surface_defect"] is False
    assert "defect_volume" not in report["formal_runnable"]
    assert report["pending_value_ranges"] == ["ucsf_bmsr", "ucsf_pdgm"]


def test_validate_ontology_rejects_merged_scope_statuses_and_duplicates():
    config = load_ontology()
    config["anatomy"]["skm_a6_v1"]["classes"]["6"]["name"] = "meniscus_medial"
    report = validate_ontology(config)
    assert report["pass"] is False
    assert "DUPLICATE_CLASS_NAME" in report["errors"]
    config = load_ontology()
    config["skm_tissue_table_v1"]["2"]["status"] = "background"
    report = validate_ontology(config)
    assert "INVALID_SCOPE_STATUS" in report["errors"]
    config = load_ontology()
    config["anatomy"]["skm_a6_v1"]["classes"]["3"]["laterality"] = "inferred"
    assert "INVALID_LATERALITY" in validate_ontology(config)["errors"]
