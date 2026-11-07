from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.source_policy_v5 import (
    SourcePolicy,
    load_source_policies_v5,
    source_binding_v5,
    validate_source_policy_v5,
)

TEMPLATE = Path("configs/v5/sources_v5.template.yaml")


def test_policies_load_and_bind_sources():
    policies = load_source_policies_v5(TEMPLATE)
    assert set(policies) == {"isles_2022", "skm_tea"}
    isles = policies["isles_2022"]
    assert isinstance(isles, SourcePolicy)
    assert (isles.source_dataset, isles.scope_id, isles.model_modalities) == ("ISLES_ltr", "isles_u", ("dwi", "adc"))
    assert source_binding_v5(policies["skm_tea"]) == {"source_id": "skm_tea",
                                                      "source_version": "skm_tea_annotations_v1.0.0"}
    assert isles.annotation_modalities == {"U": None} and isles.index_evidence is None


@pytest.mark.parametrize("mutate", [
    lambda e: e.update(extra=1),
    lambda e: e.pop("scope_id"),
    lambda e: e.update(model_modalities=[]),
    lambda e: e.update(physical_units_policy="guess"),
    lambda e: e.update(annotation_modalities={"U": ["dwi", 3]}),
    lambda e: e.update(source_version=""),
])
def test_policy_rejects_unknown_or_missing_fields(mutate):
    entry = yaml.safe_load(TEMPLATE.read_text())["isles_2022"]
    mutate(entry)
    with pytest.raises(V5Error) as error:
        validate_source_policy_v5("isles_2022", entry)
    assert error.value.code == "INVALID_SOURCE_POLICY"
