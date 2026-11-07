"""Deployment-side tissue-state composition (Proposal_v5 §2.1 state rules 1-5).

Only predictions, acceptance maps and reasons enter here. Reference, review,
visibility and eligibility arrays are never parameters of this module.
"""
from __future__ import annotations

import numpy as np

from .contracts_v5 import V5Error

TARGET_NEGATIVE = 0
TARGET_POSITIVE = 1
NOT_ASSESSABLE = 2
REASON_BITS = {
    "out_of_fov": 1,
    "out_of_scope": 2,
    "anatomy_uncertain": 4,
    "abnormality_uncertain": 8,
    "input_insufficient": 16,
}


def derive_state_map_v5(anatomy_map, anatomy_accepted, anatomy_reason, abnormality_map,
                        abnormality_accepted, abnormality_reason,
                        state_tissue_ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    anatomy_map = np.asarray(anatomy_map)
    anatomy_accepted = np.asarray(anatomy_accepted, dtype=bool)
    anatomy_reason = np.asarray(anatomy_reason, dtype=np.uint8)
    abnormality_map = np.asarray(abnormality_map, dtype=bool)
    abnormality_accepted = np.asarray(abnormality_accepted, dtype=bool)
    abnormality_reason = np.asarray(abnormality_reason, dtype=np.uint8)
    if not state_tissue_ids:
        raise V5Error("STATE_CLASSES_REQUIRED", "state_tissue_ids")
    if abnormality_map.ndim != anatomy_map.ndim + 1 or abnormality_map.shape[0] == 0:
        raise V5Error("JOINT_REQUIRES_U", "abnormality_map")
    spatial = anatomy_map.shape
    channels = (abnormality_map.shape[0], *spatial)
    if (anatomy_accepted.shape, anatomy_reason.shape) != (spatial, spatial) or \
            {abnormality_map.shape, abnormality_accepted.shape, abnormality_reason.shape} != {channels}:
        raise V5Error("GRID_MISMATCH", "abnormality_map")

    in_tissue = np.isin(anatomy_map, np.asarray(state_tissue_ids))
    in_state = anatomy_accepted & in_tissue
    positive = np.any(abnormality_accepted & abnormality_map, axis=0)
    all_negative = np.all(abnormality_accepted, axis=0) & ~np.any(abnormality_map, axis=0)

    state = np.full(spatial, NOT_ASSESSABLE, dtype=np.uint8)
    state[in_state & all_negative] = TARGET_NEGATIVE
    state[in_state & positive] = TARGET_POSITIVE

    reason = np.zeros(spatial, dtype=np.uint8)
    reason[~anatomy_accepted] = anatomy_reason[~anatomy_accepted]
    reason[anatomy_accepted & ~in_tissue] = REASON_BITS["out_of_scope"]
    undecided = in_state & ~positive & ~all_negative
    rejected_reasons = np.bitwise_or.reduce(np.where(abnormality_accepted, 0, abnormality_reason).astype(np.uint8), axis=0)
    undecided_reason = np.where(rejected_reasons == 0, REASON_BITS["abnormality_uncertain"], rejected_reasons)
    reason[undecided] = undecided_reason[undecided]
    return state, reason
