from typing import Optional, Tuple

import numpy as np

from sixg_radio_mgmt import Association, UEs


class IndustrialAssociation(Association):
    def __init__(
        self,
        ues: UEs,
        max_number_ues: int,
        max_number_basestations: int,
        max_number_slices: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(
            ues,
            max_number_ues,
            max_number_basestations,
            max_number_slices,
            rng,
        )

    def step(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: Optional[dict],
        step_number: int,
        episode_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
        if slice_req is not None:
            slice_ue_assoc = np.zeros_like(slice_ue_assoc)
            num_ue_embb = slice_req["embb"]["number_ues"]
            num_ue_urllc = slice_req["urllc"]["number_ues"]
            num_ue_mmtc = slice_req["mmtc"]["number_ues"]
            slice_ue_assoc[0, 0:num_ue_embb] = 1
            slice_ue_assoc[1, num_ue_embb : num_ue_embb + num_ue_urllc] = 1
            slice_ue_assoc[
                2,
                num_ue_embb
                + num_ue_urllc : num_ue_embb
                + num_ue_urllc
                + num_ue_mmtc,
            ] = 1

        return (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        )
