from typing import Union

import numpy as np

from sixg_radio_mgmt import Agent, CommunicationEnv


class SSR(Agent):
    def __init__(
        self,
        env: CommunicationEnv,
        max_number_ues: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
    ) -> None:
        super().__init__(
            env, max_number_ues, max_number_basestations, num_available_rbs
        )

        # Variables for round-robin scheduling
        self.current_ues = np.array([])
        self.rbs_per_ue = np.zeros(
            (self.env.max_number_slices, max_number_ues)
        )
        self.allocation_rbs = []

    def step(self, obs_space: Union[np.ndarray, dict]) -> np.ndarray:
        rbs_per_slice = self.inter_slice_scheduling(obs_space)
        allocation_rbs = self.intra_slice_scheduling(
            rbs_per_slice, "round_robin", dict(obs_space)
        )

        return allocation_rbs

    def inter_slice_scheduling(
        self, obs_space: Union[np.ndarray, dict]
    ) -> np.ndarray:
        return np.array([20, 20, 60])

    def intra_slice_scheduling(
        self, rbs_per_slice: np.ndarray, method: str, obs_space: dict
    ) -> np.ndarray:
        match method:
            case "round_robin":
                return self.round_robin(
                    rbs_per_slice, obs_space["slice_ue_assoc"]
                )

        return np.array([])

    def round_robin(
        self,
        rbs_per_slice: np.ndarray,
        slice_ue_assoc: np.ndarray,
    ) -> np.ndarray:
        number_slices = len(rbs_per_slice)
        if np.array_equal(self.allocation_rbs, np.array([])):
            initial_rb = 0
            self.allocation_rbs = [
                np.zeros(
                    (
                        self.max_number_ues,
                        self.num_available_rbs[basestation],
                    )
                )
                for basestation in np.arange(self.max_number_basestations)
            ]
            for slice_idx in np.arange(number_slices):
                idx_active_ues = slice_ue_assoc[slice_idx].nonzero()[0]
                num_active_ues = np.sum(slice_ue_assoc[slice_idx]).astype(int)
                num_rbs_per_ue = int(
                    (
                        np.floor(rbs_per_slice[slice_idx] / num_active_ues)
                        if num_active_ues > 0
                        else 0
                    )
                )
                remaining_rbs = (
                    rbs_per_slice[slice_idx] - num_rbs_per_ue * num_active_ues
                )
                self.rbs_per_ue = np.ones(num_active_ues) * num_rbs_per_ue
                self.rbs_per_ue[:remaining_rbs] += 1

                for idx, ue_idx in enumerate(idx_active_ues):
                    self.allocation_rbs[0][
                        ue_idx,
                        initial_rb : initial_rb + int(self.rbs_per_ue[idx]),
                    ] = 1
                    initial_rb += int(self.rbs_per_ue[idx])

        return np.array(self.allocation_rbs)

    def obs_space_format(self, obs_space: dict) -> dict:
        return obs_space

    def calculate_reward(self, obs_space: dict) -> float:
        return 0

    def action_format(self, action: np.ndarray) -> np.ndarray:
        return action
