from typing import Union

import joblib
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from custom_env import CustomEnv
from marl_custom_env import MARLCustomEnv
from sixg_radio_mgmt import Agent, CommunicationEnv, MARLCommEnv


class SSRProtectMARL(Agent):
    def __init__(
        self,
        env: CommunicationEnv,
        max_number_ues: int,
        max_number_slices: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        hyperparameters: dict = {},
        seed: int = np.random.randint(1000),
        hyperparams: str = "",
    ) -> None:
        super().__init__(
            env,
            max_number_ues,
            max_number_slices,
            max_number_basestations,
            num_available_rbs,
            seed,
        )
        # Variables for round-robin scheduling
        self.current_ues = np.array([])
        self.rbs_per_ue = np.zeros((self.max_number_slices, max_number_ues))
        self.allocation_rbs = []

    def obs_space_format(
        self, obs_space: dict, normalization: bool = True
    ) -> dict:
        formatted_obs_space = np.array([])
        hist_labels = [
            "pkt_throughputs",
            "buffer_latencies",
            "buffer_occupancies",
        ]
        if normalization:
            normalization_factors = {
                "pkt_throughputs": 50,
                "buffer_latencies": 100,
                "buffer_occupancies": 1,
            }
        else:
            normalization_factors = {
                "pkt_throughputs": 1,
                "buffer_latencies": 1,
                "buffer_occupancies": 1,
            }
        for hist_label in hist_labels:
            formatted_obs_space = np.append(
                formatted_obs_space,
                self.slice_average(obs_space, hist_label)
                / normalization_factors[hist_label],
                axis=0,
            )

        return {
            "player_0": formatted_obs_space,
            "player_1": np.array([0]),
            "player_2": np.array([0]),
            "player_3": np.array([0]),
        }

    def slice_average(self, obs_space: dict, metric: str) -> np.ndarray:
        number_slices = obs_space["slice_ue_assoc"].shape[0]
        slice_values = np.zeros(number_slices)
        pkts_to_mbps = 8192 * 8 / 1e6 if metric in ["pkt_throughputs"] else 1
        for slice_idx in np.arange(number_slices):
            slice_values[slice_idx] = np.sum(
                pkts_to_mbps
                * obs_space[metric]
                * obs_space["slice_ue_assoc"][slice_idx]
            ) / np.sum(obs_space["slice_ue_assoc"][slice_idx])

        return slice_values

    def calculate_reward(self, obs_space: dict) -> dict:
        assert isinstance(
            self.env, (MARLCustomEnv, MARLCommEnv)
        ), "The environment must be an instance of the MARLCustomEnv"
        metric_slices = self.obs_space_format(obs_space, False)["player_0"]
        maximum_buffer_latency = 100
        reward = {
            "urllc": {
                "throughput": {
                    "value": 0,
                    "weight": 0.5,
                },
                "latency": {
                    "value": 0,
                    "weight": 0.5,
                },
            },
            "embb": {
                "throughput": {
                    "value": 0,
                    "weight": 0.5,
                },
                "latency": {
                    "value": 0,
                    "weight": 0.3,
                },
            },
            "mmtc": {
                "latency": {
                    "value": 0,
                    "weight": 0.2,
                },
            },
        }

        # URLLC
        urllc_req_throughput = self.env.comm_env.slice_req["urllc"][
            "ue_throughput"
        ]
        urllc_req_latency = self.env.comm_env.slice_req["urllc"]["latency"]
        reward["urllc"]["throughput"]["value"] -= (
            1 - metric_slices[1] / urllc_req_throughput
            if metric_slices[1] < urllc_req_throughput
            else 0
        )
        reward["urllc"]["latency"]["value"] -= (
            (metric_slices[4] - urllc_req_latency)
            / (maximum_buffer_latency - urllc_req_latency)
            if metric_slices[4] > urllc_req_latency
            else 0
        )

        if np.isclose(
            reward["urllc"]["throughput"]["value"]
            + reward["urllc"]["latency"]["value"],
            0,
        ):
            # eMBB
            embb_req_throughput = self.env.comm_env.slice_req["embb"][
                "ue_throughput"
            ]
            embb_req_latency = self.env.comm_env.slice_req["embb"]["latency"]
            reward["embb"]["throughput"]["value"] -= (
                1 - metric_slices[0] / embb_req_throughput
                if metric_slices[0] < embb_req_throughput
                else 0
            )
            reward["embb"]["latency"]["value"] -= (
                (metric_slices[3] - embb_req_latency)
                / (maximum_buffer_latency - embb_req_latency)
                if metric_slices[3] > embb_req_latency
                else 0
            )

            # mMTC
            mmtc_req_latency = self.env.comm_env.slice_req["mmtc"]["latency"]
            reward["mmtc"]["latency"]["value"] -= (
                (metric_slices[5] - mmtc_req_latency)
                / (maximum_buffer_latency - mmtc_req_latency)
                if metric_slices[5] > mmtc_req_latency
                else 0
            )

            total_reward = (
                reward["embb"]["throughput"]["weight"]
                * reward["embb"]["throughput"]["value"]
                + reward["embb"]["latency"]["weight"]
                * reward["embb"]["latency"]["value"]
                + reward["mmtc"]["latency"]["value"]
                * reward["mmtc"]["latency"]["weight"]
            )
        else:
            total_reward = (
                reward["urllc"]["throughput"]["value"]
                * reward["urllc"]["throughput"]["weight"]
                + reward["urllc"]["latency"]["value"]
                * reward["urllc"]["latency"]["weight"]
            ) - 1

        return {
            "player_0": total_reward,
            "player_1": 0,
            "player_2": 0,
            "player_3": 0,
        }

    @staticmethod
    def get_action_space() -> spaces.Dict:
        return spaces.Dict(
            {
                "player_0": spaces.Box(low=-1, high=1, shape=(3,)),
                "player_1": spaces.Box(low=-1, high=1, shape=(1,)),
                "player_2": spaces.Box(low=-1, high=1, shape=(1,)),
                "player_3": spaces.Box(low=-1, high=1, shape=(1,)),
            }
        )

    @staticmethod
    def get_obs_space() -> spaces.Dict:
        return spaces.Dict(
            {
                "player_0": spaces.Box(
                    low=0, high=np.inf, shape=(3 * 3,), dtype=np.float64
                ),
                "player_1": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "player_2": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
                "player_3": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float64
                ),
            }
        )

    def action_format(
        self,
        action: Union[np.ndarray, dict],
    ) -> np.ndarray:
        assert isinstance(action, dict), "Action must be a dict"
        assert isinstance(
            self.env, (MARLCustomEnv, MARLCommEnv)
        ), "The environment must be an instance of the MARLCustomEnv"
        action_rbs = (
            np.around(
                self.num_available_rbs[0]
                * (action["player_0"] + 1)
                / np.sum(action["player_0"] + 1)
            )
            if not np.isclose(np.sum(action["player_0"] + 1), 0)
            else np.zeros(3)
        )
        sched_decision = self.round_robin(
            action_rbs, self.env.comm_env.slices.ue_assoc
        )

        return sched_decision

    def round_robin(
        self,
        rbs_per_slice: np.ndarray,
        slice_ue_assoc: np.ndarray,
    ) -> np.ndarray:
        number_slices = len(rbs_per_slice)
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
            ).astype(int)
            self.rbs_per_ue = np.ones(num_active_ues) * num_rbs_per_ue
            self.rbs_per_ue[:remaining_rbs] += 1

            for idx, ue_idx in enumerate(idx_active_ues):
                self.allocation_rbs[0][
                    ue_idx,
                    initial_rb : initial_rb + int(self.rbs_per_ue[idx]),
                ] = 1
                initial_rb += int(self.rbs_per_ue[idx])

        return np.array(self.allocation_rbs)

    def step(self):
        raise NotImplementedError
