from typing import Union

import joblib
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.sac.sac import SAC

from sixg_radio_mgmt import Agent
from marl_custom_env import MARLCustomEnv


class MARLSafe(Agent):

    def __init__(
        self,
        env: MARLCustomEnv,
        max_number_ues: int,
        max_number_slices: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        hyperparameters: dict = {},
        seed: int = np.random.randint(1000),
        hyperparams: str = "",
    ) -> None:
        super().__init__(
            env,  # type: ignore We created a new class instead of using MARLCommEnv
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

    def step(self, obs_space: Union[np.ndarray, dict]) -> np.ndarray:
        return self.agent.predict(np.asarray(obs_space), deterministic=True)[0]

    def train(self, total_timesteps: int) -> None:
        pass

    def save(self, filename: str) -> None:
        self.agent.save(filename)

    def load(self, filename: str, env: MARLCustomEnv) -> None:
        self.agent = SAC.load(filename, env=env)

    def obs_space_format(
        self, obs_space: dict, normalization: bool = True
    ) -> dict:
        formatted_obs_space = {
            "player_0": np.array([]),
            "player_1": None,
            "player_2": None,
            "player_3": None,
        }
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
            formatted_obs_space["player_0"] = np.append(
                formatted_obs_space["player_0"],
                self.slice_average(obs_space, hist_label)
                / normalization_factors[hist_label],
                axis=0,
            )

        for intra_idx in np.arange(1, 4):  # Intra-slice TODO
            formatted_obs_space[f"player_{intra_idx}"] = np.zeros(3)

        return formatted_obs_space

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
            self.env, MARLCustomEnv
        ), "The environment must be an instance of the CommunicationEnv class"
        metric_slices = self.obs_space_format(obs_space, False)
        metric_slices = metric_slices["player_0"]
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
        reward_dict = {  # TODO
            "player_0": total_reward,
            "player_1": 0,
            "player_2": 0,
            "player_3": 0,
        }

        return reward_dict

    @staticmethod
    def get_action_space() -> spaces.Dict:
        action_space = spaces.Dict(
            {
                f"player_{idx}": (
                    spaces.Box(
                        low=-1,
                        high=1,
                        shape=(3,),
                        dtype=np.float64,
                    )
                    if idx == 0
                    else spaces.Discrete(3)
                )
                for idx in range(4)
            }
        )
        return action_space

    @staticmethod
    def get_obs_space() -> spaces.Dict:
        obs_space = spaces.Dict(
            {
                f"player_{idx}": (
                    spaces.Box(
                        low=0,
                        high=np.inf,
                        shape=(9,),
                        dtype=np.float64,
                    )
                    if idx == 0
                    else spaces.Box(
                        low=0,
                        high=np.inf,
                        shape=(3,),
                        dtype=np.float64,
                    )
                )
                for idx in range(4)
            }
        )

        return obs_space

    def action_format(
        self,
        action: Union[np.ndarray, dict],
    ) -> np.ndarray:
        assert isinstance(action, dict), "Action must be a Dict"
        assert isinstance(
            self.env, MARLCustomEnv
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
        # TODO Implement intra-slice schedulers based on intra-slice actions
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
