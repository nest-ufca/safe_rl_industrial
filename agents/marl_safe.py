from collections import deque
from typing import Union

import joblib
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.sac.sac import SAC

from agents.common import (
    max_throughput,
    proportional_fairness,
    round_robin,
    scores_to_rbs,
)
from marl_custom_env import MARLCustomEnv
from sixg_radio_mgmt import Agent


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
        self.last_unformatted_obs = deque(maxlen=10)

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
        self.last_unformatted_obs.appendleft(obs_space)
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
        # Include requirements in the observation
        requirements = np.array(
            [
                self.env.comm_env.slice_req["urllc"]["ue_throughput"],
                self.env.comm_env.slice_req["urllc"]["latency"],
                self.env.comm_env.slice_req["embb"]["ue_throughput"],
                self.env.comm_env.slice_req["embb"]["latency"],
                self.env.comm_env.slice_req["mmtc"]["ue_throughput"],
                self.env.comm_env.slice_req["mmtc"]["latency"],
            ]
        )
        formatted_obs_space["player_0"] = np.append(
            formatted_obs_space["player_0"], requirements
        )

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

        for intra_idx in np.arange(1, 4):  # TODO Intra-slice
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

    def get_slice_metrics(self, obs_space: dict, metric: str) -> np.ndarray:
        # TODO Define intra-slice metrics
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
                        shape=(5 + 9,),
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
        sched_decision = np.array(
            [
                np.zeros(
                    (self.max_number_ues, self.num_available_rbs[basestation])
                )
                for basestation in np.arange(self.max_number_basestations)
            ]
        )
        action_rbs = scores_to_rbs(
            action["player_0"],
            self.num_available_rbs[0].astype(int),
            self.last_unformatted_obs[0]["basestation_slice_assoc"][0, :],
        )
        for player_idx in np.arange(1, len(action)):
            slice_ues = self.last_unformatted_obs[0]["slice_ue_assoc"][
                player_idx - 1
            ].nonzero()[0]
            if np.isclose(action_rbs[player_idx - 1], 0):
                continue
            match action[f"player_{player_idx}"]:
                case 0:
                    sched_decision = round_robin(
                        allocation_rbs=sched_decision,
                        slice_idx=player_idx - 1,
                        rbs_per_slice=action_rbs,
                        slice_ues=slice_ues,
                        last_unformatted_obs=self.last_unformatted_obs,
                        account_buffer=False,
                    )
                case 1:
                    sched_decision = proportional_fairness(
                        allocation_rbs=sched_decision,
                        slice_idx=player_idx - 1,
                        rbs_per_slice=action_rbs,
                        slice_ues=slice_ues,
                        env=self.env,
                        last_unformatted_obs=self.last_unformatted_obs,
                        num_available_rbs=self.num_available_rbs,
                    )
                case 2:
                    sched_decision = max_throughput(
                        allocation_rbs=sched_decision,
                        slice_idx=player_idx - 1,
                        rbs_per_slice=action_rbs,
                        slice_ues=slice_ues,
                        env=self.env,
                        last_unformatted_obs=self.last_unformatted_obs,
                        num_available_rbs=self.num_available_rbs,
                    )
                case _:
                    raise ValueError("Invalid intra-slice scheduling action")
        assert (
            np.sum(sched_decision) == self.num_available_rbs[0]
        ), f"Allocated RBs {np.sum(sched_decision)} are different from available RBs {self.num_available_rbs[0]}"

        return sched_decision
