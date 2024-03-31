from sixg_radio_mgmt import MARLCommEnv

import gymnasium as gym
from typing import Optional, Callable, Union
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MARLCustomEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.marl_comm_env = MARLCommEnv(**kwargs)
        self.observation_space = self.marl_comm_env.observation_space
        self.action_space = self.marl_comm_env.action_space
        self.aggregate_actions_steps = 4
        self.slice_req = self.marl_comm_env.comm_env.slice_req
        self.slices = self.marl_comm_env.comm_env.slices

    def reset(
        self,
        seed: Optional[int] = None,
        options: dict = {"initial_episode": -1},
    ):
        obs, _ = self.marl_comm_env.reset(seed=seed, options=options)
        return obs, {}

    def step(self, sched_decision):
        for _ in range(self.aggregate_actions_steps):
            obs, reward, terminated, truncated, info = self.marl_comm_env.step(
                sched_decision
            )
        self.slice_req = self.marl_comm_env.comm_env.slice_req
        self.slices = self.marl_comm_env.comm_env.slices
        print(
            f"Ep: {self.marl_comm_env.comm_env.episode_number}, Step: {self.marl_comm_env.comm_env.step_number}"
        )
        # IMPORTANT: We are returning only the information for the last step
        # but we could also make an average and return
        return obs, reward, terminated, truncated, info

    def set_agent_functions(
        self,
        obs_space_format: Optional[
            Callable[[dict], Union[np.ndarray, dict]]
        ] = None,
        action_format: Optional[
            Callable[[Union[np.ndarray, dict]], np.ndarray]
        ] = None,
        calculate_reward: Optional[
            Callable[[dict], Union[float, dict]]
        ] = None,
        obs_space: gym.spaces.Space = gym.spaces.Space(),
        action_space: gym.spaces.Space = gym.spaces.Space(),
    ):
        self.marl_comm_env.set_agent_functions(
            obs_space_format,
            action_format,
            calculate_reward,
            obs_space,
            action_space,
        )
        self.observation_space = self.marl_comm_env.observation_space
        self.action_space = self.marl_comm_env.action_space
