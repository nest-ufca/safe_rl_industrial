from sixg_radio_mgmt import CommunicationEnv

import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union
class CustomEnv(gym.Env):
    def __init__(self, **kwargs):
        self.comm_env = CommunicationEnv(**kwargs)
        self.observation_space = self.comm_env.observation_space
        self.action_space = self.comm_env.action_space
        self.aggregate_actions_steps = 4
        self.slice_req = self.comm_env.slice_req
        self.slices = self.comm_env.slices
        self.seed: int = np.random.randint(1000)

    def reset(self, seed: Optional[int] = None):
        return self.comm_env.reset(seed=seed)

    def step(self, sched_decision):
        for _ in range(self.aggregate_actions_steps):
            obs, reward, terminated, truncated, info = self.comm_env.step(
                sched_decision
            )
        self.slice_req = self.comm_env.slice_req
        self.slices = self.comm_env.slices
        print(
            f"Ep: {self.comm_env.episode_number}, Step: {self.comm_env.step_number}"
        )
        # IMPORTANT: We are returning only the information for the last step
        # but we could also make an average and return
        return obs, reward, terminated, truncated, info
