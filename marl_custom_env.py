from typing import Callable, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from sixg_radio_mgmt import CommunicationEnv


class MARLCustomEnv(MultiAgentEnv):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.comm_env = CommunicationEnv(*args, **kwargs)
        self.agents = {
            "player_" + str(r)
            for r in range(self.comm_env.max_number_slices + 1)
        }
        self._agent_ids = set(self.agents)
        self._obs_space_in_preferred_format = True
        self.observation_space = self.comm_env.observation_space
        self._action_space_in_preferred_format = True
        self.action_space = self.comm_env.action_space
        self.aggregate_actions_steps = 4

        super().__init__()

    def reset(
        self,
        seed: Optional[int] = None,
        options: dict = {"initial_episode": -1},
        return_info: bool = True,
    ):
        if options is None or options == {}:
            options = {"initial_episode": -1}
        obs, _ = self.comm_env.reset(seed=seed, options=options)
        return obs, {}

    def step(self, action_dict):
        obs, rewards, terminated, truncated, info = {}, {}, {}, {}, {}
        for _ in range(self.aggregate_actions_steps):
            (
                obs,
                rewards,
                termination,
                truncation,
                info,
            ) = self.comm_env.step(action_dict)
            if termination:
                if isinstance(action_dict, dict):
                    terminated = {agent: True for agent in self.agents}
                    truncated = {agent: True for agent in self.agents}
                    terminated["__all__"], truncated["__all__"] = True, True
                else:
                    terminated, truncated = True, True
                break
            else:
                if isinstance(action_dict, dict):
                    terminated = {agent: False for agent in self.agents}
                    truncated = {agent: False for agent in self.agents}
                    terminated["__all__"], truncated["__all__"] = False, False
                else:
                    terminated = False
                    truncated = False
        # IMPORTANT: We are returning only the information for the last step
        # but we could also make an average and return
        print(
            f"Ep: {self.comm_env.episode_number}, Step: {self.comm_env.step_number}"
        )
        return obs, rewards, terminated, truncated, info

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
        obs_space: spaces.Space = spaces.Space(),
        action_space: spaces.Space = spaces.Space(),
    ):
        self.observation_space = obs_space
        self.action_space = action_space
        self.comm_env.set_agent_functions(
            obs_space_format,
            action_format,
            calculate_reward,
            obs_space,
            action_space,
        )
