import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

from agents.round_robin import RoundRobin
from agents.ssr_protect import SSRProtect
from agents.ssr_rl import SSRRL
from associations.industrial import IndustrialAssociation
from channels.mimic_quadriga import MimicQuadriga
from channels.quadriga import QuadrigaChannels
from custom_env import CustomEnv
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import CommunicationEnv
from traffics.industrial import IndustrialTraffic

scenarios = ["industrial"]
agents = ["ssr_protect", "ssr"]
# agents = ["ssr"]
env_type = "4step"  # option "simple" uses 1 step in the environment per agent step and "4step" uses 4 steps per agent step
EnvClass = CustomEnv if env_type == "4step" else CommunicationEnv

seed = 10
for scenario in scenarios:
    for agent_name in agents:
        custom_env = EnvClass(
            ChannelClass=QuadrigaChannels,
            TrafficClass=IndustrialTraffic,
            MobilityClass=SimpleMobility,
            AssociationClass=IndustrialAssociation,
            config_file=scenario,
            agent_name=agent_name,
            seed=seed,
            obs_space=(
                SSRRL.get_obs_space()
                if agent_name in agents
                else gym.spaces.Space()
            ),
            action_space=SSRRL.get_action_space(),
        )

        match agent_name:
            case "ssr":
                AgentClass = SSRRL
            case "ssr_protect":
                AgentClass = SSRProtect
            case _:
                raise Exception("Agent not implemented")
        if isinstance(custom_env, CustomEnv):
            agent = AgentClass(
                custom_env,  # type: ignore
                custom_env.comm_env.max_number_ues,
                custom_env.comm_env.max_number_slices,
                custom_env.comm_env.max_number_basestations,
                custom_env.comm_env.num_available_rbs,
                seed=seed,
                hyperparams=(
                    "ssr_protect" if agent_name == "ssr_protect" else ""
                ),
            )
            assert (
                agent.action_format is not None
            ), "Action format not implemented"
            custom_env.comm_env.set_agent_functions(
                obs_space_format=agent.obs_space_format,
                action_format=agent.action_format,
                calculate_reward=agent.calculate_reward,
                obs_space=(
                    SSRRL.get_obs_space()
                    if agent_name in agents
                    else gym.spaces.Space()
                ),
                action_space=SSRRL.get_action_space(),
            )
        else:
            agent = AgentClass(
                custom_env,
                custom_env.max_number_ues,
                custom_env.max_number_slices,
                custom_env.max_number_basestations,
                custom_env.num_available_rbs,
                seed=seed,
                hyperparams=(
                    "ssr_protect" if agent_name == "ssr_protect" else ""
                ),
            )
            custom_env.set_agent_functions(
                obs_space_format=agent.obs_space_format,
                action_format=agent.action_format,
                calculate_reward=agent.calculate_reward,
                obs_space=(
                    SSRRL.get_obs_space()
                    if agent_name in agents
                    else gym.spaces.Space()
                ),
                action_space=SSRRL.get_action_space(),
            )

        # check_env(comm_env)
        print(f"\n\n########### Agent: {agent_name} ###########")
        print("########### TRAIN ###########")
        train_episodes = 140
        max_number_steps = (
            custom_env.comm_env.max_number_steps
            if isinstance(custom_env, CustomEnv)
            else custom_env.max_number_steps
        )
        aggregate_steps = (
            custom_env.aggregate_actions_steps
            if isinstance(custom_env, CustomEnv)
            else 1
        )
        steps_per_episode = np.floor(
            max_number_steps / aggregate_steps
        ).astype(int)
        train_runs = 2
        total_number_steps = train_episodes * steps_per_episode * train_runs
        if isinstance(custom_env, CustomEnv):
            custom_env.comm_env.max_number_episodes = train_episodes
        else:
            custom_env.max_number_episodes = train_episodes
        agent.train(total_number_steps)

        # Test
        print("########### TEST ###########")
        max_test_episodes = 200
        if isinstance(custom_env, CustomEnv):
            custom_env.comm_env.max_number_episodes = max_test_episodes
            obs = custom_env.comm_env.reset(
                seed=seed, options={"initial_episode": train_episodes}
            )[0]
        else:
            custom_env.max_number_episodes = max_test_episodes
            obs = custom_env.reset(
                seed=seed, options={"initial_episode": train_episodes}
            )[0]
        for step_number in tqdm(
            np.arange(steps_per_episode * (max_test_episodes - train_episodes))
        ):
            sched_decision = agent.step(obs)
            obs, _, end_ep, _, _ = custom_env.step(sched_decision)
            if end_ep and (step_number + 1) < total_number_steps:
                custom_env.reset()
