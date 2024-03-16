import numpy as np
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

from agents.round_robin import RoundRobin
from agents.ssr_protect import SSRProtect
from agents.ssr_rl import SSRRL
from associations.industrial import IndustrialAssociation
from channels.quadriga import QuadrigaChannels
from mobilities.simple import SimpleMobility
from custom_env import CustomEnv
from traffics.industrial import IndustrialTraffic
import gymnasium as gym

scenarios = ["industrial"]
agents = ["ssr_protect", "ssr"]

seed = 10
for scenario in scenarios:
    for agent_name in agents:
        custom_env = CustomEnv(
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

        agent = AgentClass(
            custom_env,  # type: ignore
            custom_env.comm_env.max_number_ues,
            custom_env.comm_env.max_number_slices,
            custom_env.comm_env.max_number_basestations,
            custom_env.comm_env.num_available_rbs,
            seed=seed,
            hyperparams="ssr_protect" if agent_name == "ssr_protect" else "",
        )
        assert agent.action_format is not None, "Action format not implemented"
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

        # check_env(comm_env)
        print(f"\n\n########### Agent: {agent_name} ###########")
        print("########### TRAIN ###########")
        train_episodes = 140
        steps_per_episode = custom_env.comm_env.max_number_steps
        train_runs = 1
        total_number_steps = train_episodes * steps_per_episode * train_runs
        custom_env.comm_env.max_number_episodes = train_episodes
        agent.train(total_number_steps)

        # Test
        print("########### TEST ###########")
        custom_env.comm_env.max_number_episodes = 200
        obs = custom_env.comm_env.reset(
            seed=seed, options={"initial_episode": train_episodes}
        )[0]
        for step_number in tqdm(
            np.arange(
                custom_env.comm_env.max_number_steps
                * (custom_env.comm_env.max_number_episodes - train_episodes)
            )
        ):
            sched_decision = agent.step(obs)
            obs, _, end_ep, _, _ = custom_env.comm_env.step(sched_decision)
            if end_ep and (step_number + 1) < total_number_steps:
                custom_env.comm_env.reset()
