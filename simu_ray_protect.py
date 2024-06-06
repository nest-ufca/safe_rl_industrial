from os import getcwd
from pathlib import Path

import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from tqdm import tqdm

from agents.marl_safe import MARLSafe
from agents.ssr_protect_marl import SSRProtectMARL
from associations.industrial import IndustrialAssociation
from channels.mimic_quadriga import MimicQuadriga
from channels.quadriga import QuadrigaChannels
from channels.simple import SimpleChannel
from mobilities.simple import SimpleMobility
from traffics.industrial import IndustrialTraffic
from custom_env import CustomEnv
from sixg_radio_mgmt import CommunicationEnv
from agents.ssr_protect_ray import SSRProtectRay

read_checkpoint = str(Path("./ray_results/").resolve())
train_batch_size = 256
eps_per_iteration = train_batch_size / 1000


training_flag = True  # False for reading from checkpoint
debug_mode = (
    False  # When true executes in a local mode where GPU cannot be used
)
enable_restore = True  # Restore agent from checkpoint
agent = "ray_protect"
env_config = {
    "seed": 10,
    "seed_test": 15,
    "agent_class": SSRProtectRay,
    "channel_class": MimicQuadriga,
    "traffic_class": IndustrialTraffic,
    "mobility_class": SimpleMobility,
    "association_class": IndustrialAssociation,
    "scenario": "industrial",
    "root_path": str(getcwd()),
    "training_episodes": 70,
    "max_episode_number": 70,
    "training_epochs": 4,
    "testing_episodes": 30,
    "episode_evaluation_freq": 70,
    "number_evaluation_episodes": 30,
    "eval_initial_env_episode": 70,
}
EnvClass = CommunicationEnv

ray.init(local_mode=debug_mode)


def env_creator(env_config):
    env = EnvClass(
        ChannelClass=env_config["channel_class"],
        TrafficClass=env_config["traffic_class"],
        MobilityClass=env_config["mobility_class"],
        AssociationClass=env_config["association_class"],
        config_file=env_config["scenario"],
        agent_name=env_config["agent"],
        seed=env_config["seed"],
        root_path=env_config["root_path"],
        initial_episode_number=(
            env_config["initial_episode_number"]
            if "initial_episode_number" in env_config.keys()
            else 0
        ),
        max_episode_number=(
            env_config["max_episode_number"]
            if "max_episode_number" in env_config.keys()
            else None
        ),
    )
    agent = env_config["agent_class"](
        env,
        env.max_number_ues,
        env.max_number_slices,
        env.max_number_basestations,
        env.num_available_rbs,
    )
    env.set_agent_functions(
        agent.obs_space_format,
        agent.action_format,
        agent.calculate_reward,
        agent.get_obs_space(),
        agent.get_action_space(),
    )

    return env


# Ray RLlib
register_env("env", lambda config: env_creator(config))


env_config["agent"] = agent

# Training
if training_flag:
    algo_config = (
        SACConfig()
        .environment(
            env="env",
            env_config=env_config,
            is_atari=False,
            disable_env_checking=True,
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=0,
            num_envs_per_worker=1,
            preprocessor_pref=None,
        )
        .training(
            lr=0.0003,  # SB3 LR
            train_batch_size=train_batch_size,  # SB3 n_steps
            gamma=0.99,  # SB3 gamma
            n_step=1,  # type: ignore
            target_network_update_freq=1,  # type: ignore
            num_steps_sampled_before_learning_starts=100,  # type: ignore
        )
        .evaluation(
            evaluation_interval=np.rint(
                env_config["episode_evaluation_freq"] / eps_per_iteration
            ).astype(
                int
            ),  # Convert to iterations
            evaluation_duration=env_config["number_evaluation_episodes"],
            evaluation_duration_unit="episodes",
            evaluation_config={
                "explore": False,
                "env_config": dict(
                    env_config,
                    initial_episode_number=env_config[
                        "eval_initial_env_episode"
                    ],
                    max_episode_number=(
                        env_config["eval_initial_env_episode"]
                        + env_config["number_evaluation_episodes"]
                    ),
                ),
            },
            always_attach_evaluation_results=True,
        )
        .experimental(_enable_new_api_stack=False)
        .debugging(
            seed=env_config["seed"],
        )
        .reporting(
            min_sample_timesteps_per_iteration=train_batch_size,
        )
    )
    stop = {
        "episodes_total": env_config["training_episodes"]
        * env_config["training_epochs"],
    }
    if enable_restore and tune.Tuner.can_restore(
        f"{read_checkpoint}/{env_config['scenario']}/{env_config['agent']}/"
    ):
        tuner = tune.Tuner.restore(
            f"{read_checkpoint}/{env_config['scenario']}/{env_config['agent']}/",
            trainable="SAC",
            param_space=algo_config.to_dict(),
        )
    else:
        tuner = tune.Tuner(
            "SAC",
            param_space=algo_config.to_dict(),
            run_config=air.RunConfig(
                storage_path=f"{read_checkpoint}/{env_config['scenario']}/",
                name=env_config["agent"],
                stop=stop,
                verbose=2,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=3,
                    checkpoint_at_end=True,
                ),
            ),
        )
    results = tuner.fit()

# Testing
analysis = tune.ExperimentAnalysis(
    f"{read_checkpoint}/{env_config['scenario']}/{env_config['agent']}/"
)
assert analysis.trials is not None, "Analysis trial is None"
best_checkpoint = analysis.get_best_checkpoint(
    analysis.trials[0], "evaluation/episode_reward_mean", "max"
)
assert best_checkpoint is not None, "Best checkpoint is None"
algo = Algorithm.from_checkpoint(best_checkpoint)
env = env_creator(env_config)
env.max_number_episodes = (
    env_config["testing_episodes"] + env_config["training_episodes"]
)
env.save_hist = True  # Save metrics for test
obs, _ = env.reset(
    seed=env_config["seed_test"],
    options={"initial_episode": env_config["training_episodes"]},
)
aggregate_actions = 1
for step in tqdm(
    np.arange(
        np.floor(env.max_number_steps / aggregate_actions).astype(int)
        * env_config["testing_episodes"]
    ),
    desc="Testing...",
):
    action = {}
    assert isinstance(obs, np.ndarray), "Observation must be a Numpy array"
    action = algo.compute_single_action(
        obs,
        explore=False,
    )
    assert isinstance(action, np.ndarray), "Action must be a Numpy array"
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(terminated, bool), "Termination must be a bool"
    if terminated:
        obs, _ = env.reset()

ray.shutdown()
