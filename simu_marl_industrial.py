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

from associations.industrial import IndustrialAssociation
from channels.quadriga import QuadrigaChannels
from channels.simple import SimpleChannel
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt.sixg_radio_mgmt.mobility import Mobility
from traffics.industrial import IndustrialTraffic
from marl_custom_env import MARLCustomEnv
from agents.marl_safe import MARLSafe

read_checkpoint = str(Path("./ray_results/").resolve())
training_flag = True  # False for reading from checkpoint
debug_mode = (
    True  # When true executes in a local mode where GPU cannot be used
)
agent = "marl_safe"
env_config = {
    "seed": 10,
    "seed_test": 15,
    "agent_class": MARLSafe,
    "channel_class": SimpleChannel,  # QuadrigaChannels,
    "traffic_class": IndustrialTraffic,
    "mobility_class": SimpleMobility,
    "association_class": IndustrialAssociation,
    "scenario": "industrial",
    "root_path": str(getcwd()),
    "training_episodes": 140,
    "max_episode_number": 140,
    "training_epochs": 1,
    "testing_episodes": 30,  # TODO 1000,
}

ray.init(local_mode=debug_mode)


def env_creator(env_config):
    marl_custom = MARLCustomEnv(
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
        marl_custom,
        marl_custom.comm_env.max_number_ues,
        marl_custom.comm_env.max_number_slices,
        marl_custom.comm_env.max_number_basestations,
        marl_custom.comm_env.num_available_rbs,
    )
    marl_custom.set_agent_functions(
        agent.obs_space_format,
        agent.action_format,
        agent.calculate_reward,
        agent.get_obs_space(),
        agent.get_action_space(),
    )

    return marl_custom


# Ray RLlib
register_env("marl_custom", lambda config: env_creator(config))


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    agent_idx = int(agent_id.partition("_")[2])

    return "inter_slice_sched" if agent_idx == 0 else "intra_slice_sched"


env_config["agent"] = agent

# Training
if training_flag:
    algo_config = (
        PPOConfig()  # TODO
        .environment(
            env="marl_custom",
            env_config=env_config,
            is_atari=False,
            disable_env_checking=True,
        )
        .multi_agent(
            policies={
                "inter_slice_sched": PolicySpec(),
                "intra_slice_sched": PolicySpec(),
            },
            policy_mapping_fn=policy_mapping_fn,
            count_steps_by="env_steps",
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=0,
            num_envs_per_worker=1,
            preprocessor_pref=None,
        )
        .resources(
            num_gpus=1,
            num_gpus_per_worker=1,
            num_gpus_per_learner_worker=1,
        )
        .training(
            lr=0.0003,  # SB3 LR
            train_batch_size=2048,  # SB3 n_steps
            sgd_minibatch_size=64,  # type: ignore SB3 batch_size
            num_sgd_iter=10,  # type: ignore SB3 n_epochs
            gamma=0.99,  # SB3 gamma
            lambda_=0.95,  # type: ignore # SB3 gae_lambda
            clip_param=0.2,  # type: ignore SB3 clip_range,
            vf_clip_param=np.inf,  # type: ignore SB3 equivalent to clip_range_vf=None
            use_gae=True,  # type: ignore SB3 normalize_advantage
            entropy_coeff=0.01,  # type: ignore SB3 ent_coef
            vf_loss_coeff=0.5,  # type: ignore SB3 vf_coef
            grad_clip=0.5,  # SB3 max_grad_norm TODO
            # kl_target=0.00001,  # SB3 target_kl
        )
        .experimental(_enable_new_api_stack=False)
        .debugging(
            seed=env_config["seed"],
        )
    )
    algo_config["model"]["fcnet_hiddens"] = [
        64,
        64,
    ]  # Set neural network size
    stop = {
        "episodes_total": env_config["training_episodes"]
        * env_config["training_epochs"],
    }
    results = tune.Tuner(
        "PPO",  # TODO
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
    ).fit()

# Testing
analysis = tune.ExperimentAnalysis(
    f"{read_checkpoint}/{env_config['scenario']}/{env_config['agent']}/"
)
assert analysis.trials is not None, "Analysis trial is None"
best_checkpoint = analysis.get_best_checkpoint(
    analysis.trials[0], "episode_reward_mean", "max"
)
assert best_checkpoint is not None, "Best checkpoint is None"
last_checkpoint = analysis.get_last_checkpoint(analysis.trials[0])
assert last_checkpoint is not None, "Last checkpoint is None"
algo = Algorithm.from_checkpoint(last_checkpoint)
marl_custom = env_creator(env_config)
marl_custom.comm_env.max_number_episodes = (
    env_config["testing_episodes"] + env_config["training_episodes"]
)
obs, _ = marl_custom.reset(
    seed=env_config["seed_test"],
    options={"initial_episode": env_config["training_episodes"]},
)
for step in tqdm(
    np.arange(
        marl_custom.comm_env.max_number_steps * env_config["testing_episodes"]
    ),
    desc="Testing...",
):
    action = {}
    assert isinstance(obs, dict), "Observation must be a dict"
    for agent_id, agent_obs in obs.items():
        policy_id = policy_mapping_fn(agent_id)
        action[agent_id] = algo.compute_single_action(
            agent_obs,
            policy_id=policy_id,
            explore=False,
        )
    obs, reward, terminated, truncated, info = marl_custom.step(action)
    assert isinstance(terminated, dict), "Termination must be a dict"
    if terminated["__all__"]:
        obs, _ = marl_custom.reset()

ray.shutdown()
