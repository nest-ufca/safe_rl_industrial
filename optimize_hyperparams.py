from typing import Any, Dict

import gymnasium as gym
import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.round_robin import RoundRobin
from agents.ssr_protect import SSRProtect
from agents.ssr_rl import SSRRL
from associations.industrial import IndustrialAssociation
from channels.quadriga import QuadrigaChannels
from mobilities.simple import SimpleMobility
from sixg_radio_mgmt import CommunicationEnv
from traffics.industrial import IndustrialTraffic

N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 5
N_TIMESTEPS = 1e4
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 1
SEED = 10
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
}


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(1e5), int(1e6)]
    )
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1000, 10000, 20000]
    )
    train_freq = trial.suggest_categorical(
        "train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    # Polyak coeff
    tau = trial.suggest_categorical(
        "tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08]
    )
    gradient_steps = train_freq
    ent_coef = "auto"
    net_arch = trial.suggest_categorical(
        "net_arch", ["small", "medium", "big"]
    )

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
    }[net_arch]

    target_entropy = "auto"

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    return hyperparams


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(float(self.last_mean_reward), self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    env, agent = create_env()
    env = Monitor(env)
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(sample_sac_params(trial))
    model = SAC(env=env, seed=SEED, **kwargs)
    model.set_random_seed(SEED)
    eval_callback = TrialEvalCallback(
        env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except Exception as e:
        print(e)
        nan_encountered = True

    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return float(eval_callback.last_mean_reward)


def create_env():
    scenario = "industrial"
    agent = "ssr_protect"
    comm_env = CommunicationEnv(
        QuadrigaChannels,
        IndustrialTraffic,
        SimpleMobility,
        IndustrialAssociation,
        scenario,
        agent,
        seed=SEED,
        obs_space=SSRRL.get_obs_space,
        action_space=SSRRL.get_action_space,
    )
    AgentClass = SSRProtect
    agent = AgentClass(
        comm_env,
        comm_env.max_number_ues,
        comm_env.max_number_basestations,
        comm_env.num_available_rbs,
        seed=SEED,
    )
    comm_env.set_agent_functions(
        agent.obs_space_format,
        agent.action_format,
        agent.calculate_reward,
    )

    return (comm_env, agent)


if __name__ == "__main__":
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=SEED)
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=N_EVALUATIONS // 3,
    )

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        study_name="ssr_protect",
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

    joblib.dump(
        study,
        "./hyperparameter_opt/ssr_protect.pkl",
    )
