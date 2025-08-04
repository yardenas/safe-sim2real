import sys

import hydra
import optuna
from omegaconf import OmegaConf
from undecorated import undecorated

from ss2r.common.logging import TrainingLogger
from train_brax import main as train_brax_main

# Define the hyperparameters to optimize and their search spaces
SEARCH_SPACE = [
    {
        "name": "agent.min_replay_size",
        "type": "int",
        "low": 250,
        "high": 3000,
    },
    {
        "name": "agent.max_replay_size",
        "type": "int",
        "low": 10000,
        "high": 2000000,
    },
    {
        "name": "agent.critic_grad_updates_per_step",
        "type": "int",
        "low": 1000,
        "high": 10000,
    },
    {
        "name": "agent.model_grad_updates_per_step",
        "type": "int",
        "low": 10000,
        "high": 200000,
    },
    {
        "name": "agent.num_model_rollouts",
        "type": "int",
        "low": 10000,
        "high": 200000,
    },
    {
        "name": "agent.critic_learning_rate",
        "type": "loguniform",
        "low": 1e-9,
        "high": 1e-2,
    },
    {
        "name": "agent.learning_rate",
        "type": "loguniform",
        "low": 1e-9,
        "high": 1e-2,
    },
    {
        "name": "agent.model_hidden_layer_sizes",
        "type": "categorical",
        "choices": [
            [512, 512],
            [256, 256, 256],
        ],
    },
    {
        "name": "agent.num_critic_updates_per_actor_update",
        "type": "int",
        "low": 1,
        "high": 20,
    },
]


def set_nested_attr(cfg, dotted_key, value):
    """Set a nested attribute in an OmegaConf object using a dotted key."""
    keys = dotted_key.split(".")
    obj = cfg
    for k in keys[:-1]:
        obj = obj[k]
    obj[keys[-1]] = value


def run_train_brax_with_overrides(overrides):
    # Compose config and run train_brax.main, return metrics
    from train_brax import main as train_brax_main

    # Hydra requires sys.argv to be set for overrides
    orig_argv = sys.argv
    sys.argv = [orig_argv[0]] + overrides
    try:
        # train_brax.main returns metrics dict
        metrics = train_brax_main()
    finally:
        sys.argv = orig_argv
    return metrics


def objective(trial, train_brax_main_undecorated, base_cfg):
    # Suggest hyperparameters from SEARCH_SPACE
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))  # Deep copy
    param_names = set()
    param_str = ""
    for param in SEARCH_SPACE:
        name = param["name"]
        param_names.add(name)
        if param["type"] == "loguniform":
            value = trial.suggest_loguniform(name, param["low"], param["high"])
        elif param["type"] == "uniform":
            value = trial.suggest_uniform(name, param["low"], param["high"])
        elif param["type"] == "int":
            value = trial.suggest_int(name, param["low"], param["high"])
        elif param["type"] == "categorical":
            value = trial.suggest_categorical(name, param["choices"])
        else:
            raise ValueError(f"Unknown param type: {param['type']}")
        set_nested_attr(cfg, name, value)
        param_str += f",{name}={value}"

    # Set unique wandb.name
    cfg.wandb.name += param_str + f",optuna_trial={trial.number}"
    # Disable wandb writer in cfg for individual runs
    cfg.writers = ["jsonl", "stderr"]

    # Run train_brax and get metrics
    try:
        metrics = train_brax_main_undecorated(cfg)
        reward = metrics.get("eval/episode_reward") or (
            metrics.get("eval", {}).get("episode_reward")
            if isinstance(metrics.get("eval"), dict)
            else None
        )
        if reward is None:
            reward = metrics.get("eval", {}).get("episode_reward")
        if reward is None:
            print("Warning: eval/episode_reward not found in metrics, using 0.0")
            reward = 0.0
    except Exception as e:
        print(f"Trial failed: {e}")
        reward = 0.0

    # Log config and objective value to wandb for this trial
    # log_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    # log_cfg.writers = ["wandb"]
    # log_cfg.wandb.name += ",optuna_trial_log"
    # logger = TrainingLogger(log_cfg)
    # log_data = {f"optuna/trial_{k}": v for k, v in trial.params.items()}
    # log_data["optuna/trial_value"] = reward
    # logger.log(log_data, step=trial.number)

    # Optuna maximizes reward, so return -reward for minimization
    return -reward


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="train_brax")
def main(cfg):
    train_brax_main_undecorated = undecorated(train_brax_main)

    def optuna_objective(trial):
        return objective(trial, train_brax_main_undecorated, cfg)

    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=50, n_jobs=3)
    print("Best trial:")
    print(study.best_trial)

    # Log best trial to wandb
    best_trial = study.best_trial
    best_log_data = {f"optuna/best_{k}": v for k, v in best_trial.params.items()}
    best_log_data["optuna/best_value"] = best_trial.value
    # Use TrainingLogger with wandb for final logging
    cfg.wandb.name += ",optuna_best_trial"
    cfg.writers = ["wandb"]
    logger = TrainingLogger(cfg)
    logger.log(best_log_data, step=0)


if __name__ == "__main__":
    main()
