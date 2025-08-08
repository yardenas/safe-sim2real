import os
import traceback

import hydra
import optuna
from omegaconf import OmegaConf
from undecorated import undecorated

from train_brax import main as train_brax_main

# Define the hyperparameters to optimize and their search spaces
SEARCH_SPACE = [
    # {
    #     "name": "agent.model_hidden_layer_sizes",
    #     "type": "categorical",
    #     "choices": [
    #         [256, 256, 256],
    #         [400, 400, 400],
    #     ],
    # },
    {
        "name": "agent.min_replay_size",
        "type": "int",
        "low": 250,
        "high": 3000,
    },
    # {
    #     "name": "agent.max_replay_size",
    #     "type": "int",
    #     "low": 10000,
    #     "high": 2000000,
    # },
    {
        "name": "agent.critic_grad_updates_per_step",
        "type": "int",
        "low": 50,
        "high": 6000,
    },
    {
        "name": "agent.model_grad_updates_per_step",
        "type": "int",
        "low": 50,
        "high": 85000,
    },
    {
        "name": "agent.num_critic_updates_per_actor_update",
        "type": "int",
        "low": 1,
        "high": 20,
    },
    {
        "name": "agent.num_model_rollouts",
        "type": "int",
        "low": 50000,
        "high": 150000,
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
]


def set_nested_attr(cfg, dotted_key, value):
    """Set a nested attribute in an OmegaConf object using a dotted key."""
    OmegaConf.update(cfg, dotted_key, value, merge=False)


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
    # cfg.writers = ["jsonl", "stderr"]

    try:
        reward = train_brax_main_undecorated(cfg)
        print(f"Trial {trial.number} completed with reward: {reward}")
    except Exception as e:
        print(f"Trial failed: {e}")
        traceback.print_exc()
        reward = 0.0
    finally:
        # Ensure wandb is finished to avoid hanging processes
        import wandb

        wandb.finish()
    return -reward


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="train_brax")
def main(cfg):
    optuna_storage = getattr(cfg, "optuna_storage", None)
    study_name = cfg.get("wandb", {}).get("notes", "ss2r_optuna_study")
    if optuna_storage:
        del cfg.optuna_storage  # remove from cfg to avoid passing to train_brax_main
        optuna_storage = f"sqlite:///{optuna_storage}/{study_name}.db"
        db_path = optuna_storage[len("sqlite:///") :]

    train_brax_main_undecorated = undecorated(train_brax_main)

    def optuna_objective(trial):
        return objective(trial, train_brax_main_undecorated, cfg)

    if optuna_storage and os.path.exists(db_path):
        try:
            study = optuna.load_study(study_name=study_name, storage=optuna_storage)
            print(f"Loaded existing Optuna study '{study_name}' from {optuna_storage}")
        except Exception:
            study = optuna.create_study(
                direction="minimize", study_name=study_name, storage=optuna_storage
            )
            print(f"Created new Optuna study '{study_name}' at {optuna_storage}")
    elif optuna_storage:
        print(
            f"Storage path {optuna_storage} does not exist. Creating new Optuna study '{study_name}'."
        )
        study = optuna.create_study(
            direction="minimize", study_name=study_name, storage=optuna_storage
        )
    else:
        print(
            f"Creating new Optuna study '{study_name}' without storage since no storage path was provided."
        )
        study = optuna.create_study(direction="minimize", study_name=study_name)

    study.optimize(optuna_objective, n_trials=50, n_jobs=1)
    print("Best trial:")
    print(study.best_trial)


if __name__ == "__main__":
    main()
