import functools
import jax.nn as jnn
from ss2r.algorithms.mb_ppo import train as mb_ppo
from ss2r.algorithms.penalizers import get_penalizer

def get_train_fn(cfg, checkpoint_path=None, restore_checkpoint_path=None):
    agent_cfg = dict(cfg.agent)
    training_cfg = {
        k: v
        for k, v in cfg.training.items()
        if k
        not in [
            "render_episodes",
            "train_domain_randomization",
            "eval_domain_randomization",
            "render",
            "store_checkpoint",
            "value_privileged",
            "policy_privileged",
            "wandb_id",
            "safe"
        ]
    }
    
    # Extract relevant configuration parameters
    policy_hidden_layer_sizes = agent_cfg.pop("policy_hidden_layer_sizes")
    value_hidden_layer_sizes = agent_cfg.pop("value_hidden_layer_sizes")
    hidden_layer_sizes = agent_cfg.pop("hidden_layer_sizes", (256, 256))
    activation = getattr(jnn, agent_cfg.pop("activation", "relu"))
    
    # Remove the 'name' field to prevent it from being passed to train()
    del agent_cfg["name"]
    
    # Handle privileged observations
    value_obs_key = "privileged_state" if cfg.training.get('value_privileged', False) else "state"
    policy_obs_key = "privileged_state" if cfg.training.get('policy_privileged', False) else "state"
    
    # Delete unnecessary fields
    if "value_privileged" in training_cfg:
        del training_cfg["value_privileged"]
    if "policy_privileged" in training_cfg:
        del training_cfg["policy_privileged"]
    
    # Get penalizer if configured
    penalizer, penalizer_params = None, None
    if "penalizer" in agent_cfg:
        penalizer, penalizer_params = get_penalizer(cfg)
        del agent_cfg["penalizer"]
    
    # Handle other configurations that should be removed
    if "propagation" in agent_cfg:
        del agent_cfg["propagation"]
    if "cost_robustness" in agent_cfg:
        del agent_cfg["cost_robustness"]
    if "reward_robustness" in agent_cfg:
        del agent_cfg["reward_robustness"]
    if "data_collection" in agent_cfg:
        del agent_cfg["data_collection"]
    
    # Rename learning rate parameters to match our API
    if "lr" in agent_cfg:
        agent_cfg["learning_rate"] = agent_cfg.pop("lr")
    if "critic_lr" in agent_cfg:
        agent_cfg["critic_learning_rate"] = agent_cfg.pop("critic_lr")
    if "cost_critic_lr" in agent_cfg:
        agent_cfg["cost_critic_learning_rate"] = agent_cfg.pop("cost_critic_lr")
    
    # Create train function
    train_fn = functools.partial(
        mb_ppo.train,
        **agent_cfg,
        **training_cfg,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        hidden_layer_sizes=hidden_layer_sizes,
        penalizer=penalizer,
        penalizer_params=penalizer_params,
    )
    
    return train_fn