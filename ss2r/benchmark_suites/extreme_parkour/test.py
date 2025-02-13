import isaacgym  # noqa
import jax
import jax.dlpack
import os
from omegaconf import OmegaConf

# import legged_gym.envs  # noqa
from legged_gym.envs.constraint_wrapper import ConstrainedLeggedRobot
from legged_gym.utils import task_registry
from bridge import ExtremeParkourBridge
from ss2r.benchmark_suites.utils import get_task_config

# Load the configuration file
config_path = os.path.expanduser('~/safe/safe-sim2real/configs/environment/extremeparkour.yaml')
cfg = OmegaConf.load(config_path)

task_cfg = get_task_config(cfg)
args = task_cfg
env, _ = task_registry.make_env(name=args.task, args=args)
env = ConstrainedLeggedRobot(env)
env = ExtremeParkourBridge(env)
rng = jax.random.PRNGKey(cfg.training.seed)
action = jax.random.uniform(rng, (6144, env.action_size()))
state = None

for _ in range(10):
    state = env.step(state, action)
    print(f"Observation: {state.obs}, Reward: {state.reward}, Done: {state.done}")

print("Everything is working fine")
