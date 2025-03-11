import functools

import jax
from brax import envs

from ss2r.benchmark_suites import brax, mujoco_playground, wrappers
from ss2r.benchmark_suites.brax.ant import ant
from ss2r.benchmark_suites.brax.cartpole import cartpole
from ss2r.benchmark_suites.brax.humanoid import humanoid
from ss2r.benchmark_suites.mujoco_playground.cartpole import cartpole as dm_cartpole
from ss2r.benchmark_suites.mujoco_playground.go1_joystick import go1_joystick
from ss2r.benchmark_suites.mujoco_playground.walker import walker
from ss2r.benchmark_suites.rccar import rccar
from ss2r.benchmark_suites.utils import get_domain_name, get_task_config
from ss2r.benchmark_suites.wrappers import (
    ActionObservationDelayWrapper,
    FrameActionStack,
)


def make(cfg, train_wrap_env_fn=lambda env: env):
    domain_name = get_domain_name(cfg)
    if domain_name == "brax":
        return make_brax_envs(cfg, train_wrap_env_fn)
    elif domain_name == "rccar":
        return make_rccar_envs(cfg, train_wrap_env_fn)
    elif domain_name == "mujoco_playground":
        return make_mujoco_playground_envs(cfg, train_wrap_env_fn)


def prepare_randomization_fn(key, num_envs, cfg, task_name):
    randomize_fn = lambda sys, rng: randomization_fns[task_name](sys, rng, cfg)
    v_randomization_fn = functools.partial(
        randomize_fn, rng=jax.random.split(key, num_envs)
    )
    vf_randomization_fn = lambda sys: v_randomization_fn(sys)  # type: ignore
    return vf_randomization_fn


def make_rccar_envs(cfg, train_wrap_env_fn):
    task_cfg = dict(get_task_config(cfg))
    task_cfg.pop("domain_name")
    task_cfg.pop("task_name")
    train_car_params = task_cfg.pop("train_car_params")
    eval_car_params = task_cfg.pop("eval_car_params")
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    action_delay, obs_delay = (
        task_cfg.pop("action_delay"),
        task_cfg.pop("observation_delay"),
    )
    sliding_window = task_cfg.pop("sliding_window")
    train_env = rccar.RCCar(train_car_params["nominal"], **task_cfg)
    train_env = train_wrap_env_fn(train_env)
    if action_delay > 0 or obs_delay > 0:
        train_env = ActionObservationDelayWrapper(
            train_env, action_delay=action_delay, obs_delay=obs_delay
        )
    if sliding_window > 0:
        train_env = FrameActionStack(train_env, num_stack=sliding_window)
    # FIXME (yarden): train_car_params should be instead the same as the rest of the environment types
    train_randomization_fn = (
        prepare_randomization_fn(
            train_key,
            cfg.training.num_envs,
            train_car_params["bounds"],
            cfg.environment.task_name,
        )
        if cfg.training.train_domain_randomization
        else None
    )
    train_env = wrappers.wrap(
        train_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=train_randomization_fn,
    )
    eval_env = rccar.RCCar(eval_car_params["nominal"], **task_cfg)
    if action_delay > 0 or obs_delay > 0:
        eval_env = ActionObservationDelayWrapper(
            eval_env, action_delay=action_delay, obs_delay=obs_delay
        )
    if sliding_window > 0:
        eval_env = FrameActionStack(eval_env, num_stack=sliding_window)
    eval_randomization_fn = (
        prepare_randomization_fn(
            eval_key,
            cfg.training.num_eval_envs,
            eval_car_params["bounds"],
            cfg.environment.task_name,
        )
        if cfg.training.eval_domain_randomization
        else None
    )
    eval_env = wrappers.wrap(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn,
        augment_state=cfg.training.train_domain_randomization,
    )
    return train_env, eval_env


def make_brax_envs(cfg, train_wrap_env_fn):
    task_cfg = get_task_config(cfg)
    train_env = envs.get_environment(
        task_cfg.task_name, backend=cfg.environment.backend, **task_cfg.task_params
    )
    train_env = train_wrap_env_fn(train_env)
    eval_env = envs.get_environment(
        task_cfg.task_name, backend=cfg.environment.backend, **task_cfg.task_params
    )
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    train_randomization_fn = (
        prepare_randomization_fn(
            train_key, cfg.training.num_envs, task_cfg.train_params, task_cfg.task_name
        )
        if cfg.training.train_domain_randomization
        else None
    )
    train_env = wrappers.wrap(
        train_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=train_randomization_fn,
    )
    eval_randomization_fn = prepare_randomization_fn(
        eval_key, cfg.training.num_eval_envs, task_cfg.eval_params, task_cfg.task_name
    )
    eval_env = wrappers.wrap(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn
        if cfg.training.eval_domain_randomization
        else None,
        augment_state=cfg.training.train_domain_randomization,
    )
    return train_env, eval_env


def make_mujoco_playground_envs(cfg, train_wrap_env_fn):
    from ml_collections import config_dict
    from mujoco_playground import registry

    from ss2r.benchmark_suites.mujoco_playground import wrap_for_brax_training

    task_cfg = get_task_config(cfg)
    task_params = config_dict.ConfigDict(task_cfg.task_params)
    train_env = registry.load(task_cfg.task_name, config=task_params)
    train_env = train_wrap_env_fn(train_env)
    eval_env = registry.load(task_cfg.task_name, config=task_params)
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    train_randomization_fn = (
        prepare_randomization_fn(
            train_key, cfg.training.num_envs, task_cfg.train_params, task_cfg.task_name
        )
        if cfg.training.train_domain_randomization
        else None
    )
    train_env = wrap_for_brax_training(
        train_env,
        randomization_fn=train_randomization_fn,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        augment_state=False,
    )
    eval_randomization_fn = (
        prepare_randomization_fn(
            eval_key,
            cfg.training.num_eval_envs,
            task_cfg.eval_params,
            task_cfg.task_name,
        )
        if cfg.training.eval_domain_randomization
        else None
    )
    eval_env = wrap_for_brax_training(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn,
        augment_state=False,
    )
    return train_env, eval_env


randomization_fns = {
    "cartpole": cartpole.domain_randomization,
    "cartpole_safe": cartpole.domain_randomization,
    "rccar": rccar.domain_randomization,
    "humanoid": humanoid.domain_randomization,
    "humanoid_safe": humanoid.domain_randomization,
    "Go1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "SafeJointGo1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "SafeFlipGo1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "SafeJointTorqueGo1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "ant": ant.domain_randomization,
    "ant_safe": ant.domain_randomization,
    "WalkerWalk": walker.domain_randomization,
    "WalkerRun": walker.domain_randomization,
    "SafeWalkerWalk": walker.domain_randomization,
    "SafeWalkerRun": walker.domain_randomization,
    "SafeCartpoleSwingup": dm_cartpole.domain_randomization,
    "SafeCartpoleSwingupSparse": dm_cartpole.domain_randomization,
    "SafeCartpoleBalanceSparse": dm_cartpole.domain_randomization,
    "SafeCartpoleBalance": dm_cartpole.domain_randomization,
    "CartpoleSwingup": dm_cartpole.domain_randomization,
    "CartpoleSwingupSparse": dm_cartpole.domain_randomization,
    "CartpoleBalanceSparse": dm_cartpole.domain_randomization,
    "CartpoleBalance": dm_cartpole.domain_randomization,
}

render_fns = {
    "cartpole": brax.render,
    "cartpole_safe": brax.render,
    "humanoid": functools.partial(brax.render, camera="track"),
    "humanoid_safe": functools.partial(brax.render, camera="track"),
    "ant": functools.partial(brax.render, camera="track"),
    "ant_safe": functools.partial(brax.render, camera="track"),
    "rccar": rccar.render,
    "Go1JoystickFlatTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "SafeJointGo1JoystickFlatTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "SafeFlipGo1JoystickFlatTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "SafeJointTorqueGo1JoystickFlatTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "SafeCartpoleSwingup": functools.partial(mujoco_playground.render),
    "SafeCartpoleSwingupSparse": functools.partial(mujoco_playground.render),
    "SafeCartpoleBalanceSparse": functools.partial(mujoco_playground.render),
    "SafeCartpoleBalance": functools.partial(mujoco_playground.render),
    "CartpoleSwingup": functools.partial(mujoco_playground.render),
    "CartpoleSwingupSparse": functools.partial(mujoco_playground.render),
    "CartpoleBalanceSparse": functools.partial(mujoco_playground.render),
    "CartpoleBalance": functools.partial(mujoco_playground.render),
    "WalkerWalk": functools.partial(mujoco_playground.render, camera="side"),
    "WalkerRun": functools.partial(mujoco_playground.render, camera="side"),
    "SafeWalkerWalk": functools.partial(mujoco_playground.render, camera="side"),
    "SafeWalkerRun": functools.partial(mujoco_playground.render, camera="side"),
}
