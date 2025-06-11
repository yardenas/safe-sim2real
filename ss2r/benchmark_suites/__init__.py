import functools

import jax
from brax import envs

from ss2r.algorithms.mbpo.wrappers import TrackOnlineCostsInObservation
from ss2r.algorithms.ppo.wrappers import Saute
from ss2r.benchmark_suites import brax, mujoco_playground, safety_gym
from ss2r.benchmark_suites.brax.ant import ant
from ss2r.benchmark_suites.brax.cartpole import cartpole
from ss2r.benchmark_suites.brax.humanoid import humanoid
from ss2r.benchmark_suites.mujoco_playground.cartpole import cartpole as dm_cartpole
from ss2r.benchmark_suites.mujoco_playground.go1_joystick import go1_joystick
from ss2r.benchmark_suites.mujoco_playground.humanoid import humanoid as dm_humanoid
from ss2r.benchmark_suites.mujoco_playground.quadruped import quadruped
from ss2r.benchmark_suites.mujoco_playground.walker import walker
from ss2r.benchmark_suites.rccar import rccar
from ss2r.benchmark_suites.safety_gym import go_to_goal
from ss2r.benchmark_suites.utils import get_domain_name, get_task_config
from ss2r.benchmark_suites.wrappers import (
    ActionObservationDelayWrapper,
    FrameActionStack,
    SPiDR,
    wrap,
)


def get_wrap_env_fn(cfg):
    if "propagation" not in cfg.agent:
        out = lambda env: env, lambda env: env
    elif cfg.agent.propagation.name == "spidr":

        def fn(env):
            key = jax.random.PRNGKey(cfg.training.seed)
            env = SPiDR(
                env,
                prepare_randomization_fn(
                    key,
                    cfg.agent.propagation.num_envs,
                    cfg.environment.train_params,
                    cfg.environment.task_name,
                ),
                cfg.agent.propagation.num_envs,
                cfg.agent.propagation.lambda_,
                cfg.agent.propagation.alpha,
            )
            return env

        out = fn, lambda env: env
    else:
        raise ValueError("Propagation method not provided.")
    if "penalizer" in cfg.agent and cfg.agent.penalizer.name == "saute":

        def saute_train(env):
            env = out[0](env)
            env = Saute(
                env,
                cfg.agent.safety_discounting,
                cfg.training.safety_budget,
                cfg.agent.penalizer.penalty,
                cfg.agent.penalizer.terminate,
            )
            return env

        def saute_eval(env):
            env = out[1](env)
            env = Saute(
                env,
                cfg.agent.safety_discounting,
                cfg.training.safety_budget,
                0.0,
                False,
            )
            return env

        out = saute_train, saute_eval

    if cfg.agent.name == "mbpo" and cfg.training.safe:

        def safe_mbpo_train(env):
            env = TrackOnlineCostsInObservation(env, cfg.agent.safety_discounting)
            return env

        def safe_mbpo_eval(env):
            env = TrackOnlineCostsInObservation(env, cfg.agent.safety_discounting)
            return env

        out = safe_mbpo_train, safe_mbpo_eval

    return out


def make(cfg, train_wrap_env_fn=lambda env: env, eval_wrap_env_fn=lambda env: env):
    domain_name = get_domain_name(cfg)
    if domain_name == "brax":
        return make_brax_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn)
    elif domain_name == "rccar":
        return make_rccar_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn)
    elif domain_name == "mujoco_playground":
        return make_mujoco_playground_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn)
    elif domain_name == "safety_gym":
        return make_safety_gym_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn)


def prepare_randomization_fn(key, num_envs, cfg, task_name):
    randomize_fn = lambda sys, rng: randomization_fns[task_name](sys, rng, cfg)
    v_randomization_fn = functools.partial(
        randomize_fn, rng=jax.random.split(key, num_envs)
    )
    vf_randomization_fn = lambda sys: v_randomization_fn(sys)  # type: ignore
    return vf_randomization_fn


def make_rccar_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn):
    task_cfg = dict(get_task_config(cfg))
    task_cfg.pop("domain_name")
    task_cfg.pop("task_name")
    train_car_params = task_cfg.pop("train_params")
    eval_car_params = task_cfg.pop("eval_params")
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
    train_randomization_fn = (
        prepare_randomization_fn(
            train_key,
            cfg.training.num_envs,
            train_car_params,
            cfg.environment.task_name,
        )
        if cfg.training.train_domain_randomization
        else None
    )
    train_env = wrap(
        train_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=train_randomization_fn,
        augment_state=False,
    )
    eval_env = rccar.RCCar(eval_car_params["nominal"], **task_cfg)
    eval_env = eval_wrap_env_fn(eval_env)
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
            eval_car_params,
            cfg.environment.task_name,
        )
        if cfg.training.eval_domain_randomization
        else None
    )
    eval_env = wrap(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn,
        augment_state=False,
    )
    return train_env, eval_env


def make_brax_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn):
    task_cfg = get_task_config(cfg)
    train_env = envs.get_environment(
        task_cfg.task_name, backend=cfg.environment.backend, **task_cfg.task_params
    )
    train_env = train_wrap_env_fn(train_env)
    eval_env = envs.get_environment(
        task_cfg.task_name, backend=cfg.environment.backend, **task_cfg.task_params
    )
    eval_env = eval_wrap_env_fn(eval_env)
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    train_randomization_fn = (
        prepare_randomization_fn(
            train_key, cfg.training.num_envs, task_cfg.train_params, task_cfg.task_name
        )
        if cfg.training.train_domain_randomization
        else None
    )
    train_env = wrap(
        train_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=train_randomization_fn,
        augment_state=False,
        hard_resets=cfg.training.hard_resets,
    )
    eval_randomization_fn = prepare_randomization_fn(
        eval_key, cfg.training.num_eval_envs, task_cfg.eval_params, task_cfg.task_name
    )
    eval_env = wrap(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn
        if cfg.training.eval_domain_randomization
        else None,
        augment_state=False,
    )
    return train_env, eval_env


def make_mujoco_playground_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn):
    from ml_collections import config_dict
    from mujoco_playground import registry

    from ss2r.benchmark_suites.mujoco_playground import wrap_for_brax_training

    task_cfg = get_task_config(cfg)
    task_params = config_dict.ConfigDict(task_cfg.task_params)
    train_env = registry.load(task_cfg.task_name, config=task_params)
    train_env = train_wrap_env_fn(train_env)
    eval_env = registry.load(task_cfg.task_name, config=task_params)
    eval_env = eval_wrap_env_fn(eval_env)
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
        hard_resets=cfg.training.hard_resets,
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


def make_safety_gym_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn):
    from ss2r.benchmark_suites.mujoco_playground import wrap_for_brax_training
    from ss2r.benchmark_suites.safety_gym import go_to_goal

    task_cfg = get_task_config(cfg)
    train_env = go_to_goal.GoToGoal()
    train_env = train_wrap_env_fn(train_env)
    eval_env = go_to_goal.GoToGoal()
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
        hard_resets=cfg.training.hard_resets,
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
    "Go1JoystickRoughTerrain": go1_joystick.domain_randomization,
    "SafeJointGo1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "SafeFlipGo1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "SafeJointTorqueGo1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "ant": ant.domain_randomization,
    "ant_safe": ant.domain_randomization,
    "QuadrupedWalk": quadruped.domain_randomization,
    "QuadrupedRun": quadruped.domain_randomization,
    "SafeQuadrupedWalk": quadruped.domain_randomization,
    "SafeQuadrupedRun": quadruped.domain_randomization,
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
    "HumanoidWalk": dm_humanoid.domain_randomization,
    "SafeHumanoidWalk": dm_humanoid.domain_randomization,
    "go_to_goal": go_to_goal.domain_randomization,
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
    "Go1JoystickRoughTerrain": functools.partial(
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
    "QuadrupedWalk": functools.partial(mujoco_playground.render, camera="x"),
    "QuadrupedRun": functools.partial(mujoco_playground.render, camera="x"),
    "SafeQuadrupedWalk": functools.partial(mujoco_playground.render, camera="x"),
    "SafeQuadrupedRun": functools.partial(mujoco_playground.render, camera="x"),
    "WalkerWalk": functools.partial(mujoco_playground.render, camera="side"),
    "WalkerRun": functools.partial(mujoco_playground.render, camera="side"),
    "SafeWalkerWalk": functools.partial(mujoco_playground.render, camera="side"),
    "SafeWalkerRun": functools.partial(mujoco_playground.render, camera="side"),
    "HumanoidWalk": mujoco_playground.render,
    "SafeHumanoidWalk": mujoco_playground.render,
    "go_to_goal": functools.partial(safety_gym.render, camera="fixedfar"),
}
