import functools

import jax
from brax import envs

from ss2r.benchmark_suites.brax import randomization_fns
from ss2r.benchmark_suites.rccar import rccar
from ss2r.benchmark_suites.utils import get_domain_name, get_task_config


def make(cfg):
    domain_name = get_domain_name(cfg)
    if domain_name == "brax":
        return make_brax_envs(cfg)
    elif domain_name == "rccar":
        return make_rccar_envs(cfg)


def prepare_randomization_fn(key, num_envs, cfg, task_name):
    randomize_fn = lambda sys, rng: randomization_fns[task_name](sys, rng, cfg)
    v_randomization_fn = functools.partial(
        randomize_fn, rng=jax.random.split(key, num_envs)
    )
    vf_randomization_fn = lambda sys: v_randomization_fn(sys)[:-1]  # type: ignore
    params_fn = lambda sys: v_randomization_fn(sys)[-1]
    return vf_randomization_fn, params_fn


def make_rccar_envs(cfg):
    task_cfg = dict(get_task_config(cfg))
    task_cfg.pop("domain_name")
    task_cfg.pop("task_name")
    train_car_params = task_cfg.pop("train_car_params")
    eval_car_params = task_cfg.pop("eval_car_params")
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    train_env = rccar.RCCar(train_car_params["nominal"], **task_cfg)
    train_randomization_fn, params_fn = (
        prepare_randomization_fn(
            train_key,
            cfg.training.num_envs,
            train_car_params["bounds"],
            cfg.environment.task_name,
        )
        if cfg.training.train_domain_randomization
        else (None, None)
    )
    train_env = envs.training.wrap(
        train_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=train_randomization_fn,
    )
    eval_env = rccar.RCCar(eval_car_params["nominal"], **task_cfg)
    eval_randomization_fn, _ = prepare_randomization_fn(
        eval_key,
        cfg.training.num_eval_envs,
        eval_car_params["bounds"],
        cfg.environment.task_name,
    )
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn,
    )
    return train_env, eval_env, None


def make_brax_envs(cfg):
    task_cfg = get_task_config(cfg)
    train_env = envs.get_environment(
        task_cfg.task_name, backend=cfg.environment.backend
    )
    eval_env = envs.get_environment(task_cfg.task_name, backend=cfg.environment.backend)
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    train_randomization_fn, params_fn = (
        prepare_randomization_fn(
            train_key, cfg.training.num_envs, task_cfg, task_cfg.task_name
        )
        if cfg.training.train_domain_randomization
        else (None, None)
    )
    train_env = envs.training.wrap(
        train_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=train_randomization_fn,
    )
    eval_randomization_fn, _ = prepare_randomization_fn(
        eval_key, cfg.training.num_eval_envs, task_cfg, task_cfg.task_name
    )
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn
        if cfg.training.eval_domain_randomization
        else None,
    )
    if cfg.training.train_domain_randomization and cfg.training.privileged:
        domain_parameters = params_fn(train_env.sys)
    else:
        domain_parameters = None
    return train_env, eval_env, domain_parameters
