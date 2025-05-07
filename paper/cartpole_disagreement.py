# %%
from math import floor, log10

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn.objects as so
from hydra import compose, initialize
from matplotlib.colors import Normalize
from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from seaborn import axes_style
from tueplots import bundles, figsizes

from ss2r import benchmark_suites
from ss2r.algorithms.sac.wrappers import PTSD, ModelDisagreement


# %%
def make_config(additional_overrides=None):
    if additional_overrides is None:
        additional_overrides = []
    with initialize(version_base=None, config_path="../ss2r/configs"):
        cfg = compose(
            config_name="train_brax",
            overrides=[
                "writers=[stderr]",
                "+experiment=cartpole_swingup_sim_to_sim",
                "training.num_envs=1",
                "environment.train_params.pole_length=[-0.5, 0.5]",
                "environment.train_params.gear=[0., 10.]",
            ]
            + additional_overrides,
        )
        return cfg


cfg = make_config()


def make_env():
    env = registry.load("CartpoleSwingup")
    key = jax.random.PRNGKey(cfg.training.seed)
    env = PTSD(
        env,
        benchmark_suites.prepare_randomization_fn(
            key,
            cfg.agent.propagation.num_envs,
            cfg.environment.train_params,
            cfg.environment.task_name,
        ),
        cfg.agent.propagation.num_envs,
    )
    env = ModelDisagreement(env)
    return env


# %%

theta = np.linspace(-np.pi, np.pi, 50)
theta_dot = np.linspace(-2 * np.pi, 2 * np.pi, 50)
grid = np.array(np.meshgrid(theta, theta_dot))
grid_shape = grid.shape


@jax.jit
def generate_data(action):
    # Create a grid of all combinations of theta and theta_dot
    x = 0  # Fixed value
    x_dot = 0
    flat_states = grid.reshape(2, -1).T  # Shape: (N, 3), where N = 100^3
    q = np.zeros((flat_states.shape[0], 2))
    q[:, 0] = x  # Set x to 0 for all states
    q[:, 1] = flat_states[:, 0]  # theta
    # Set qvel with the rest of the dimensions
    qvel = np.zeros((flat_states.shape[0], 2))
    qvel[:, 0] = x_dot
    qvel[:, 1] = flat_states[:, 1]
    env = make_env()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    init = lambda q, qvel: mjx_env.init(env.mjx_model, qpos=q, qvel=qvel)
    data = jax.jit(jax.vmap(init))(q, qvel)
    obs = lambda data: env._get_obs(data, state.info)
    obs = jax.jit(jax.vmap(obs))(data)
    # Tile dimensions to match flat_states.shape[0]
    dummy_state = jax.tree_map(
        lambda x: jnp.tile(x[None], (flat_states.shape[0],) + (1,) * x.ndim), state
    )
    state = dummy_state.replace(data=data)
    step = lambda state: env.step(state, action)
    state = jax.jit(jax.vmap(step))(state)
    return state.info["disagreement"]


actions = np.linspace(0, 1, 4)[:, None]
value_matrices = [generate_data(action) for action in actions]


# %%
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r"${0:.{2}f}\times10^{{{1:d}}}$".format(coeff, exponent, precision)


def sci_format(val):
    if val == 0:
        return "$0$"
    exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    coeff = val / (10**exp)
    if coeff == 1:
        return rf"$10^{{{exp}}}$"
    return rf"${{{coeff:.0f} \times 10^{{{exp}}}}}$"


# %%
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(
    figsizes.neurips2024(nrows=1, ncols=4, pad_inches=0.050, height_to_width_ratio=1.0)
)
fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
vmin = np.min([v.min() for v in value_matrices])
vmax = np.max([v.max() for v in value_matrices])
norm = Normalize(vmin=vmin, vmax=vmax)
# Plot each subplot
for i, (ax, a, val) in enumerate(zip(axes, actions, value_matrices)):
    cp = ax.contourf(
        theta,
        theta_dot,
        value_matrices[i].reshape(grid_shape[1:]),
        50,
        cmap="rocket",
        norm=norm,
    )
    cp.set_edgecolor("face")
    ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_title(f"$a = {a[0]:.1f}$")
# Add one colorbar on the right
cbar = fig.colorbar(
    cp, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.01
)
cbar.set_label("Uncertainty")

vmin, vmax = cp.get_clim()
vmid = (vmin + vmax) / 2
# Set ticks to show only the min, mid, and max values
cbar.set_ticks([vmin, vmid, vmax])
cbar.set_ticklabels([sci_format(vmin), sci_format(vmid), sci_format(vmax)])
axes[0].set_ylabel(r"$\dot{\theta}$")
fig.supxlabel(r"$\theta$")
fig.savefig("cartpole-disagreement.pdf")
