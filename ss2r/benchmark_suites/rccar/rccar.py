import functools

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from brax.envs.base import Env, State
from omegaconf import OmegaConf

from ss2r.benchmark_suites import rewards
from ss2r.benchmark_suites.rccar.hardware import HardwareDynamics
from ss2r.benchmark_suites.rccar.model import CarParams, RaceCarDynamics
from ss2r.rl.utils import rollout

OBS_NOISE_STD_SIM_CAR: jnp.array = 0.1 * jnp.exp(
    jnp.array([-4.5, -4.5, -4.0, -2.5, -2.5, -1.0])
)

X_LIM = (-0.8, 2.5)
Y_LIM = (-1.8, 1.8)


def domain_randomization(sys, rng, cfg):
    def sample_from_bounds(value, key):
        """
        Sample from a JAX uniform distribution if the value is a list of two elements.
        """
        if isinstance(value, list) and len(value) == 2:
            lower, upper = value
            # Sample from jax.random.uniform with the given key
            return jax.random.uniform(key, shape=(), minval=lower, maxval=upper)
        return value

    @jax.vmap
    def randomize(rng):
        bounds = CarParams(**cfg)
        # Define a custom tree structure that treats lists as leaves
        treedef = jtu.tree_structure(bounds, is_leaf=lambda x: isinstance(x, list))
        # Generate random keys only for the relevant leaves (i.e., lists with 2 elements)
        keys = jax.random.split(rng, num=treedef.num_leaves)
        # Rebuild the tree with the keys, only where there are valid leaves
        keys = jtu.tree_unflatten(treedef, keys)
        # Map over the tree, generating random values where needed
        sys = jtu.tree_map(
            sample_from_bounds, bounds, keys, is_leaf=lambda x: isinstance(x, list)
        )
        return sys, jax.flatten_util.ravel_pytree(sys)[0]

    cfg = OmegaConf.to_container(cfg)
    in_axes = jax.tree_map(lambda _: 0, sys)
    sys, params = randomize(rng)
    return sys, in_axes, params


def rotate_coordinates(state: jnp.array, encode_angle: bool = False) -> jnp.array:
    x_pos, x_vel = (
        state[..., 0:1],
        state[..., 3 + int(encode_angle) : 4 + int(encode_angle)],
    )
    y_pos, y_vel = (
        state[..., 1:2],
        state[:, 4 + int(encode_angle) : 5 + int(encode_angle)],
    )
    theta = state[..., 2 : 3 + int(encode_angle)] - jnp.pi / 2
    new_state = jnp.concatenate(
        [y_pos, -x_pos, theta, y_vel, -x_vel, state[..., 5 + int(encode_angle) :]],
        axis=-1,
    )
    assert state.shape == new_state.shape
    return new_state


def encode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """Encodes the angle (theta) as sin(theta) ant cos(theta)"""
    assert angle_idx <= state.shape[-1] - 1
    theta = state[..., angle_idx : angle_idx + 1]
    state_encoded = jnp.concatenate(
        [
            state[..., :angle_idx],
            jnp.sin(theta),
            jnp.cos(theta),
            state[..., angle_idx + 1 :],
        ],
        axis=-1,
    )
    assert state_encoded.shape[-1] == state.shape[-1] + 1
    return state_encoded


def decode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """Decodes the angle (theta) from sin(theta) and cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = jnp.arctan2(
        state[..., angle_idx : angle_idx + 1], state[..., angle_idx + 1 : angle_idx + 2]
    )
    state_decoded = jnp.concatenate(
        [state[..., :angle_idx], theta, state[..., angle_idx + 2 :]], axis=-1
    )
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded


class RCCarEnvReward:
    def __init__(
        self,
        goal: jax.Array,
        encode_angle: bool = False,
        ctrl_cost_weight: float = 0.005,
        bound: float = 0.1,
        margin_factor: float = 10.0,
    ):
        self._angle_idx = 2
        self.dim_action = (2,)
        self.goal = goal
        self.ctrl_cost_weight = ctrl_cost_weight
        self.encode_angle = encode_angle
        self.tolerance_reward = functools.partial(
            rewards.tolerance,
            bounds=(0.0, bound),
            margin=margin_factor * bound,
            value_at_margin=0.1,
            sigmoid="long_tail",
        )

    def forward(self, obs: jax.Array, action: jax.Array, next_obs: jax.Array):
        """Computes the reward for the given transition"""
        reward_ctrl = self.action_reward(action)
        reward_state = self.state_reward(obs, next_obs)
        reward = reward_state + self.ctrl_cost_weight * reward_ctrl
        return reward

    @staticmethod
    def action_reward(action: jax.Array) -> jax.Array:
        """Computes the reward/penalty for the given action"""
        return -(action**2).sum(-1)

    def state_reward(self, obs: jax.Array, next_obs: jax.Array) -> jax.Array:
        """Computes the reward for the given observations"""
        if self.encode_angle:
            next_obs = decode_angles(next_obs, angle_idx=self._angle_idx)
        pos_diff = next_obs[..., :2] - self.goal[:2]
        theta_diff = next_obs[..., 2] - self.goal[2]
        pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_diff), axis=-1))
        theta_dist = jnp.abs(((theta_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        total_dist = jnp.sqrt(pos_dist**2 + theta_dist**2)
        reward = self.tolerance_reward(total_dist)
        return reward


def cost_fn(xy, obstacle_position, obstacle_radius) -> jax.Array:
    x, y = xy[..., 0], xy[..., 1]
    distance = jnp.linalg.norm(xy - obstacle_position)
    obstacle = jnp.where(distance >= obstacle_radius, 0.0, 1.0)
    in_bounds = lambda x, lower, upper: jnp.where((x >= lower) & (x <= upper), 0.0, 1.0)
    in_x = in_bounds(x, *X_LIM)
    in_y = in_bounds(y, *Y_LIM)
    return obstacle + in_x + in_y


class RCCar(Env):
    def __init__(
        self,
        car_model_params: dict,
        ctrl_cost_weight: float = 0.005,
        encode_angle: bool = True,
        use_obs_noise: bool = False,
        margin_factor: float = 10.0,
        max_throttle: float = 1.0,
        dt: float = 1 / 30.0,
        obstacle: tuple[float, float, float] = (0.75, -0.75, 0.2),
        *,
        hardware: HardwareDynamics | None = None,
    ):
        self.goal = jnp.array([0.0, 0.0, 0.0])
        self.obstacle = tuple(obstacle)
        self.init_pose = jnp.array([1.42, -1.04, jnp.pi])
        self.angle_idx = 2
        self._obs_noise_stds = OBS_NOISE_STD_SIM_CAR
        self.dim_action = (2,)
        self.dim_state = (7,) if encode_angle else (6,)
        self.encode_angle = encode_angle
        self.max_throttle = jnp.clip(max_throttle, 0.0, 1.0)
        self.dynamics_model: RaceCarDynamics | HardwareDynamics = (
            RaceCarDynamics(dt=dt) if hardware is None else hardware
        )
        self.sys = CarParams(**car_model_params)
        self.use_obs_noise = use_obs_noise
        self.reward_model = RCCarEnvReward(
            goal=self.goal,
            ctrl_cost_weight=ctrl_cost_weight,
            encode_angle=self.encode_angle,
            margin_factor=margin_factor,
        )

    def _obs(self, state: jnp.array, rng: jax.random.PRNGKey) -> jnp.array:
        """Adds observation noise to the state"""
        assert state.shape[-1] == 6
        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * jax.random.normal(
                rng, shape=self.dim_state
            )
        else:
            obs = state
        if self.encode_angle:
            obs = encode_angles(obs, self.angle_idx)
        assert (obs.shape[-1] == 7 and self.encode_angle) or (
            obs.shape[-1] == 6 and not self.encode_angle
        )
        return obs

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to a random initial state close to the initial pose"""
        key_pos, key_vel, key_obs = jax.random.split(rng, 3)
        if isinstance(self.dynamics_model, HardwareDynamics):
            init_state = self.dynamics_model.mocap_state()
        else:

            def sample_init_pos(ins):
                _, key = ins
                key, nkey = jax.random.split(key, 2)
                x_key, y_key = jax.random.split(key, 2)
                init_x = self.init_pose[:1] + jax.random.uniform(
                    x_key, shape=(1,), minval=-0.75, maxval=2.25
                )
                init_y = self.init_pose[1:2] + jax.random.uniform(
                    y_key, shape=(1,), minval=-1.5, maxval=1.5
                )
                init_pos = jnp.concatenate([init_x, init_y])
                return init_pos, nkey

            # Iterate until found a feasible initial position. Compare first key to make sure that sampling actually happens.
            init_pos, key_pos = jax.lax.while_loop(
                lambda ins: (
                    cost_fn(ins[0], jnp.asarray(self.obstacle[:2]), self.obstacle[2])
                    > 0.0
                )
                | ((ins[1] == key_pos).all()),
                sample_init_pos,
                (self.init_pose[:2], key_pos),
            )
            init_theta = self.init_pose[2:] + jax.random.uniform(
                key_pos, shape=(1,), minval=-jnp.pi, maxval=jnp.pi
            )
            init_vel = jnp.zeros((3,)) + jnp.array(
                [0.005, 0.005, 0.02]
            ) * jax.random.normal(key_vel, shape=(3,))
            init_state = jnp.concatenate([init_pos, init_theta, init_vel])
        init_state = self._obs(init_state, rng=key_obs)
        return State(
            pipeline_state=init_state,
            obs=init_state,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            info={"cost": jnp.array(0.0)},
        )

    def step(self, state: State, action: jax.Array) -> State:
        assert action.shape[-1:] == self.dim_action
        action = jnp.clip(action, -1.0, 1.0)
        action = action.at[1].set(self.max_throttle * action[1])
        obs = state.pipeline_state
        if self.encode_angle:
            dynamics_state = decode_angles(obs, self.angle_idx)
        next_dynamics_state, step_info = self.dynamics_model.step(
            dynamics_state, action, self.sys
        )
        # FIXME (yarden): hard-coded key is bad here.
        next_obs = self._obs(next_dynamics_state, rng=jax.random.PRNGKey(0))
        reward = self.reward_model.forward(obs=None, action=action, next_obs=next_obs)
        cost = cost_fn(obs[..., :2], jnp.asarray(self.obstacle[:2]), self.obstacle[2])
        done = jnp.asarray(0.0)
        info = {**state.info, "cost": cost, **step_info}
        next_state = State(
            pipeline_state=next_obs,
            obs=next_obs,
            reward=reward,
            done=done,
            metrics=state.metrics,
            info=info,
        )
        return next_state

    @property
    def observation_size(self) -> int:
        if self.encode_angle:
            return 7
        else:
            return 6

    @property
    def action_size(self) -> int:
        # [steering, throttle]
        return 2

    def backend(self) -> str:
        return "positional"


def render(env, policy, steps, rng):
    import numpy as np

    _, trajectory = rollout(env, policy, steps, rng)
    trajectory = jax.tree_map(lambda x: x[:, 0], trajectory.obs)
    if env.encode_angle:
        trajectory = decode_angles(trajectory, 2)
    obstacle_position, obstacle_radius = env.obstacle[:2], env.obstacle[2]
    images = [
        draw_scene(trajectory, timestep, obstacle_position, obstacle_radius)
        for timestep in range(steps)
    ]
    return np.asanyarray(images).transpose(0, 3, 1, 2)


def draw_scene(obs, timestep, obstacle_position, obstacle_radius):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle

    obstacle_position = jnp.array([obstacle_position[1], -obstacle_position[0]])
    # Create a figure and axis
    fig = Figure(figsize=(2.5, 2.5), dpi=300)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    pos_domain_size = 3.5
    ax.set_xlim(-pos_domain_size, pos_domain_size)
    ax.set_ylim(-pos_domain_size, pos_domain_size)
    target_center = (0, 0)
    colors = ["red", "white", "blue"]
    radii = [0.3, 0.2, 0.1]

    for radius, color in zip(radii, colors):
        circle = Circle(target_center, radius, color=color, ec="black", lw=0.5)
        ax.add_patch(circle)
    # Rotate coordinates if required
    rotated_trajectory = rotate_coordinates(obs, encode_angle=False)
    # Plot the car's position and velocity at the specified timestep
    x, y = rotated_trajectory[timestep, 0], rotated_trajectory[timestep, 1]
    vx, vy = rotated_trajectory[timestep, 3], rotated_trajectory[timestep, 4]
    car_width, car_length = 0.07, 0.2
    car = Rectangle(
        (x - car_length / 2, y - car_width / 2),
        car_length,
        car_width,
        angle=rotated_trajectory[timestep, 2] * 180 / np.pi,
        color="green",
        alpha=0.7,
        ec="black",
        rotation_point="center",
    )
    ax.add_patch(car)
    obstacle = Circle(
        obstacle_position,
        obstacle_radius,
        color="gray",
        alpha=0.5,
        ec="black",
        lw=1.5,
    )
    ax.add_patch(obstacle)
    ax.quiver(
        x,
        y,
        vx,
        vy,
        color="blue",
        scale=10,
        headlength=3,
        headaxislength=3,
        headwidth=3,
        linewidth=0.5,
    )
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    # Render figure to canvas and retrieve RGB array
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").copy()
    image = image.reshape(*reversed(canvas.get_width_height()), 3)
    return image
