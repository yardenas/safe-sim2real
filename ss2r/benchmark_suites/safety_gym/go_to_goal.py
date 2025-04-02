from typing import Any, Dict, Mapping, Tuple, Union

import jax
import jax.numpy as jp
import mujoco
import mujoco as mj
from etils import epath
from flax import struct
from mujoco import mjx
from mujoco_playground._src import mjx_env

import ss2r.benchmark_suites.safety_gym.lidar as lidar

_XML_PATH = epath.Path(__file__).parent / "point.xml"

Observation = Union[jax.Array, Mapping[str, jax.Array]]


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        return

    in_axes = jax.tree_map(lambda x: None, sys)
    return sys, in_axes, jp.zeros(())


@struct.dataclass
class State:
    data: mjx.Data
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array]
    info: Dict[str, Any]


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
    """Return True if the two geoms are colliding."""
    return get_collision_info(state.contact, geom1, geom2)[0] < 0


def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
    """Get the distance and normal of the collision between two geoms."""
    mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jp.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = jp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, 0, :3]
    return dist, normal


def build_arena(spec: mj.MjSpec, visualize: bool = False):
    """Build the arena (currently, just adds Lidar rings). Future: dynamically add obstacles, hazards, objects, goal here"""
    if visualize:
        lidar.add_lidar_rings(spec)


# TODO (yarden): should not depend on mujoco playground eventually
class GoToGoal(mjx_env.MjxEnv):
    def __init__(self, visualize_lidar: bool = False):
        mjSpec: mj.MjSpec = mj.MjSpec.from_file(filename=str(_XML_PATH), assets={})
        build_arena(mjSpec, visualize=visualize_lidar)
        self._mj_model = mjSpec.compile()
        self._post_init()
        self._mjx_model = mjx.put_model(self._mj_model)

    def _post_init(self) -> None:
        """Post initialization for the model."""
        # For reward function
        self._robot_site_id = self._mj_model.site("robot").id
        self._goal_site_id = self._mj_model.site("goal_site").id
        self._goal_x_joint_id = self._mj_model.joint("goal_x").id
        self._goal_y_joint_id = self._mj_model.joint("goal_y").id
        self._goal_body_id = self._mj_model.body("goal").id
        self._robot_body_id = self._mj_model.body("robot").id
        # For cost function
        self._collision_obstacle_ids = [
            self._mj_model.geom("vase_0").id,
            self._mj_model.geom("vase_1").id,
            self._mj_model.geom("pillar_0").id,
        ]
        self._hazard_obstacle_ids = [
            self._mj_model.body("hazard_0").id,
            self._mj_model.body("hazard_1").id,
        ]
        self._robot_geom_id = self._mj_model.geom("robot").id
        self._pointarrow_geom_id = self._mj_model.geom("pointarrow").id
        self._obstacles = ["vase", "hazard", "pillar"]
        self._obstacle_count = {"vase": 2, "hazard": 2, "pillar": 1}

    def get_reward(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> tuple[jax.Array, jax.Array]:
        goal_distance = jp.linalg.norm(
            data.site_xpos[self._goal_site_id][:2]
            - data.site_xpos[self._robot_site_id][0:2]
        )
        last_goal_dist = info["last_goal_dist"]
        reward = last_goal_dist - goal_distance
        return reward, goal_distance

    def _reset_goal(self, data: mjx.Data, rng: jax.Array) -> tuple[mjx.Data, jax.Array]:
        # Initial state
        new_rng, goal_key = jax.random.split(rng)
        new_xy = jax.random.uniform(goal_key, (2,), minval=-3.0, maxval=3.0)
        new_qpos = data.qpos.at[
            jp.array([self._goal_x_joint_id, self._goal_y_joint_id])
        ].set(new_xy)
        return data.replace(qpos=new_qpos), new_rng

    def get_cost(self, data: mjx.Data) -> jax.Array:
        # Check if robot or pointarrow geom collide with any vase or pillar
        colliding_obstacles = jp.array(
            [
                jp.logical_or(
                    geoms_colliding(data, geom, self._robot_geom_id),
                    geoms_colliding(data, geom, self._pointarrow_geom_id),
                )
                for geom in self._collision_obstacle_ids
            ]
        )
        # Hazard distance calculation (vectorized for all hazards)
        robot_pos = data.site_xpos[self._robot_site_id][:2]
        hazard_distances = jp.linalg.norm(
            data.xpos[jp.array(self._hazard_obstacle_ids)][:, :2] - robot_pos, axis=1
        )
        # Compute cost: Add cost for collisions and proximity to hazards
        cost = jp.sum(colliding_obstacles) + jp.sum(hazard_distances <= 0.2)
        return cost.astype(jp.float32)

    def lidar_observations(self, data: mjx.Data) -> jax.Array:
        """Compute Lidar observations."""
        robot_body_pos = data.xpos[self._robot_body_id]
        robot_body_mat = data.xmat[self._robot_body_id].reshape(3, 3)
        # Vectorized obstacle position retrieval
        obstacle_ids = [
            self._mj_model.body(f"{obstacle}_{i}").id
            for obstacle in self._obstacles
            for i in range(self._obstacle_count[obstacle])
        ]
        obstacle_positions = data.xpos[jp.array(obstacle_ids)]
        goal_positions = data.xpos[jp.array([self._goal_body_id])]
        object_positions = jp.array([])

        lidar_readings = jp.array(
            [
                lidar.compute_lidar(robot_body_pos, robot_body_mat, obstacle_positions),
                lidar.compute_lidar(robot_body_pos, robot_body_mat, goal_positions),
                lidar.compute_lidar(robot_body_pos, robot_body_mat, object_positions),
            ]
        )
        return lidar_readings

    def get_obs(self, data: mjx.Data) -> jax.Array:
        lidar = self.lidar_observations(data)
        return lidar.flatten()

    def reset(self, rng) -> State:
        data = mjx_env.init(self.mjx_model)
        initial_goal_dist = jp.linalg.norm(
            data.site_xpos[self._goal_site_id][:2]
            - data.site_xpos[self._robot_site_id][0:2]
        )
        info = {"rng": rng, "last_goal_dist": initial_goal_dist, "cost": 0.0}
        obs = self.get_obs(data)
        return State(data, obs, jp.zeros(()), jp.zeros(()), {}, info)  # type: ignore

    def step(self, state: State, action: jax.Array) -> State:
        data = mjx_step(self._mjx_model, state.data, action, n_substeps=1)
        reward, goal_dist = self.get_reward(data, state.info)
        # Reset goal if robot inside goal
        # TODO (yarden): not hard-code this
        condition = goal_dist < 0.3
        data, rng = jax.lax.cond(
            condition, self._reset_goal, lambda d, r: (d, r), data, state.info["rng"]
        )
        cost = self.get_cost(data)
        obs = self.get_obs(data)
        state.info["last_goal_dist"] = goal_dist
        state.info["rng"] = rng
        state.info["cost"] = cost
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(jp.float32)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)  # type: ignore
        return state

    @property
    def xml_path(self) -> str:
        return _XML_PATH

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


def mjx_step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
    def single_step(data, _):
        data = data.replace(ctrl=action)
        data = mjx.step(model, data)
        return data, None

    return jax.lax.scan(single_step, data, (), n_substeps)[0]
