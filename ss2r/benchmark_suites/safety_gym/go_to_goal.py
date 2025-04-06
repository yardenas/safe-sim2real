from collections import defaultdict
from typing import Any, Dict, Mapping, NamedTuple, Tuple, Union

import jax
import jax.numpy as jp
import mujoco as mj
from etils import epath
from flax import struct
from mujoco import mjx
from mujoco_playground._src import mjx_env

import ss2r.benchmark_suites.safety_gym.lidar as lidar

_XML_PATH = epath.Path(__file__).parent / "point.xml"

Observation = Union[jax.Array, Mapping[str, jax.Array]]
BASE_SENSORS = ["accelerometer", "velocimeter", "gyro", "magnetometer"]
_EXTENTS = (-2.0, -2.0, 2.0, 2.0)
_GOAL_SIZE = 0.3


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


class ObjectSpec(NamedTuple):
    keepout: float
    num_objects: int


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


def build_arena(
    spec: mj.MjSpec, objects: dict[str, ObjectSpec], visualize: bool = False
):
    """Build the arena (currently, just adds Lidar rings). Future: dynamically add obstacles, hazards, objects, goal here"""
    # Set floor size
    maybe_floor = spec.worldbody.geoms[0]
    assert maybe_floor.name == "floor"
    size = max(_EXTENTS)
    maybe_floor.size = jp.array([size + 0.1, size + 0.1, 0.1])
    # Reposition robot
    for i in range(objects["vases"].num_objects):
        density = 0.001
        vase = spec.worldbody.add_body(
            name=f"vase_{i}",
        )
        vase.add_geom(
            name=f"vase_{i}",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.1, 0.1, 0.1],
            rgba=[0, 1, 1, 1],
            userdata=jp.ones(1),
            density=density,
        )
        # Free joint bug in visualizer: https://github.com/google-deepmind/mujoco/issues/2508
        vase.add_freejoint(name=f"vase_{i}")
    for i in range(objects["hazards"].num_objects):
        hazard = spec.worldbody.add_body(name=f"hazard_{i}", mocap=True)
        hazard.add_geom(
            name=f"hazard_{i}",
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[0.2, 0.01, 0],
            rgba=[0.0, 0.0, 1.0, 0.25],
            userdata=jp.ones(1),
            contype=jp.zeros(()),
            conaffinity=jp.zeros(()),
        )
    goal = spec.worldbody.add_body(name="goal", mocap=True)
    goal.add_geom(
        name="goal",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=[_GOAL_SIZE, _GOAL_SIZE / 2.0, 0],
        rgba=[0, 1, 0, 0.25],
        contype=jp.zeros(()),
        conaffinity=jp.zeros(()),
    )
    if visualize:
        lidar.add_lidar_rings(spec)


# TODO (yarden): should not depend on mujoco playground eventually
class GoToGoal(mjx_env.MjxEnv):
    def __init__(self, visualize_lidar: bool = True):
        self.spec = {
            "robot": ObjectSpec(0.4, 1),
            "goal": ObjectSpec(_GOAL_SIZE + 0.05, 1),
            "hazards": ObjectSpec(0.18, 0),
            "vases": ObjectSpec(0.15, 0),
        }
        mj_spec: mj.MjSpec = mj.MjSpec.from_file(filename=str(_XML_PATH), assets={})
        build_arena(mj_spec, objects=self.spec, visualize=visualize_lidar)
        self._mj_model = mj_spec.compile()
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

    def _post_init(self) -> None:
        """Post initialization for the model."""
        # For reward function
        self._robot_site_id = self._mj_model.site("robot").id
        self._goal_body_id = self._mj_model.body("goal").id
        # For cost function
        vases_names = [f"vase_{id_}" for id_ in range(self.spec["vases"].num_objects)]
        self._vases_qpos_ids = [
            mjx_env.get_qpos_ids(self.mj_model, [name]) for name in vases_names
        ]
        self._robot_geom_id = self._mj_model.geom("robot").id
        self._pointarrow_geom_id = self._mj_model.geom("pointarrow").id
        # Geoms, not bodies
        self._collision_obstacle_geoms_ids = [
            self._mj_model.geom(f"vase_{i}").id
            for i in range(self.spec["vases"].num_objects)
        ]
        self._hazard_body_ids = [
            self._mj_model.body(f"hazard_{i}").id
            for i in range(self.spec["hazards"].num_objects)
        ]
        # For lidar
        self._robot_body_id = self._mj_model.body("robot").id
        self._vase_body_ids = [
            self._mj_model.body(f"vase_{i}").id
            for i in range(self.spec["vases"].num_objects)
        ]
        self._obstacle_body_ids = self._vase_body_ids + self._hazard_body_ids
        self._goal_mocap_id = self._mj_model.body("goal").mocapid[0]
        self._hazard_mocap_id = [
            self._mj_model.body(f"hazard_{i}").mocapid[0]
            for i in range(self.spec["hazards"].num_objects)
        ]
        self._robot_qpos_ids = mjx_env.get_qpos_ids(self._mj_model, ["x", "y", "z"])
        self._init_q = self._mj_model.qpos0

    def get_reward(
        self, data: mjx.Data, last_goal_dist: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        goal_distance = jp.linalg.norm(self._robot_to_goal(data)[:2])
        reward = last_goal_dist - goal_distance
        return reward, goal_distance

    def _robot_to_goal(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._goal_body_id] - data.xpos[self._robot_body_id]

    def _compass(self, data: mjx.Data) -> jax.Array:
        v = self._robot_to_goal(data)
        ori = data.xmat[self._robot_body_id].reshape(3, 3)
        v = jp.matmul(v, ori)
        v = v / (jp.linalg.norm(v) + 0.001)
        return v[:2]

    def _reset_goal(self, data: mjx.Data, rng: jax.Array) -> tuple[mjx.Data, jax.Array]:
        # other_xy = self.obstacle_positions(data)[:, :2]
        other_xy = jp.array([])
        num_vases = self.spec["vases"].num_objects
        num_hazards = self.spec["hazards"].num_objects
        hazard_keepout = jp.full((num_hazards,), self.spec["hazards"].keepout)
        vases_keepout = jp.full((num_vases,), self.spec["vases"].keepout)
        other_keepout = jp.hstack([hazard_keepout, vases_keepout])
        rng, goal_key = jax.random.split(rng)
        xy, _ = draw_until_valid(
            goal_key, self.spec["goal"].keepout, other_xy, other_keepout
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._goal_mocap_id, :2].set(xy)
        )
        return data, rng

    def obstacle_positions(self, data: mjx.Data) -> jax.Array:
        # obstacle_positions = data.xpos[jp.array(self._obstacle_body_ids)]
        obstacle_positions = jp.array([])
        return obstacle_positions

    def get_cost(self, data: mjx.Data) -> jax.Array:
        return 0.0
        # Check if robot or pointarrow geom collide with any vase or pillar
        colliding_obstacles = jp.array(
            [
                jp.logical_or(
                    geoms_colliding(data, geom, self._robot_geom_id),
                    geoms_colliding(data, geom, self._pointarrow_geom_id),
                )
                for geom in self._collision_obstacle_geoms_ids
            ]
        )
        # Hazard distance calculation (vectorized for all hazards)
        robot_pos = data.site_xpos[self._robot_site_id][:2]
        hazard_distances = jp.linalg.norm(
            data.xpos[jp.array(self._hazard_body_ids)][:, :2] - robot_pos, axis=1
        )
        cost = jp.sum(colliding_obstacles) + jp.sum(hazard_distances <= 0.2)
        return (cost > 0.0).astype(jp.float32)

    def lidar_observations(self, data: mjx.Data) -> jax.Array:
        """Compute Lidar observations."""
        robot_body_pos = data.xpos[self._robot_body_id]
        robot_body_mat = data.xmat[self._robot_body_id].reshape(3, 3)
        # obstacle_positions = data.xpos[jp.array(self._obstacle_body_ids)]
        obstacle_positions = jp.array([])
        goal_positions = data.mocap_pos[jp.array([self._goal_mocap_id])]
        object_positions = jp.array([])
        lidar_readings = jp.hstack(
            (
                lidar.compute_lidar(robot_body_pos, robot_body_mat, obstacle_positions),
                lidar.compute_lidar(robot_body_pos, robot_body_mat, goal_positions),
                lidar.compute_lidar(robot_body_pos, robot_body_mat, object_positions),
            )
        )
        return lidar_readings

    def sensor_observations(self, data: mjx.Data) -> jax.Array:
        vals = []
        for sensor in BASE_SENSORS:
            vals.append(mjx_env.get_sensor_data(self.mj_model, data, sensor))
        return jp.hstack(vals)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        # lidar = self.lidar_observations(data)
        # FIXME (yarden): this is wrong
        magnitude = jp.linalg.norm(self._robot_to_goal(data))
        direction = self._compass(data)
        other_sensors = self.sensor_observations(data)
        return jp.hstack([magnitude, direction, other_sensors])

    def _update_data(
        self,
        layout: dict[str, list[tuple[int, jax.Array]]],
        rng: jax.Array,
        data: mjx.Data | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if data is None:
            new_qpos = jp.asarray(self._init_q)
            new_mocap_pos = jp.zeros((self.mjx_model.nmocap, 3))
        else:
            new_qpos = data.qpos
            new_mocap_pos = data.mocap_pos
        for name, data in layout.items():
            _, positions = zip(*data)
            if name == "goal":
                assert len(positions) == 1
                xyz = jp.hstack([positions[0], _GOAL_SIZE / 2.0 + 1e-2])
                new_mocap_pos = new_mocap_pos.at[self._goal_mocap_id].set(xyz)
            elif name == "hazards":
                for xy, id_ in zip(positions, self._hazard_mocap_id):
                    xyz = jp.hstack([xy, 0.02])
                    new_mocap_pos = new_mocap_pos.at[id_].set(xyz)
            elif name == "vases":
                for xy, ids in zip(positions, self._vases_qpos_ids):
                    rng, rng_ = jax.random.split(rng)
                    rotation = jax.random.uniform(rng_, minval=-jp.pi, maxval=jp.pi)
                    quat = _rot2quat(rotation)
                    pos = jp.hstack((xy, 0.1 - 4e-5, quat))
                    new_qpos = new_qpos.at[ids].set(pos)
            elif name == "robot":
                assert len(positions) == 1
                xy = positions[0]
                rng, rng_ = jax.random.split(rng)
                rotation = jax.random.uniform(rng_, minval=-jp.pi, maxval=jp.pi)
                pos = jp.hstack((xy, rotation))
                new_qpos = new_qpos.at[self._robot_qpos_ids].set(pos)
            else:
                assert False, "Something is off with the names provided in spec."
        return new_qpos, new_mocap_pos

    def reset(self, rng) -> State:
        layout = _sample_layout(rng, self.spec)
        rng, rng_ = jax.random.split(rng)
        qpos, mocap_pos = self._update_data(layout, rng_)
        mocap_quat = jp.zeros((self.mjx_model.nmocap, 4))
        mocap_quat = mocap_quat.at[:, 0].set(1.0)
        data = mjx_env.init(
            self.mjx_model, qpos=qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat
        )
        initial_goal_dist = jp.linalg.norm(
            data.mocap_pos[self._goal_mocap_id][:2]
            - data.site_xpos[self._robot_site_id][0:2]
        )
        info = {
            "rng": rng,
            "last_goal_dist": initial_goal_dist,
            "cost": jp.zeros(()),
            "goal_reached": jp.zeros(()),
        }
        obs = self.get_obs(data)
        return State(data, obs, jp.zeros(()), jp.zeros(()), {}, info)  # type: ignore

    def step(self, state: State, action: jax.Array) -> State:
        lower, upper = (
            self._mj_model.actuator_ctrlrange[:, 0],
            self._mj_model.actuator_ctrlrange[:, 1],
        )
        action = (action + 1.0) / 2.0 * (upper - lower) + lower
        data = mjx_env.step(self._mjx_model, state.data, action, n_substeps=2)
        reward, goal_dist = self.get_reward(data, state.info["last_goal_dist"])
        # Reset goal if robot inside goal
        condition = goal_dist < _GOAL_SIZE
        data, rng = jax.lax.cond(
            condition, self._reset_goal, lambda d, r: (d, r), data, state.info["rng"]
        )
        reward = jp.where(condition, reward + 1.0, reward)
        goal_dist = jp.where(
            condition, jp.linalg.norm(self._robot_to_goal(data)[:2]), goal_dist
        )
        cost = self.get_cost(data)
        obs = self.get_obs(data)
        state.info["last_goal_dist"] = goal_dist
        state.info["rng"] = rng
        state.info["cost"] = cost
        state.info["goal_reached"] = condition.astype(jp.float32)
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
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


# PLACEMENT FNS
def _rot2quat(theta):
    return jp.array([jp.cos(theta / 2), 0, 0, jp.sin(theta / 2)])


def placement_not_valid(xy, object_keepout, other_xy, other_keepout):
    def check_single(other_xy, other_keepout):
        dist = jp.linalg.norm(xy - other_xy)
        return dist < (other_keepout + object_keepout)

    validity_checks = jax.vmap(check_single)(other_xy, other_keepout)
    return jp.any(validity_checks)


def draw_until_valid(rng, object_keepout, other_xy, other_keepout):
    def cond_fn(val):
        i, conflicted, *_ = val
        return jp.logical_and(i < 10000, conflicted)

    def body_fn(val):
        i, _, _, rng = val
        rng, rng_ = jax.random.split(rng)
        xy = draw_placement(rng_, object_keepout)
        conflicted = placement_not_valid(xy, object_keepout, other_xy, other_keepout)
        return i + 1, conflicted, xy, rng

    # Initial state: (iteration index, conflicted flag, placeholder for xy)
    init_val = (0, True, jp.zeros((2,)), rng)  # Assuming xy is a 2D point
    i, _, xy, *_ = jax.lax.while_loop(cond_fn, body_fn, init_val)
    return xy, i


def _sample_layout(
    rng: jax.Array, objects_spec: dict[str, ObjectSpec]
) -> dict[str, list[tuple[int, jax.Array]]]:
    num_objects = sum(spec.num_objects for spec in objects_spec.values())
    all_placements = jp.ones((num_objects, 2)) * 100.0
    all_keepouts = jp.zeros(num_objects)
    layout = defaultdict(list)
    flat_idx = 0
    for _, (name, object_spec) in enumerate(objects_spec.items()):
        rng, rng_ = jax.random.split(rng)
        keys = jax.random.split(rng_, object_spec.num_objects)
        for _, key in enumerate(keys):
            xy, iter_ = draw_until_valid(
                key, object_spec.keepout, all_placements, all_keepouts
            )
            # TODO (yarden): technically should quit if not valid sampling.
            all_placements = all_placements.at[flat_idx, :].set(xy)
            all_keepouts = all_keepouts.at[flat_idx].set(object_spec.keepout)
            layout[name].append((flat_idx, xy))
            flat_idx += 1
    return layout


def constrain_placement(placement: tuple, keepout: float) -> tuple:
    """Helper function to constrain a single placement by the keepout radius"""
    xmin, ymin, xmax, ymax = placement
    return xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout


def draw_placement(rng: jax.Array, keepout) -> jax.Array:
    choice = constrain_placement(_EXTENTS, keepout)
    xmin, ymin, xmax, ymax = choice
    min_ = jp.hstack((xmin, ymin))
    max_ = jp.hstack((xmax, ymax))
    pos = jax.random.uniform(rng, shape=(2,), minval=min_, maxval=max_)
    return pos
