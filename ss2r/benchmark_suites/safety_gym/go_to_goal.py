from collections import defaultdict
from typing import Any, Dict, Mapping, NamedTuple, Tuple, Union

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
BASE_SENSORS = ["accelerometer", "velocimeter", "gyro", "magnetometer"]
_EXTENTS = (-2.0, -2.0, 2.0, 2.0)


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
    spec: mj.MjSpec,
    layout: dict[str, list[tuple[int, jax.Array]]],
    visualize: bool = False,
):
    """Build the arena (currently, just adds Lidar rings). Future: dynamically add obstacles, hazards, objects, goal here"""
    if visualize:
        lidar.add_lidar_rings(spec)

    vases_spec = layout["vases"]
    maybe_floor = spec.worldbody.geoms[0]
    assert maybe_floor.name == "floor"
    size = max(_EXTENTS)
    maybe_floor.size = jp.array([size + 0.1, size + 0.1, 0.1])
    for i, (_, xy) in enumerate(vases_spec):
        name = f"vase_{i}"
        xyz = jp.hstack([xy, 0.1])
        volume = 0.1**3
        density = 0.001
        vase = spec.worldbody.add_body(name=name, pos=xyz, mass=volume * density)
        geom = dict(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 0.1, 0.1],
            rgba=[0, 1, 1, 1],
            userdata=jp.ones(1),
        )
        vase.add_geom(name=f"{name}_geom", **geom)
        vase.add_freejoint()
    hazards_spec = layout["hazards"]
    for i, (_, xy) in enumerate(hazards_spec):
        name = f"hazard_{i}"
        geom = dict(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.2, 0.01, 0],
            rgba=[0.0, 0.0, 1.0, 0.25],
            userdata=jp.ones(1),
            contype=jp.zeros(()),
            conaffinity=jp.zeros(()),
        )
        xyz = jp.hstack([xy, 2e-2])
        hazard = spec.worldbody.add_body(name=name, pos=xyz)
        hazard.add_geom(name=f"{name}_geom", **geom)
    goal_pos = layout["goal"][0][1]
    xyz = jp.hstack([goal_pos, 0.3 / 2.0 + 1e-2])
    goal = spec.worldbody.add_body(name="goal", pos=xyz, mocap=True)
    goal.add_geom(
        name="goal_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.3, 0.15, 0],
        rgba=[0, 1, 0, 0.25],
        contype=jp.zeros(()),
        conaffinity=jp.zeros(()),
    )
    # TODO: add site to goal


# TODO (yarden): should not depend on mujoco playground eventually
class GoToGoal(mjx_env.MjxEnv):
    def __init__(self, visualize_lidar: bool = False):
        self.spec = {
            "robot": ObjectSpec(0.4, 1),
            "goal": ObjectSpec(0.305, 1),
            "hazards": ObjectSpec(0.18, 10),
            "vases": ObjectSpec(0.15, 10),
        }
        mj_spec: mj.MjSpec = mj.MjSpec.from_file(filename=str(_XML_PATH), assets={})
        layout = _sample_layout(jax.random.PRNGKey(0), self.spec)
        build_arena(mj_spec, layout, visualize=visualize_lidar)
        self._mj_model = mj_spec.compile()
        path = epath.Path(__file__).parent / "test.xml"
        mj_spec.to_file(str(path))
        self._mjx_model = mjx.put_model(self._mj_model)
        # FIXME (yarden): make sure to handle the sizes of the vases/hazards
        self._post_init()

    def _post_init(self) -> None:
        """Post initialization for the model."""
        # For reward function
        self._robot_site_id = self._mj_model.site("robot").id
        # TODO: not sure the goal should have a site.
        self._goal_site_id = self._mj_model.body("goal_site").id
        self._goal_body_id = self._mj_model.body("goal").id
        self._goal_mocap_id = self._mj_model.mocap("goal").id
        self._robot_body_id = self._mj_model.body("robot").id
        self._hazards_body_ids = [
            self._mj_model.body(f"hazard_{id_}")
            for id_ in range(self.spec["hazards"].num_objects)
        ]
        vases_names = [f"vase_{id_}" for id_ in range(self.spec["vases"].num_objects)]
        self._vases_body_ids = [self._mj_model.body(name) for name in vases_names]
        self._vases_qpos_ids = [
            mjx_env.get_qpos_ids(self.mj_model, name) for name in vases_names
        ]
        # For cost function
        self._robot_geom_id = self._mj_model.geom("robot").id
        self._pointarrow_geom_id = self._mj_model.geom("pointarrow").id
        self._init_q = self._mj_model.qpos0

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
        # TODO (yarden): goal position should not be sampled via joint, but as
        # a mocap
        # TODO (yarden): should resample the goal position besed on the
        # positions of the other objects
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
        robot_body_pos = data.xpos[self._robot_body_id]
        robot_body_mat = data.xmat[self._robot_body_id].reshape(3, 3)
        # Vectorized obstacle position retrieval
        obstacle_positions = data.xpos[
            jp.array(self._hazards_body_ids + self._vases_body_ids)
        ]
        # FIXME: the goal_body_id should actually be goal_mocap_id, and we should get
        # the position via the mocap
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

    def sensor_observations(self, data: mjx.Data) -> jax.Array:
        vals = []
        for sensor in BASE_SENSORS:
            vals.append(mjx_env.get_sensor_data(self.mj_model, data, sensor))
        return jp.hstack(vals)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        lidar = self.lidar_observations(data)
        other_sensors = self.sensor_observations(data)
        return jp.hstack([lidar.flatten(), other_sensors])

    def _update_data(
        self,
        layout: dict[str, list[tuple[int, jax.Array]]],
        rng: jax.Array,
        data: mjx.Data | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if data is None:
            new_qpos = self._init_q
            hazard_mocap_pos = self._mj_model.body_pos[self._hazards_body_ids]
            goal_mocap_pos = self._mj_model.body_pos[self._goal_body_id]
            # FIXME (yarden): not sure about the right order here
            new_mocap_pos = jp.hstack((hazard_mocap_pos, goal_mocap_pos))
        else:
            new_qpos = data.qpos
            new_mocap_pos = data.mocap_pos
        for name, (_, positions) in layout.items():
            if name == "goal":
                assert len(positions) == 1
                new_mocap_pos = new_mocap_pos[self._goal_mocap_id].set(positions[0])
            elif name == "hazards":
                for xy, id_ in zip(positions, self._hazards_body_ids):
                    new_mocap_pos = new_mocap_pos.at[id_].set(xy)
            elif name == "vases":
                for xy, ids in zip(positions, self._vases_qpos_ids):
                    rng, rng_ = jax.random.split(rng)
                    rotation = jax.random.uniform(rng_, minval=0.0, maxval=2 * jp.pi)
                    quat = _rot2quat(rotation)
                    # FIXME (yarden): this is not necessarily the right order.
                    pos = jp.hstack((xy, quat))
                    new_qpos = new_qpos.at[ids].set(pos)
            else:
                assert False, "Something is off with the names provided in spec."
        return new_mocap_pos, new_qpos

    def reset(self, rng) -> State:
        layout = _sample_layout(rng, self.spec)
        mocap_pos, qpos = self._update_data(layout, rng)
        data = mjx_env.init(self.mjx_model, qpos=qpos, mocap_pos=mocap_pos)
        initial_goal_dist = jp.linalg.norm(
            data.site_xpos[self._goal_site_id][:2]
            - data.site_xpos[self._robot_site_id][0:2]
        )
        info = {"rng": rng, "last_goal_dist": initial_goal_dist, "cost": 0.0}
        obs = self.get_obs(data)
        return State(data, obs, jp.zeros(()), jp.zeros(()), {}, info)  # type: ignore

    def step(self, state: State, action: jax.Array) -> State:
        lower, upper = (
            self._mj_model.actuator_ctrlrange[:, 0],
            self._mj_model.actuator_ctrlrange[:, 1],
        )
        action = (action + 1.0) / 2.0 * (upper - lower) + lower
        data = mjx_step(self._mjx_model, state.data, action, n_substeps=2)
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
        return jp.logical_and(i < 1000, conflicted)

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
            if iter_ >= 1000:
                print("Failed to find a valid sample for", name)
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
