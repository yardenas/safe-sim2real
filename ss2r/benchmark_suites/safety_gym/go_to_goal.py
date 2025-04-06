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

# _XML_PATH = "xml/point.xml"
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


# sample_layout(vase: [10, 5], hazard: [20, 2], goal : []): [-2, -2, 2, 2]-> (vase: [x y theta])
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
        volume = 0.1**3
        density = 0.001
        vase = spec.worldbody.add_body(
            name=f"vase_{i}",
            mass=volume * density,
        )

        vase.add_geom(
            name=f"vase_{i}_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.1, 0.1, 0.1],
            rgba=[0, 1, 1, 1],
            userdata=jp.ones(1),
        )

        # Free joint bug in visualizer: https://github.com/google-deepmind/mujoco/issues/2508
        vase.add_freejoint(name=f"vase_{i}_joint")

    for i in range(objects["hazards"].num_objects):
        hazard = spec.worldbody.add_body(name=f"hazard_{i}", mocap=True)
        hazard.add_geom(
            name=f"hazard_{i}_geom",
            type=mj.mjtGeom.mjGEOM_CYLINDER,
            size=[0.2, 0.01, 0],
            rgba=[0.0, 0.0, 1.0, 0.25],
            userdata=jp.ones(1),
            contype=jp.zeros(()),
            conaffinity=jp.zeros(()),
        )

    goal = spec.worldbody.add_body(name="goal", mocap=True)
    goal.add_geom(
        name="goal_geom",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=[0.3, 0.15, 0],
        rgba=[0, 1, 0, 0.25],
        contype=jp.zeros(()),
        conaffinity=jp.zeros(()),
    )

    # Visualize lidar rings
    if visualize:
        lidar.add_lidar_rings(spec)


class GoToGoal:
    def __init__(self):
        self.spec = {
            "robot": ObjectSpec(0.4, 1),
            "goal": ObjectSpec(0.305, 1),
            "hazards": ObjectSpec(0.18, 10),
            "vases": ObjectSpec(0.15, 10),
        }

        mjSpec: mj.MjSpec = mj.MjSpec.from_file(filename=str(_XML_PATH), assets={})
        build_arena(mjSpec, objects=self.spec, visualize=True)
        self._mj_model = mjSpec.compile()

        # print(mjSpec.to_xml())

        self._mjx_model = mjx.put_model(self._mj_model)

        self._post_init()

    def _post_init(self) -> None:
        """Post initialization for the model."""
        # For reward function
        self._robot_site_id = self._mj_model.site("robot").id
        self._goal_body_id = self._mj_model.body("goal").id

        # For cost function
        self._robot_geom_id = self._mj_model.geom("robot").id
        self._pointarrow_geom_id = self._mj_model.geom("pointarrow").id
        # Geoms, not bodies
        self._collision_obstacle_geoms_ids = [
            self._mj_model.geom(f"vase_{i}_geom").id
            for i in range(self.spec["vases"].num_objects)
            # + self._mj_model.geom(f'pillar{i}').id for i in range(self._num_pillars)
        ]
        self._hazard_body_ids = [
            self._mj_model.body(f"hazard_{i}").id
            for i in range(self.spec["hazards"].num_objects)
        ]  # Bodies, not geoms

        # For lidar
        self._robot_body_id = self._mj_model.body("robot").id
        self._vase_body_ids = [
            self._mj_model.body(f"vase_{i}").id
            for i in range(self.spec["vases"].num_objects)
        ]
        self._obstacle_body_ids = self._vase_body_ids + self._hazard_body_ids
        self._object_body_ids: list[int] = []
        # For position updates
        self._robot_x_id = self._mj_model.joint("x").id
        self._robot_y_id = self._mj_model.joint("y").id
        self._robot_joint_qposadr = [
            self._mj_model.jnt_qposadr[joint_id]
            for joint_id in [self._robot_x_id, self._robot_y_id]
        ]
        self._goal_mocap_id = self._mj_model.body("goal").mocapid[0]
        self._hazard_mocap_id = [
            self._mj_model.body(f"hazard_{i}").mocapid[0]
            for i in range(self.spec["hazards"].num_objects)
        ]
        self._vase_joint_ids = [
            self._mj_model.joint(f"vase_{i}_joint").id
            for i in range(self.spec["vases"].num_objects)
        ]
        self._vase_joint_qposadr = [
            self._mj_model.jnt_qposadr[joint_id] for joint_id in self._vase_joint_ids
        ]

    def get_reward(
        self, data: mjx.Data, last_goal_dist: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        goal_distance = jp.linalg.norm(
            data.xpos[self._goal_body_id][:2] - data.site_xpos[self._robot_site_id][0:2]
        )
        reward = last_goal_dist - goal_distance
        return reward, goal_distance

    def _reset_goal(self, data: mjx.Data, rng: jax.Array) -> tuple[mjx.Data, jax.Array]:
        # Initial state

        # new_rng, goal_key = jax.random.split(rng)
        # new_xy = jax.random.uniform(goal_key, (2,), minval=-2.0, maxval=2.0)
        # # new_qpos = data.qpos.at[jp.array([self._goal_x_joint_id, self._goal_y_joint_id])].set(new_xy)
        # # data = data.replace(qpos=new_qpos)
        # data = data.replace(mocap_pos=data.mocap_pos.at[self._goal_mocap_id, :2].set(new_xy))
        # jax.debug.print("New goal position: {pos}", pos=new_xy)
        # return data, rng

        # TODO: probably could just use xpos with self._obstacle_body_ids instead of mocap_pos as well - it seems to work
        rng, goal_key = jax.random.split(rng)
        hazard_pos = data.mocap_pos[jp.array(self._hazard_mocap_id)][:, :2]
        vases_pos = data.xpos[jp.array(self._vase_body_ids)][:, :2]
        other_xy = jp.vstack([hazard_pos, vases_pos])

        hazard_keepout = jp.full((hazard_pos.shape[0],), self.spec["hazards"].keepout)
        vases_keepout = jp.full((vases_pos.shape[0],), self.spec["vases"].keepout)
        other_keepout = jp.hstack([hazard_keepout, vases_keepout])

        xy, _ = draw_until_valid(
            goal_key, self.spec["goal"].keepout, other_xy, other_keepout
        )

        # new_qpos = data.qpos.at[jp.array([self._goal_x_joint_id, self._goal_y_joint_id])].set(new_xy)
        # data = data.replace(qpos=new_qpos)
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._goal_mocap_id, :2].set(xy)
        )
        # jax.debug.print("New goal position: {pos}", pos=xy)
        return data, rng

    def obstacle_positions(self, data: mjx.Data) -> jax.Array:
        obstacle_positions = data.xpos[jp.array(self._obstacle_body_ids)]
        return obstacle_positions

    def get_cost(self, data: mjx.Data) -> jax.Array:
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
        # FOR DEBUG PURPOSES, UNCOMMENT THIS
        colliding_obstacles = jax.lax.cond(
            jp.any(colliding_obstacles),  # If there's any collision
            lambda x: jax.debug.print(
                "Collision detected with obstacles: {collisions}", collisions=x
            )
            or x,  # Print and return the collisions
            lambda x: x,  # Otherwise, return the input unchanged
            colliding_obstacles,  # The value to pass into the lambda
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

        # Vectorized obstacle position retrieval -- note we can use xpos even for mocap positions after they have been updated
        # These values seem to be equal; TODO: using mocap_pos is maybe more correct
        obstacle_positions = data.xpos[jp.array(self._obstacle_body_ids)]
        goal_positions = data.mocap_pos[jp.array([self._goal_mocap_id])]
        object_positions = (
            data.xpos[jp.array(self._object_body_ids)] if self._object_body_ids else []
        )

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

    def update_positions(
        self,
        data: mjx.Data,
        layout: dict[str, list[tuple[int, jax.Array]]],
        rng: jax.Array,
    ) -> mjx.Data:
        mocap_pos = data.mocap_pos
        qpos = data.qpos

        # Set robot position
        qpos = data.qpos.at[jp.array(self._robot_joint_qposadr)].set(
            layout["robot"][0][1]
        )

        # N.B. could not figure out how to do it with get_qpos_ids, it seems to repeat some indices and hence does not set stuff correctly
        for i, (_, xy) in enumerate(layout["vases"]):
            rng, rng_ = jax.random.split(rng)
            adr = self._vase_joint_qposadr[i]
            rotation = jax.random.uniform(rng_, minval=0.0, maxval=2 * jp.pi)
            quat = _rot2quat(rotation)
            qpos = qpos.at[adr : adr + 7].set(jp.hstack([xy, 0.1, quat]))

        # Set hazard positions
        for i, (_, xy) in enumerate(layout["hazards"]):
            mocap_pos = mocap_pos.at[self._hazard_mocap_id[i]].set(
                jp.hstack([xy, 0.02])
            )

        # Set goal position
        mocap_pos = mocap_pos.at[self._goal_mocap_id].set(
            jp.hstack([layout["goal"][0][1], 0.3 / 2.0 + 1e-2])
        )

        data = data.replace(qpos=qpos, mocap_pos=mocap_pos)

        return data, rng

    def reset(self, rng) -> State:
        data = mjx.make_data(self._mjx_model)

        # Set initial object positions
        layout = _sample_layout(rng, self.spec)
        data, rng = self.update_positions(data, layout, rng)
        data = mjx.forward(
            self._mjx_model, data
        )  # Make sure updated positions are reflected in data

        # Check updated positiosn are correct
        # print("Hazards:")
        # print(layout["hazards"])
        # print(data.mocap_pos[jp.array(self._hazard_mocap_id)])

        # print("Vases:")
        # print(layout["vases"])
        # print(data.xpos[jp.array(self._vase_body_ids)])

        # print("Goal:")
        # print(layout["goal"])
        # print(data.mocap_pos[self._goal_mocap_id])

        # print("Robot:")
        # print(layout["robot"])
        # print(data.xpos[jp.array(self._robot_body_id)])

        initial_goal_dist = jp.linalg.norm(
            data.mocap_pos[self._goal_mocap_id][:2]
            - data.site_xpos[self._robot_site_id][0:2]
        )
        info = {"rng": rng, "last_goal_dist": initial_goal_dist, "cost": jp.zeros(())}

        obs = self.get_obs(data)

        return State(data, obs, jp.zeros(()), jp.zeros(()), {}, info)  # type: ignore

    def step(self, state: State, action: jax.Array) -> State:
        lower, upper = (
            self._mj_model.actuator_ctrlrange[:, 0],
            self._mj_model.actuator_ctrlrange[:, 1],
        )
        action = (action + 1.0) / 2.0 * (upper - lower) + lower
        data = mjx_env.mjx_step(self._mjx_model, state.data, action, n_substeps=2)
        reward, goal_dist = self.get_reward(data, state.info["last_goal_dist"])
        # Reset goal if robot inside goal
        condition = goal_dist < 0.3
        data, rng = jax.lax.cond(
            condition, self._reset_goal, lambda d, r: (d, r), data, state.info["rng"]
        )
        cost = self.get_cost(data)
        observations = self.get_obs(data)
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(jp.float32)
        info = {"rng": rng, "cost": cost, "last_goal_dist": goal_dist}
        return State(data, observations, reward, done, state.metrics, info)  # type: ignore

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

    @property
    def observation_size(self) -> int:
        return 3 * lidar.NUM_LIDAR_BINS + len(BASE_SENSORS) * 3


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

            jax.lax.cond(
                iter_ >= 1000,
                lambda _: jax.debug.print(f"Failed to find a valid sample for {name}"),
                lambda _: None,
                operand=None,
            )
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
