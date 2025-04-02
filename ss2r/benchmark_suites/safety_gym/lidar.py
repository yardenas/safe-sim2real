import jax
import jax.numpy as jp
import mujoco as mj

LIDAR_GROUPS = ["obstacle", "goal", "object"]

# For calculations
NUM_LIDAR_BINS = 16
LIDAR_MAX_DIST = 2.0

# Visualisation
BASE_OFFSET = 0.5
OFFSET_STEP = 0.06
RADIANS = 0.15
LIDAR_SIZE = 0.025


def _lidar(
    robot_pos: jax.Array, robot_mat: jax.Array, targets_pos: jax.Array
) -> jax.Array:
    obs = jp.zeros(NUM_LIDAR_BINS)

    def ego_xy(pos):
        robot_3vec = robot_pos
        """Transforms world position to ego-centric robot frame in 2D."""
        pos_3vec = jp.concatenate(
            [pos, jp.array([0.0])]
        )  # Add zero z-coordinate -- not needed I thin
        world_3vec = pos_3vec - robot_3vec  # make sure obstacle pos is 3D
        return jp.matmul(world_3vec, robot_mat)[:2]  # Extract X, Y onl

    for pos in targets_pos:
        #   pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]  # Truncate Z coordinate

        z = jax.lax.complex(*ego_xy(pos))
        # print(f"z: {z}")
        dist = jp.abs(z)
        angle = jp.angle(z) % (jp.pi * 2)
        # print(f"dist: {dist}, angle: {angle}")
        bin_size = (jp.pi * 2) / NUM_LIDAR_BINS
        bin_ = (angle / bin_size).astype(jp.int32)
        bin_angle = bin_size * bin_
        # sensor = max(0, LIDAR_MAX_DIST - dist) / LIDAR_MAX_DIST
        sensor = jp.maximum(0, LIDAR_MAX_DIST - dist) / LIDAR_MAX_DIST
        # sensor = jp.maximum(0, 1 - dist)  # / self.LIDAR_MAX_DIST
        obs = obs.at[bin_].set(jp.maximum(obs[bin_], sensor))
        alias = (angle - bin_angle) / bin_size
        # assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle ' \
        #                         f'{angle}, bin {bin_}'
        bin_plus = (bin_ + 1) % NUM_LIDAR_BINS
        bin_minus = (bin_ - 1) % NUM_LIDAR_BINS
        obs = obs.at[bin_plus].set(jp.maximum(obs[bin_plus], alias * sensor))
        obs = obs.at[bin_minus].set(jp.maximum(obs[bin_minus], (1 - alias) * sensor))

    return obs


def add_lidar_rings(spec: mj.MjSpec):
    robot_body = spec.body("robot")

    # Add LIDAR rings above the robot body
    for i, category in enumerate(LIDAR_GROUPS):
        lidar_body = robot_body.add_body(name=f"lidar_{category}")
        for bin in range(NUM_LIDAR_BINS):
            theta = 2 * jp.pi * (bin + 0.5) / NUM_LIDAR_BINS
            binpos = jp.array(
                [
                    jp.cos(theta) * RADIANS,
                    jp.sin(theta) * RADIANS,
                    BASE_OFFSET + OFFSET_STEP * i,
                ]
            )
            rgba = [0, 0, 0, 1]
            rgba[i] = 1
            lidar_body.add_site(
                name=f"lidar_{category}_{bin}",
                size=LIDAR_SIZE * jp.ones(3),  # Size of the lidar site
                rgba=rgba,  # Color of the lidar site
                pos=binpos,  # Position of the lidar site
            )


def update_lidar_rings(lidar_values: jax.Array, model: mj.MjModel):
    obstacle_lidar, goal_lidar, object_lidar = lidar_values
    # Update data just for viewer
    for lidars, category in zip(
        [obstacle_lidar, goal_lidar, object_lidar], LIDAR_GROUPS
    ):
        for i, value in enumerate(lidars):
            lidar_site_id = model.site(f"lidar_{category}_{i}").id
            model.site_rgba[lidar_site_id][3] = min(1.0, value + 0.1)  # CHange alpah


# def _lidar(robot_pos : jax.Array, robot_mat : jax.Array, targets_pos : jax.Array) -> jax.Array:
#     obs = jp.zeros(NUM_LIDAR_BINS)

#     def ego_xy(pos):
#         robot_3vec = robot_pos
#         """Transforms world position to ego-centric robot frame in 2D."""
#         pos_3vec = jp.concatenate([pos, jp.array([0.0])])  # Add zero z-coordinate -- not needed I thin
#         world_3vec = pos_3vec - robot_3vec # make sure obstacle pos is 3D
#         return jp.matmul(world_3vec, robot_mat)[:2]  # Extract X, Y onl

#     for pos in targets_pos:
#     #   pos = np.asarray(pos)
#         if pos.shape == (3,):
#             pos = pos[:2]  # Truncate Z coordinate

#         z = complex(*ego_xy(pos))  # X, Y as real, imaginary components
#         # print(f"z: {z}")
#         dist = jp.abs(z)
#         angle = jp.angle(z) % (jp.pi * 2)
#         # print(f"dist: {dist}, angle: {angle}")
#         bin_size = (jp.pi * 2) / NUM_LIDAR_BINS
#         bin_ = int(angle / bin_size)
#         bin_angle = bin_size * bin_
#         # sensor = max(0, LIDAR_MAX_DIST - dist) / LIDAR_MAX_DIST
#         sensor = jp.maximum(0, LIDAR_MAX_DIST - dist) / LIDAR_MAX_DIST
#         # sensor = jp.maximum(0, 1 - dist)  # / self.LIDAR_MAX_DIST
#         obs = obs.at[bin_].set(jp.maximum(obs[bin_], sensor))
#         alias = (angle - bin_angle) / bin_size
#         assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle ' \
#                                 f'{angle}, bin {bin_}'
#         bin_plus = (bin_ + 1) % NUM_LIDAR_BINS
#         bin_minus = (bin_ - 1) % NUM_LIDAR_BINS
#         obs = obs.at[bin_plus].set(jp.maximum(obs[bin_plus], alias * sensor))
#         obs = obs.at[bin_minus].set(jp.maximum(obs[bin_minus], (1 - alias) * sensor))

#     return obs

# # Need m for the ids -- can possibly pass directly
# def lidar_observations(m, dx) -> jp.array:
#     # print("lidar observations")
#     obstacles = ["vase", "hazard", "pillar"]
#     obstacle_count = {
#         "vase": 2,
#         "hazard": 2,
#         "pillar": 1
#     }
#     # lidar = jp.array()
#     # dx.

#     robot_body_id = m.body('robot').id
#     robot_body_pos = dx.xpos[robot_body_id]
#     robot_body_mat = dx.xmat[robot_body_id].reshape(3,3)


#     ### TODO TODO TODO
#     # TODO: MAKE jp.arrays (lidar, positions, etc)
#     # TODO: Make the lidar update vectorized
#     ### TODO TODO TODO
#     lidar = []

#     obstacle_positions = []
#     # TODO: Do this with user groups or return e.g. the IDs/obstacle objects of all obstacles when adding them to the scene
#     for obstacle in obstacles:
#         for i in range(obstacle_count[obstacle]):
#             id = m.body(f'{obstacle}_{i}').id
#             pos = dx.xpos[id]
#             obstacle_positions.append(pos)

#     goal_positions = [dx.xpos[m.body('goal').id]]
#     object_positions = [] # Empty for now

#     # print(type(mx.body_pos[id]))
#     lidar.append(_lidar(robot_body_pos, robot_body_mat, obstacle_positions))
#     lidar.append(_lidar(robot_body_pos, robot_body_mat, goal_positions))
#     lidar.append(_lidar(robot_body_pos, robot_body_mat, object_positions))

#     # print("LIDAR LENGHT: ")
#     # print(len(lidar))
#     return lidar


# def _lidar_np(robot_pos, robot_mat, positions: List[np.ndarray]) -> np.ndarray:
#     """
#     Return a robot-centric lidar observation of a list of positions.

#     Lidar is a set of bins around the robot (divided evenly in a circle).
#     The detection directions are exclusive and exhaustive for a full 360 view.
#     Each bin reads 0 if there are no objects in that direction.
#     If there are multiple objects, the distance to the closest one is used.
#     Otherwise, the bin reads the fraction of the distance towards the robot.

#     E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
#     and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
#     (The reading can be thought of as "closeness" or inverse distance)

#     This encoding has some desirable properties:
#         - bins read 0 when empty
#         - bins smoothly increase as objects get close
#         - maximum reading is 1.0 (where the object overlaps the robot)
#         - close objects occlude far objects
#         - constant size observation with variable numbers of objects
#     """
#     obs = np.zeros(NUM_LIDAR_BINS)

#     def ego_xy(pos):
#       robot_3vec = robot_pos
#       pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
#       world_3vec = pos_3vec - robot_3vec
#       return np.matmul(world_3vec, robot_mat)[:2]

#     for pos in positions:
#       pos = np.asarray(pos)
#       if pos.shape == (3,):
#         pos = pos[:2]  # Truncate Z coordinate
#       z = complex(*ego_xy(pos))  # X, Y as real, imaginary components
#       print(f"z_np: {z}")
#       dist = np.abs(z)
#       angle = np.angle(z) % (np.pi * 2)
#       print(f"np_dist: {dist}, np_angle: {angle}")
#       bin_size = (np.pi * 2) / NUM_LIDAR_BINS
#       bin_ = int(angle / bin_size)
#       bin_angle = bin_size * bin_
#       sensor = max(0, LIDAR_MAX_DIST - dist) / LIDAR_MAX_DIST
#       obs[bin_] = max(obs[bin_], sensor)
#       alias = (angle - bin_angle) / bin_size
#       assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle ' \
#                               f'{angle}, bin {bin_}'
#       bin_plus = (bin_ + 1) % NUM_LIDAR_BINS
#       bin_minus = (bin_ - 1) % NUM_LIDAR_BINS
#       obs[bin_plus] = max(obs[bin_plus], alias * sensor)
#       obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)

#     return obs
