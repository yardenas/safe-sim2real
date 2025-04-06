import jax
import jax.numpy as jp
import mujoco as mj

LIDAR_GROUPS = ["obstacle", "goal", "object"]

# For calculations
NUM_LIDAR_BINS = 16
LIDAR_MAX_DIST = 3.0

# Visualisation
BASE_OFFSET = 0.5
OFFSET_STEP = 0.06
RADIANS = 0.15
LIDAR_SIZE = 0.025


def compute_lidar(
    robot_pos: jax.Array, robot_mat: jax.Array, targets_pos: jax.Array
) -> jax.Array:
    obs = jp.zeros(NUM_LIDAR_BINS)

    def ego_xy(pos):
        robot_3vec = robot_pos
        """Transforms world position to ego-centric robot frame in 2D."""
        pos_3vec = jp.concatenate([pos, jp.array([0.0])])
        world_3vec = pos_3vec - robot_3vec
        return jp.matmul(world_3vec, robot_mat)[:2]

    for pos in targets_pos:
        if pos.shape == (3,):
            pos = pos[:2]  # Truncate Z coordinate
        z = jax.lax.complex(*ego_xy(pos))
        dist = jp.abs(z)
        angle = jp.angle(z) % (jp.pi * 2)
        bin_size = (jp.pi * 2) / NUM_LIDAR_BINS
        bin_ = (angle / bin_size).astype(jp.int32)
        bin_angle = bin_size * bin_
        sensor = jp.maximum(0, LIDAR_MAX_DIST - dist) / LIDAR_MAX_DIST
        obs = obs.at[bin_].set(jp.maximum(obs[bin_], sensor))
        alias = (angle - bin_angle) / bin_size
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
    obstacle_lidar, goal_lidar, object_lidar = jp.split(lidar_values, 3)
    # Update data just for viewer
    for lidars, category in zip(
        [obstacle_lidar, goal_lidar, object_lidar], LIDAR_GROUPS
    ):
        for i, value in enumerate(lidars):
            lidar_site_id = model.site(f"lidar_{category}_{i}").id
            model.site_rgba[lidar_site_id][3] = min(1.0, value + 0.1)  # Change alpha
