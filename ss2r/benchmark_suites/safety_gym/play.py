import collections
import time
from typing import Sequence

import go_to_goal
import jax
import lidar
import mujoco as mj
import mujoco.viewer
from absl import app
from jax import numpy as jp
from mujoco import mjx
from pynput import keyboard

#  Global state for interactive viewer
VIEWERGLOBAL_STATE = {
    "ctrl": [0.0, 0.0],  # The control values for the robot
    "reset": False,
}


# Pynput key press handler
def on_press(key):
    try:
        if key == keyboard.Key.up:  # '1' key to toggle control 1
            VIEWERGLOBAL_STATE["ctrl"][0] = 0.5
        if key == keyboard.Key.down:  # '1' key to toggle control 1
            VIEWERGLOBAL_STATE["ctrl"][0] = -0.5
        if key == keyboard.Key.left:  # '2' key to toggle control 2
            VIEWERGLOBAL_STATE["ctrl"][1] = 1.0
        elif key == keyboard.Key.right:
            VIEWERGLOBAL_STATE["ctrl"][1] = -1.0
    except AttributeError:
        # Handle special keys like 'space'
        pass


def on_release(key):
    try:
        if key == keyboard.Key.up:  # '1' key to toggle control 1
            VIEWERGLOBAL_STATE["ctrl"][0] = 0.0
        if key == keyboard.Key.down:  # '1' key to toggle control 1
            VIEWERGLOBAL_STATE["ctrl"][0] = 0.0
        if key == keyboard.Key.left:  # '2' key to toggle control 2
            VIEWERGLOBAL_STATE["ctrl"][1] = 0.0
        elif key == keyboard.Key.right:
            VIEWERGLOBAL_STATE["ctrl"][1] = 0.0
        if key == keyboard.Key.enter:
            print("Reset pressed!")
            VIEWERGLOBAL_STATE["reset"] = True
    except AttributeError:
        pass


def _main(argv: Sequence[str]) -> None:
    """Launches MuJoCo interactive viewer fed by MJX."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    with jax.disable_jit(False):
        task = go_to_goal.GoToGoal()
        reset_fn = jax.jit(task.reset)
        rng = jax.random.PRNGKey(0)
        states = collections.deque([reset_fn(rng)], maxlen=10)
        # Initialize the simulation step function
        # JIT compile the step function if needed
        print("JIT-compiling the model physics step...")
        start = time.time()
        step_fn = (
            jax.jit(task.step)
            .lower(
                states[0],
                jp.zeros((task.action_size,)),
            )
            .compile()
        )
        elapsed = time.time() - start
        print(f"Compilation took {elapsed}s.")
        # Set up key listener (pynput in a separate thread)
        with keyboard.Listener(on_press=on_press, on_release=on_release):
            # Launch interactive viewer
            m = task._mj_model
            d = mj.MjData(m)
            mjx.get_data_into(d, m, states[-1].data)
            with mujoco.viewer.launch_passive(m, d) as viewer:
                viewer.sync()
                count = 0
                reward = 0
                while viewer.is_running():
                    start = time.time()
                    mujoco.mjv_applyPerturbPose(m, d, viewer.perturb, 0)
                    mujoco.mjv_applyPerturbForce(m, d, viewer.perturb)
                    data = states[-1].data.replace(
                        qpos=jp.array(d.qpos),
                        qvel=jp.array(d.qvel),
                        mocap_pos=jp.array(d.mocap_pos),
                        mocap_quat=jp.array(d.mocap_quat),
                        xfrc_applied=jp.array(d.xfrc_applied),
                    )
                    states[-1] = states[-1].replace(data=data)
                    ctrl = jp.array(VIEWERGLOBAL_STATE["ctrl"])
                    states.append(step_fn(states[-1], ctrl))
                    print(states[-1].obs)
                    if states[-1].reward < -1e-1:
                        print("reward", states[-1].reward, count)
                    if states[-1].info["goal_reached"]:
                        print("Goal reached", count)
                        print("reward", states[-1].reward, count)
                    count += 1
                    reward += states[-1].reward
                    if count % 1000 == 0:
                        print("reward", reward, count)
                        reward = 0
                    # lidar.update_lidar_rings(states[-1].obs[: 16 * 3], m)
                    if VIEWERGLOBAL_STATE["reset"]:
                        rng, rng_ = jax.random.split(rng)
                        states.append(reset_fn(rng_))
                        VIEWERGLOBAL_STATE["reset"] = False
                    mjx.get_data_into(d, m, states[-1].data)
                    viewer.sync()
                    elapsed = time.time() - start
                    if elapsed < task._mj_model.opt.timestep:
                        time.sleep(task._mj_model.opt.timestep - elapsed)


def main():
    app.run(_main)


if __name__ == "__main__":
    main()
