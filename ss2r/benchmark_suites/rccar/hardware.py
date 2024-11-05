import sys
from contextlib import contextmanager
from typing import Any

import jax
import numpy as np

try:
    sys.path.append(
        "C:/Users/Panda/Desktop/rcCarInterface/rc-car-interface/build/src/libs/pyCarController"
    )
    import carl
except ImportError as e:
    print("Could not import carl: ", e)


@contextmanager
def connect(
    car_id: int = 2,
    control_frequency: float = 41.66664,
    max_wait_time: float = 1,
    window_size: int = 6,
    port_number: int = 8,
):
    if car_id == 1:
        mocap_id = 1003
    elif car_id == 2:
        mocap_id = 1034
    else:
        raise Exception("Only 2 cars have a mocap id")
    controller = carl.controller(
        w_size=window_size,
        p_number=port_number,
        mocap_id=mocap_id,
        wait_time=max_wait_time,
        control_freq=control_frequency,
    )
    try:
        yield controller
    finally:
        controller.stop()


@contextmanager
def start(controller):
    controller.start()
    try:
        yield
    finally:
        controller.stop()


class HardwareDynamics:
    def __init__(self, controller, max_throttle) -> None:
        self.controller = controller
        self.max_throttle = max_throttle

    # FIXME (yarden): make sure that this really conforms to the actual state
    def step(self, x: jax.Array, u: jax.Array, params: Any) -> tuple[jax.Array, dict]:
        scaled_action = np.array(u).copy()
        scaled_action[1] *= self.max_throttle
        self.controller.control_mode()
        command_set_in_time = self.controller.set_command(scaled_action)
        assert command_set_in_time, "API blocked python thread for too long"
        time_elapsed = self.controller.get_time_elapsed()
        raw_state = self.mocap_state()
        return raw_state, {"elapsed_time": time_elapsed}

    def mocap_state(self):
        current_state = self.controller.get_state()
        mocap_x = current_state[[0, 3]]
        current_state[[0, 3]] = current_state[[1, 4]]
        current_state[[1, 4]] = mocap_x
        current_state[2] += np.pi
        current_state[2] = ((current_state[2] + np.pi) % (2 * np.pi)) - np.pi
        return current_state
