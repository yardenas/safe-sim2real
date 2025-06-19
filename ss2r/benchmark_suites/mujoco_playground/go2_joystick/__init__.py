from functools import partial

from mujoco_playground import locomotion

from ss2r.benchmark_suites.mujoco_playground.go2_joystick import (
    getup,
    handstand,
    joystick,
)

locomotion.register_environment(
    "Go2JoystickFlatTerrain",
    partial(joystick.Joystick, task="flat_terrain"),
    joystick.default_config,
)
locomotion.register_environment(
    "Go2JoystickRoughTerrain",
    partial(joystick.Joystick, task="rough_terrain"),
    joystick.default_config,
)
locomotion.register_environment("Go2Getup", getup.Getup, getup.default_config)
locomotion.register_environment(
    "Go2Handstand", handstand.Handstand, handstand.default_config
)
locomotion.register_environment(
    "Go2Footstand", handstand.Footstand, handstand.default_config
)
