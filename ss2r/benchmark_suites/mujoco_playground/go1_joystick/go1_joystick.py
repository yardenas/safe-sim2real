from mujoco_playground import locomotion


def domain_randomization(sys, rng, cfg):
    return locomotion.g1_randomize(sys, rng, cfg)
