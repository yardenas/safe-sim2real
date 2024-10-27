from hydra import compose, initialize


def make_test_config(additional_overrides=None):
    if additional_overrides is None:
        additional_overrides = []
    with initialize(version_base=None, config_path="../ss2r/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "writers=[stderr]",
                "+experiment=debug",
            ]
            + additional_overrides,
        )
        return cfg
