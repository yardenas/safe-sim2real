import os

import wandb


def get_wandb_checkpoint(run_id):
    api = wandb.Api()
    artifact = api.artifact(f"ss2r/checkpoint:{run_id}")
    download_dir = artifact.download(f"{get_state_path()}/{run_id}")
    return download_dir


def get_state_path() -> str:
    log_path = os.getcwd() + "/ckpt"
    return log_path
