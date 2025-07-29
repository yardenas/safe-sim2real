import json
import os
import shlex
import subprocess
import sys


def extract_num_seeds(args):
    for arg in args:
        if arg.startswith("+eval_params.num_seeds="):
            return int(arg.split("=", 1)[1])
    return 1


def remove_num_seeds_arg(args):
    return [a for a in args if not a.startswith("+eval_params.num_seeds=")]


def average_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = [key for key in metrics_list[0].keys() if not key.startswith("training/")]
    avg = {}
    for k in keys:
        values = [m[k] for m in metrics_list if k in m]
        if values and isinstance(values[0], (int, float)):
            avg[k] = sum(values) / len(values)
    return avg


def remove_wandb_writer(args):
    new_args = []
    for arg in args:
        if arg.startswith("writers="):
            writers = arg.split("=", 1)[1]
            writers_list = [
                w.strip() for w in writers.replace("[", "").replace("]", "").split(",")
            ]
            writers_list = [w for w in writers_list if w != "wandb"]
            if writers_list:
                new_arg = f"writers={','.join(writers_list)}"
                new_args.append(new_arg)
        else:
            new_args.append(arg)
    return new_args


def main():
    base_args = sys.argv[1:]
    num_seeds = extract_num_seeds(base_args)
    args_wo_num_seeds = remove_num_seeds_arg(base_args)
    # Remove wandb from writers for subprocesses to not clutter wandb with multiple runs
    args_wo_wandb = remove_wandb_writer(args_wo_num_seeds)

    wandb_config = {
        "writers": ["wandb"],
        "wandb": {
            "project": "ss2r",
            "name": "eval_policy_seeds_avg_" + str(base_args),
            "entity": None,
        },
    }
    for arg in base_args:
        if arg.startswith("wandb.entity="):
            wandb_config["wandb"]["entity"] = arg.split("=", 1)[1]
        if arg.startswith("wandb.project="):
            wandb_config["wandb"]["project"] = arg.split("=", 1)[1]

    metrics_list = []
    for seed in range(num_seeds):
        cmd = [
            sys.executable,
            "eval_policy.py",
            f"training.seed={seed}",
            "writers=[stderr]",
            *args_wo_wandb,
        ]
        print(f"Running: {' '.join(shlex.quote(a) for a in cmd)}")
        with open(f"eval_policy_seed_{seed}.out", "w") as out, open(
            f"eval_policy_seed_{seed}.err", "w"
        ) as err:
            result = subprocess.run(
                cmd, stdout=out, stderr=err, start_new_session=True, close_fds=True
            )
        path = os.path.abspath(os.path.dirname(__file__))
        metrics_path = os.path.join(path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            metrics_list.append(metrics)
            os.rename(metrics_path, f"metrics_seed_{seed}.json")
            if result.returncode != 0:
                print(
                    f"Warning: Run with seed {seed} exited with code {result.returncode}, but metrics.json was found. Treating as success."
                )
        else:
            print(
                f"Run with seed {seed} failed and metrics.json not found. See eval_policy_seed_{seed}.err for details."
            )
            sys.exit(result.returncode)

    # Average metrics
    avg_metrics = average_metrics(metrics_list)
    avg_metrics["num_seeds"] = num_seeds
    avg_metrics["seeds"] = list(range(num_seeds))

    print("Averaged metrics across seeds:")
    print(json.dumps(avg_metrics, indent=2))

    # Log averaged metrics to wandb using TrainingLogger
    try:
        import wandb

        wandb.init(
            project=wandb_config["wandb"]["project"],
            name=wandb_config["wandb"]["name"],
            entity=wandb_config["wandb"]["entity"],
            config=wandb_config,
        )
        wandb.log(avg_metrics)
        print("Averaged metrics logged to wandb.")
    except Exception as e:
        print(f"Failed to log averaged metrics to wandb: {e}")


if __name__ == "__main__":
    main()
