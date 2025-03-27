# %%
import argparse
import pickle

# %%
parser = argparse.ArgumentParser(description="Load and play policy")
parser.add_argument(
    "--trajectory_path",
    type=str,
    required=False,
    default="./data/trajectory.pkl",
    help="Path to the trajectory file",
)

args = parser.parse_args()
# %%

with open(args.trajectory_path, "rb") as f:
    trajectory = pickle.load(f)


# %%
terminated = 1 - trajectory.extras["state_extras"]["truncation"]
