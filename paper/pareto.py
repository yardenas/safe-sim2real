# %%
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so
import wandb
from seaborn import axes_style
from tueplots import bundles, figsizes

warnings.filterwarnings("ignore")


# %%
api = wandb.Api()


def load_evaluation(data, seed, lambda_, environment):
    if data.index.name is None or data.index.name != "step":
        data.rename(columns={"_step": "step"}, inplace=True)
        data.set_index("step", inplace=True)
    data = data[data.columns[data.columns.str.startswith("eval")]].dropna()
    data["seed"] = seed
    data["lambda"] = lambda_
    data["environment"] = environment
    return data


def config_to_lambda(config):
    lambda_ = config["agent"]["cost_robustness"]["cost_penalty"]
    return lambda_


def handle_run(run):
    metrics = run.history(
        keys=["eval/episode_reward", "eval/episode_cost", "_step"],
        x_axis="_step",
        pandas=False,
    )
    config = run.config
    metrics = pd.DataFrame(metrics)
    lambda_ = config_to_lambda(config)
    seed = config["training"]["seed"]
    environment = config["environment"]["task_name"]
    if environment == "go_to_goal":
        environment = "PointGoal2"
    return metrics, seed, lambda_, environment


def walk_wandb_runs(project, filters):
    runs = api.runs(project, filters=filters)
    print(f"Fetching {len(runs)} runs")
    for run in runs:
        out = handle_run(run)
        if out is None:
            continue
        yield out


filters = {"display_name": {"$regex": "apr16-pareto-g2"}}

data = pd.concat(
    [
        load_evaluation(*summary_files_data)
        for summary_files_data in walk_wandb_runs("yardas/ss2r", filters)
    ]
)

# %%
last_steps = data.groupby(["seed", "lambda", "environment"]).tail(1).reset_index()
aggregated_data = (
    last_steps.groupby(["lambda", "environment"])[
        ["eval/episode_reward", "eval/episode_cost"]
    ]
    .mean()
    .reset_index()
)
budgets = {
    "RaceCar": 5,
    "SafeCartpoleSwingup": 100,
    "SafeHumanoidWalk": 100,
    "SafeQuadrupedRun": 100,
    "SafeWalkerWalk": 100,
    "PointGoal2": 25,
}

aggregated_data["normalized_episode_cost"] = aggregated_data.apply(
    lambda row: 1 / (row["eval/episode_cost"] / budgets[row["environment"]]), axis=1
)

cmap = "viridis"  # or try "plasma", "magma", "cividis"
cmap = plt.get_cmap(cmap)
norm = mcolors.Normalize(
    vmin=aggregated_data["lambda"].min(), vmax=aggregated_data["lambda"].max()
)
aggregated_data["color_mapped"] = aggregated_data["lambda"].map(lambda x: cmap(norm(x)))  # type: ignore
# aggregated_data = aggregated_data[aggregated_data["lambda"] <= 0.15]


def pareto_front(df):
    front = []
    for _, row in df.sort_values(
        by="normalized_episode_cost", ascending=False
    ).iterrows():
        if not front or row["eval/episode_reward"] > front[-1]["eval/episode_reward"]:
            front.append(row)
    return pd.DataFrame(front)


front = pareto_front(aggregated_data)

# %%


def fill_unsafe(ax):
    ax.axvspan(0, 0.99, color="red", alpha=0.1)


theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=1))
fig, ax = plt.subplots()

# Set up the colormap and normalization
cmap = plt.get_cmap("viridis")
satisfies_constraint = aggregated_data["normalized_episode_cost"] > 1
norm = mcolors.Normalize(
    vmin=aggregated_data["lambda"][satisfies_constraint].min(),
    vmax=aggregated_data["lambda"][satisfies_constraint].max(),
)

# Scatter plot with color mapped to lambda

# Plot all points — satisfying ones in color, others grey/empty
sc = ax.scatter(
    aggregated_data.loc[satisfies_constraint, "normalized_episode_cost"],
    aggregated_data.loc[satisfies_constraint, "eval/episode_reward"],
    c=aggregated_data.loc[satisfies_constraint, "lambda"],
    cmap=cmap,
    norm=norm,
    s=35,
    edgecolors="k",
    linewidths=0.3,
    label="Feasible",
)

# Violating points — transparent/empty or grey
ax.scatter(
    aggregated_data.loc[~satisfies_constraint, "normalized_episode_cost"],
    aggregated_data.loc[~satisfies_constraint, "eval/episode_reward"],
    facecolors="none",
    edgecolors="k",
    linewidths=1.0,
    alpha=0.7,
    s=35,
    label="Infeasible",
)

# Plot Pareto front line (over full front)
ax.plot(
    front["normalized_episode_cost"],
    front["eval/episode_reward"],
    color="black",
    linestyle="-",
    linewidth=1.0,
    label="Pareto front",
    zorder=0,
)

fill_unsafe(ax)
ax.axvline(
    x=1.0,
    color="black",
    alpha=0.2,
    linestyle=(0, (1, 1)),
    linewidth=1.25,
)

ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
# Labels and formatting
ax.set_xlabel(r"$\hat{C}(\tilde{\pi})$")
ax.set_ylabel(r"$\hat{J}(\tilde{\pi})$")
ax.set_title("Pareto Front with Constraint")

for spine in ax.spines.values():
    spine.set_linewidth(1.25)
# Add colorbar only for feasible points
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("lambda")
fig.savefig("pareto-front-g2g.pdf")
