# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import wandb
from seaborn import axes_style
from tueplots import bundles, figsizes

warnings.filterwarnings("ignore")


# %%
api = wandb.Api()


def load_evaluation(data, seed, lambda_, magnitude):
    if data.index.name is None or data.index.name != "step":
        data.rename(columns={"_step": "step"}, inplace=True)
        data.set_index("step", inplace=True)
    data = data[data.columns[data.columns.str.startswith("eval")]].dropna()
    data["seed"] = seed
    data["lambda"] = lambda_
    data["magnitude"] = magnitude
    return data


def walk_wandb_runs(project, filters):
    runs = api.runs(project, filters=filters)
    print(f"Fetching {len(runs)} runs")
    for run in runs:
        metrics = run.history(
            keys=["eval/episode_reward", "eval/episode_cost", "_step"],
            x_axis="_step",
            pandas=False,
        )
        metrics = pd.DataFrame(metrics)
        seed = run.config["training"]["seed"]
        lambda_ = run.config["agent"]["cost_robustness"]["cost_penalty"]
        magnitude = run.config["environment"]["eval_params"]["gear"][-1]
        yield metrics, seed, lambda_, magnitude


filters = {
    "display_name": {"$regex": "feb15--agi$"},
}

data = pd.concat(
    [
        load_evaluation(*summary_files_data)
        for summary_files_data in walk_wandb_runs("yardas/ss2r", filters)
    ]
)
# %%

data = data[data["lambda"] != 0]
last_steps = data.groupby(["seed", "lambda", "magnitude"]).tail(3).reset_index()
aggregated_data = (
    last_steps.groupby(["lambda", "magnitude"])[
        ["eval/episode_reward", "eval/episode_cost"]
    ]
    .median()
    .reset_index()
)


# %%

theme = bundles.jmlr2001()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(figsizes.jmlr2001(nrows=1, ncols=2, height_to_width_ratio=0.8))
fig, axes = plt.subplots(1, 2, sharey=True)
aggregated_data["score"] = aggregated_data["eval/episode_reward"] / (
    aggregated_data["eval/episode_cost"] + 10
)

sns.heatmap(
    aggregated_data.pivot(
        index="magnitude", columns="lambda", values="eval/episode_reward"
    ).sort_index(ascending=False),
    ax=axes[0],
    # cmap="viridis",
    linewidths=1.75,
)
axes[0].axvspan(5, 5.1, color="white")
sns.heatmap(
    aggregated_data.pivot(
        index="magnitude", columns="lambda", values="eval/episode_cost"
    ).sort_index(ascending=False),
    ax=axes[1],
    # cmap="viridis",
    linewidths=1.75,
)
axes[1].axvspan(4, 4.1, color="white")
fig.savefig("tune-lambda-cartpole.pdf")

# %%

theme = bundles.jmlr2001()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(figsizes.jmlr2001(nrows=1, ncols=1, height_to_width_ratio=0.8))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

reward_pivot = aggregated_data.pivot(
    index="lambda", columns="magnitude", values="eval/episode_reward"
)
cost_pivot = aggregated_data.pivot(
    index="lambda", columns="magnitude", values="eval/episode_cost"
)

# Optionally, sort the lambda index in ascending order (or descending, if desired)
reward_pivot = reward_pivot.sort_index(ascending=True)
cost_pivot = cost_pivot.sort_index(ascending=True)

# Convert the index and columns to numpy arrays for the meshgrid.
lambdas = reward_pivot.index.values
magnitudes = reward_pivot.columns.values

# Create meshgrid. Note: meshgrid expects x and y values, so here we treat:
# x-axis as lambda, and y-axis as magnitude.
L, M = np.meshgrid(lambdas, magnitudes)

# Our pivoted data arrays are 2D with shape (n_lambda, n_magnitude) but note:
# When we pivoted, lambda became rows and magnitude became columns. So we need to transpose.
reward_data = reward_pivot.values.T  # Now shape is (n_magnitudes, n_lambdas)
cost_data = cost_pivot.values.T  # Same for cost
# Plot the reward surface
reward_surface = ax.plot_surface(
    L, M, reward_data, cmap="viridis", alpha=0.8, edgecolor="none", label="Reward"
)

# Plot the cost surface (with some transparency)
cost_surface = ax.plot_surface(
    L, M, cost_data, cmap="coolwarm", alpha=0.8, edgecolor="none", label="Cost"
)

# Label axes
ax.set_xlabel("Lambda")
ax.set_ylabel("Perturbation Magnitude")
ax.set_zlabel("Value")
ax.set_title("3D Surface Plot: Reward and Cost")

# Optionally add colorbars by creating separate mappable objects.
fig.colorbar(reward_surface, ax=ax, shrink=0.5, aspect=10, pad=0.1, label="Reward")
fig.colorbar(cost_surface, ax=ax, shrink=0.5, aspect=10, pad=0.05, label="Cost")

plt.show()

# %%

theme = bundles.jmlr2001()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(figsizes.jmlr2001(nrows=1, ncols=5, height_to_width_ratio=1.1))
plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
fig = plt.figure()
set_size = r"\lvert\Xi\rvert"
so.Plot(
    aggregated_data,
    x="eval/episode_cost",
    y="eval/episode_reward",
    marker="lambda",
    color="lambda",
).facet(col="magnitude").share(y=True, x=True).add(
    so.Dot(pointsize=4.5, edgewidth=0.5), legend=True
).scale(
    color=so.Nominal(
        values=[
            "silver",
            "silver",
            "silver",
            "silver",
            "#5F4690",
            "silver",
            "silver",
            "silver",
        ],
    ),
).label(
    x=r"$\hat{C}(\pi)$", y=r"$\hat{J}(\pi)$", title=lambda s: f"${set_size} = {s}$"
).theme(axes_style("ticks")).on(fig).plot()


def text_lambda(s):
    return f"$\lambda$ = {s}"


axes = fig.get_axes()
for i, ax in enumerate(axes):
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
legend = fig.legends.pop(0)
fig.legend(
    legend.legend_handles,
    [text_lambda(t.get_text()) for t in legend.texts],
    loc="center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=8,
    frameon=False,
    handletextpad=-0.5,
    columnspacing=0.5,
)

fig.savefig("tune-lambda-cartpole-2.pdf")


# %%
theme = bundles.jmlr2001()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(figsizes.jmlr2001(nrows=2, ncols=5, height_to_width_ratio=1.1))
plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
fig = plt.figure()
set_size = r"\lvert\Xi\rvert"
so.Plot(
    aggregated_data,
    x="lambda",
).facet(col="magnitude").pair(y=["eval/episode_reward", "eval/episode_cost"]).share(
    x=True
).add(
    so.Line(linewidth=1.0, pointsize=2.5, edgewidth=0.1),
    legend=False,
).theme(axes_style("ticks")).on(fig).plot()


axes = fig.get_axes()
for i, ax in enumerate(axes):
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    ax.axvline(0.6, color="black", linewidth=0.5, zorder=1)

fig.savefig("tune-lambda-cartpole-3.pdf")

# %%
theme = bundles.jmlr2001()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(figsizes.jmlr2001(nrows=1, ncols=2))
plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
fig, axes = plt.subplots(1, 2)
metrics = ["eval/episode_reward", "eval/episode_cost"]
set_size = r"\lvert\Xi\rvert"
marker_styles = ["o", "x", "^", "s", "*"]
y_labels = {
    metrics[0]: r"$\hat{J}_r(\pi)$",
    metrics[1]: r"$\hat{C}(\pi)$",
}
for i, ax in enumerate(axes.flatten()):
    so.Plot(
        last_steps,
        x="lambda",
        marker="magnitude",
        color="magnitude",
        y=metrics[i],
    ).add(
        so.Line(linewidth=1.0, pointsize=2.5, edgewidth=0.1),
        so.Agg("median"),
    ).add(so.Band(alpha=0.15), so.Est("median", errorbar="ci"), legend=False).scale(
        color=so.Nominal(
            values=[
                "#5F4690",
                "#1D6996",
                "#38A6A5",
                "#0F8554",
                "#73AF48",
                "#EDAD08",
                "#E17C05",
                "#CC503E",
                "#94346E",
                "#6F4070",
                "#994E95",
                "#666666",
            ],
        ),
        marker=so.Nominal(values=marker_styles),
    ).label(
        x="$\lambda$",
        y=lambda name: y_labels[name],
    ).theme(axes_style("ticks")).on(ax).plot()


def text(s):
    return f"${set_size} = {s}$"


axes = fig.get_axes()
for i, ax in enumerate(axes):
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    ax.axvline(0.6, color="black", linewidth=1.0, zorder=1)
    if i == 1:
        ax.axhline(y=100, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
        yticks = ax.get_yticks()
        y_new = 100
        ax.set_yticks(list(yticks) + [y_new])
        ytick_labels = [f"{tick:.0f}" for tick in yticks] + ["Budget"]
        ax.set_yticklabels(ytick_labels)
        ax.set_ylim(0, 400)
    arrowprops = dict(
        arrowstyle="-|>", color="black", linewidth=1.25, connectionstyle="arc3,rad=0.2"
    )
    if i == 0:
        ax.annotate(
            "Good performance",
            xy=(0.6, 675),
            xytext=(0.3, 400),
            arrowprops=arrowprops,
            ha="center",
        )
    else:
        ax.annotate(
            "Constraint satisfied",
            xy=(0.6, 25),
            xytext=(0.75, 200),
            arrowprops=arrowprops,
            ha="center",
        )
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
legend = fig.legends.pop(0)
fig.legends = []
fig.legend(
    legend.legend_handles,
    [text(t.get_text()) for t in legend.texts],
    loc="center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=8,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.5,
    handlelength=1.5,
)

fig.savefig("tune-lambda-cartpole-4.pdf")
