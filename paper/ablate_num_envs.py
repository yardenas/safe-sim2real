# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so
import wandb
from seaborn import axes_style
from tueplots import bundles, figsizes

warnings.filterwarnings("ignore")


# %%
api = wandb.Api()


def load_evaluation(data, seed, num_envs, runtime):
    if data.index.name is None or data.index.name != "step":
        data.rename(columns={"_step": "step"}, inplace=True)
        data.set_index("step", inplace=True)
    data = data[data.columns[data.columns.str.startswith("eval")]].dropna()
    data["seed"] = seed
    data["num_envs"] = num_envs
    data["runtime"] = runtime
    return data


def config_to_num_envs(config):
    num_envs = config["agent"]["propagation"]["num_envs"]
    return num_envs


def handle_run(run):
    metrics = run.history(
        keys=["eval/episode_reward", "eval/episode_cost", "_step"],
        x_axis="_step",
        pandas=False,
    )
    config = run.config
    metrics = pd.DataFrame(metrics)
    num_envs = config_to_num_envs(config)
    seed = config["training"]["seed"]
    runtime = run.summary["_runtime"]
    return metrics, seed, num_envs, runtime


def walk_wandb_runs(project, filters):
    runs = api.runs(project, filters=filters)
    print(f"Fetching {len(runs)} runs")
    for run in runs:
        out = handle_run(run)
        if out is None:
            continue
        yield out


filters = {"display_name": {"$regex": "apr16-performance"}}

data = pd.concat(
    [
        load_evaluation(*summary_files_data)
        for summary_files_data in walk_wandb_runs("yardas/ss2r", filters)
    ]
)
# %%
last_steps = data.groupby(["seed", "num_envs"]).tail(1).reset_index()
aggregated_data = (
    last_steps.groupby(["num_envs"])[
        ["eval/episode_reward", "eval/episode_cost", "runtime"]
    ]
    .median()
    .reset_index()
)
aggregated_data = aggregated_data[
    aggregated_data["num_envs"].isin([1, 2, 4, 8, 16, 32, 64, 128])
]
# %%
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=1))
plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
fig, axes = plt.subplots(1, 2)
metrics = ["eval/episode_reward", "eval/episode_cost"]
y_labels = {
    metrics[0]: r"$\hat{J}_r(\pi)$",
    metrics[1]: r"$\hat{C}(\pi)$",
}
marker_styles = ["o", "x", "^", "s", "*"]
for i, ax in enumerate(axes.flatten()):
    so.Plot(
        aggregated_data,
        x="runtime",
        marker="num_envs",
        color="num_envs",
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
        # x="log",
    ).label(
        x="\# Environments",
        y=lambda name: y_labels[name],
    ).theme(axes_style("ticks")).on(ax).plot()

axes = fig.get_axes()
for i, ax in enumerate(axes):
    ax.grid(True, linewidth=0.5, c="gainsboro", axis="both", zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
# ax.annotate(
#     "$\downarrow$\#Workers",
#     xy=(1.7e8 * to_comm, 1500),
#     xytext=(0.1e8 * to_comm, 8700),
#     arrowprops=dict(facecolor="red", arrowstyle="->", lw=0.8),
#     color="black",
# )
# fig.legend(
#     regex_to_category.values(),
#     loc="upper right",
#     bbox_to_anchor=(1.33, 1.0),
#     ncol=1,
#     frameon=False,
#     columnspacing=0.5,
#     handletextpad=0.35,
# )
fig.savefig("ablate-num-envs.pdf")

# %%
last_steps = data.groupby(["seed", "num_envs"]).tail(1).reset_index()
aggregated_data = last_steps[last_steps["num_envs"].isin([1, 2, 4, 8, 16, 32, 64, 128])]
aggregated_data["runtime_normalized"] = (
    aggregated_data["runtime"] / aggregated_data["runtime"].min()
)
aggregated_data = aggregated_data.reset_index()
aggregated_data = pd.melt(
    aggregated_data,
    id_vars=["step", "seed", "num_envs"],  # columns to keep as-is
    value_vars=[
        "eval/episode_reward",
        "eval/episode_cost",
        "runtime_normalized",
    ],  # columns to melt
    var_name="metric_name",
    value_name="value",
)
# %%

theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=3, height_to_width_ratio=0.8))
fig = plt.figure()
y_labels["runtime_normalized"] = "Normalized Runtime"
so.Plot(
    aggregated_data,
    x="num_envs",
    y="value",
).facet(col="metric_name").share(x=True, y=False).add(
    so.Line(
        linewidth=1.0,
        pointsize=2.5,
        edgewidth=0.1,
        marker=marker_styles[0],
        color="#5F4690",
    ),
    so.Agg("mean"),
).add(
    so.Band(alpha=0.15, color="#5F4690"), so.Est("mean", errorbar="se"), legend=False
).scale(x="log2").label(
    x="\# parameterizations $n$", title=lambda name: y_labels[name], y=""
).theme(axes_style("ticks")).on(fig).plot()


axes = fig.get_axes()
for ax in axes:
    ax.tick_params(axis="x", which="minor", width=0.5, length=2)
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
axes[0].set_ylim(0, 1000)
axes[1].axhline(y=100, xmax=1.0, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
yticks = axes[1].get_yticks()
ytick_labels = [f"{tick:.0f}" if tick != 100 else "Budget" for tick in yticks]
axes[1].set_yticklabels(ytick_labels)
yticks = axes[-1].get_yticks()
yticks = list(yticks)
labels = [f"$\\times${tick:.1f}" for tick in yticks]
axes[-1].set_yticklabels(labels)

fig.savefig("ablate-num-envs.pdf")
