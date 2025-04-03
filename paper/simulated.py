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


def load_evaluation(data, seed, category, environment):
    if data.index.name is None or data.index.name != "step":
        data.rename(columns={"_step": "step"}, inplace=True)
        data.set_index("step", inplace=True)
    data = data[data.columns[data.columns.str.startswith("eval")]].dropna()
    data["seed"] = seed
    data["category"] = category
    data["environment"] = environment
    return data


def config_to_category(config):
    if "cost_robustness" in config["agent"]:
        name = config["agent"]["cost_robustness"]["name"]
        if name == "ramu":
            return "ramu"
        return "ptsd"
    elif config["training"]["train_domain_randomization"]:
        return "dr"
    elif config["training"]["eval_domain_randomization"]:
        return "nominal"
    else:
        return "simple"


def walk_wandb_runs(project, filters):
    runs = api.runs(project, filters=filters)
    print(f"Fetching {len(runs)} runs")
    for run in runs:
        metrics = run.history(
            keys=["eval/episode_reward", "eval/episode_cost", "_step"],
            x_axis="_step",
            pandas=False,
        )
        safe = run.config["training"]["safe"]
        if not safe:
            continue
        metrics = pd.DataFrame(metrics)
        seed = run.config["training"]["seed"]
        category = config_to_category(run.config)
        environment = run.config["environment"]["task_name"]
        if environment == "rccar":
            environment = "RaceCar"
        yield metrics, seed, category, environment


filters = {
    "display_name": {
        "$regex": "apr01-nominal-aga$|apr01-dr-aga$|apr01-ptsd-aga|mar28-simple-aga|apr01-ramu-aga"
    }
}

data = pd.concat(
    [
        load_evaluation(*summary_files_data)
        for summary_files_data in walk_wandb_runs("yardas/ss2r", filters)
    ]
)
# %%

last_steps = data.groupby(["category", "environment", "seed"]).tail(1).reset_index()
aggregated_data = (
    last_steps.groupby(["category", "environment"])[
        ["eval/episode_reward", "eval/episode_cost"]
    ]
    .mean()
    .reset_index()
)
reference_values = (
    aggregated_data[aggregated_data["category"] == "simple"]
    .set_index("environment")["eval/episode_reward"]
    .to_dict()
)
# Normalizing the eval/episode_reward for each row based on the corresponding "simple" value
aggregated_data["normalized_episode_reward"] = aggregated_data.apply(
    lambda row: row["eval/episode_reward"] / reference_values[row["environment"]],
    axis=1,
)


# %%
def draw_optimum(ax, x, y):
    ax.scatter(
        x,
        y,
        color="yellow",
        s=35,
        marker="*",
        edgecolor="black",
        linewidth=0.5,
        label="Optimum on $\mathcal{{M}}^\star$",
    )


def fill_unsafe(ax, x):
    lim = ax.get_xlim()[1]
    ax.axvspan(x + lim / 100, lim, color="red", alpha=0.1)


theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=5, height_to_width_ratio=1.3))
fig = plt.figure()
marker_styles = ["o", "x", "^", "s", "*"]
so.Plot(
    aggregated_data[aggregated_data["category"] != "simple"],
    x="eval/episode_cost",
    y="normalized_episode_reward",
    marker="category",
    color="category",
).facet(col="environment").share(y=True, x=False).add(
    so.Range(), so.Est(errorbar="se")
).add(so.Dot(pointsize=3.5, edgewidth=0.1), legend=True).scale(
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
        order=["ptsd", "ramu", "dr", "nominal"],
    ),
    marker=so.Nominal(values=marker_styles, order=["ptsd", "ramu", "dr", "nominal"]),
).label(
    x=r"$\hat{C}(\pi)$",
    y=r"$\hat{J}(\pi)$",
    title=lambda x: x.strip("Safe"),
).theme(axes_style("ticks")).on(fig).plot()
axes = fig.get_axes()
optimum = (
    aggregated_data[aggregated_data["category"] == "simple"]
    .groupby("environment")[["normalized_episode_reward", "eval/episode_cost"]]
    .median()
)
budgets = {
    "RaceCar": 5,
    "SafeCartpoleSwingup": 100,
    "SafeHumanoidWalk": 100,
    "SafeQuadrupedRun": 100,
    "SafeWalkerWalk": 100,
}
opts = {
    k: (
        optimum.loc[k]["eval/episode_cost"],
        optimum.loc[k]["normalized_episode_reward"],
    )
    for k in budgets.keys()
}
scale = lambda x: [y * 1 / 1.1 for y in x]
for i, (ax, env_name) in enumerate(zip(axes, budgets.keys())):
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    draw_optimum(ax, *opts[env_name])
    fill_unsafe(ax, budgets[env_name])
    ax.axvline(
        x=budgets[env_name],
        color="black",
        alpha=0.2,
        linestyle=(0, (1, 1)),
        linewidth=1.25,
    )
    xticks = xlims = ax.get_xlim()
    xticks = ax.get_xticks()
    xticks = list(xticks) + [budgets[env_name]]
    xticks = sorted(set(xticks))
    if i == 0:
        xticks.remove(10)
    xtick_labels = [
        str(int(tick)) if tick != budgets[env_name] else "Budget" for tick in xticks
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(xlims)

axes[0].annotate(
    "Unsafe",
    xy=(10, 0.75),
    va="center",
)
axes[0].annotate(
    "",
    xy=(24, 0.67),
    xytext=(16, 0.71),
    arrowprops=dict(
        arrowstyle="-|>", color="black", linewidth=0.75, connectionstyle="arc3,rad=0.2"
    ),
    va="center",
)
axes[0].annotate(
    "",
    xy=(21, 0.58),
    xytext=(16, 0.7),
    arrowprops=dict(
        arrowstyle="-|>", color="black", linewidth=0.75, connectionstyle="arc3,rad=0.2"
    ),
    va="center",
)
legend = fig.legends.pop(0)

text = {
    "ramu": "\sf RAMU",
    "ptsd": "\sf PTSD",
    "dr": "\sf Domain Randomization",
    "nominal": "\sf Nominal",
}

optimum_handle, optimum_label = ax.get_legend_handles_labels()

handles = []
for handle, marker in zip(legend.legend_handles, marker_styles):
    handle.set_marker(marker)
    handle.set_linewidth(0)
    handle.set_markersize(3.5)
    handles.append(handle)
fig.legend(
    handles + optimum_handle,
    [text[t.get_text()] for t in legend.texts] + optimum_label,
    loc="center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=5,
    frameon=False,
    handletextpad=0.25,
    handlelength=1.0,
)


fig.savefig("simulated.pdf")
