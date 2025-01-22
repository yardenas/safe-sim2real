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
        if "ucb" in run.name:
            penalty = run.config["agent"]["robustness"]["cost_penalty"]
            if penalty != 12.5:
                continue
        metrics = pd.DataFrame(metrics)
        seed = run.config["training"]["seed"]
        category = run.config["agent"].get("robustness", {"name": "neutral"})["name"]
        if category == "cvar":
            category = "ucb_cost"
        environment = run.config["environment"]["task_name"]
        yield metrics, seed, category, environment


filters = {
    "display_name": {
        "$regex": "dec6-sim2sim-dr|nov28-sim2sim-all|nov28-sim2sim-agaggaa|nov25-sim2sim-ucb|nov25-sim2sim-neutral-wtf"
    },
}

data = pd.concat(
    [
        load_evaluation(*summary_files_data)
        for summary_files_data in walk_wandb_runs("yardas/ss2r", filters)
    ]
)
# %%

last_steps = data.groupby(["category", "seed", "environment"]).tail(2).reset_index()
aggregated_data = (
    last_steps.groupby(["category", "seed", "environment"])[
        ["eval/episode_reward", "eval/episode_cost"]
    ]
    .median()
    .reset_index()
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
        label="Optimum on $p^\star$",
    )


def fill_unsafe(ax, x):
    lim = ax.get_xlim()[1]
    ax.axvspan(x + lim / 100, lim, color="red", alpha=0.1)


theme = bundles.jmlr2001()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(figsizes.jmlr2001(nrows=1, ncols=3, height_to_width_ratio=0.8))
fig = plt.figure()
marker_styles = ["o", "x", "^", "s", "*"]
titles = {
    "cartpole_swingup_safe": "CartpoleBalance",
    "humanoid_safe": "Humanoid",
    "rccar": "RaceCar",
}
so.Plot(
    aggregated_data,
    x="eval/episode_cost",
    y="eval/episode_reward",
    marker="category",
    color="category",
).facet(col="environment").share(y=False, x=False).add(
    so.Dot(pointsize=3.5, edgewidth=0.1), legend=True
).scale(
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
        order=["ucb_cost", "neutral"],
    ),
    marker=so.Nominal(values=marker_styles, order=["ucb_cost", "neutral"]),
).label(x=r"$\hat{C}(\pi)$", y=r"$\hat{J}(\pi)$", title=lambda x: titles[x]).theme(
    axes_style("ticks")
).on(fig).plot()
axes = fig.get_axes()
optimum = [(79, 988), (28, 12172), (3, 177)]
safe = [100, 100, 8]
scale = lambda x: [y * 1 / 1.1 for y in x]
for i, (ax, opt, sval) in enumerate(zip(axes, optimum, safe)):
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    draw_optimum(ax, *opt)
    fill_unsafe(ax, sval)
    ax.axvline(x=sval, color="black", alpha=0.2, linestyle=(0, (1, 1)), linewidth=1.25)
    xticks = xlims = ax.get_xlim()
    xticks = ax.get_xticks()
    xticks = list(xticks) + [sval]
    xticks = sorted(set(xticks))
    xtick_labels = [
        str(int(tick)) if tick != sval else f"{sval}\nBudget" for tick in xticks
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(xlims)
legend = fig.legends.pop(0)

text = {
    "ucb_cost": "\sf Cost UCB",
    "neutral": "\sf Neutral",
}

optimum_handle, optimum_label = ax.get_legend_handles_labels()

fig.legend(
    legend.legend_handles + optimum_handle,
    [text[t.get_text()] for t in legend.texts] + optimum_label,
    loc="center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=5,
    frameon=False,
    handletextpad=0.0,
)
fig.savefig("preliminary.pdf")
