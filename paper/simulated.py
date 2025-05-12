# %%
import warnings
from math import floor, log10

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


def handle_run(run):
    metrics = run.history(
        keys=["eval/episode_reward", "eval/episode_cost", "_step"],
        x_axis="_step",
        pandas=False,
    )
    safe = run.config["training"]["safe"]
    if not safe:
        return
    config = run.config
    environment = config["environment"]["task_name"]
    metrics = pd.DataFrame(metrics)
    if environment == "rccar":
        environment = "RaceCar"
    if environment == "go_to_goal":
        environment = "PointGoal2"
    category = config_to_category(config)
    seed = config["training"]["seed"]
    if environment == "PointGoal2" and category == "ptsd":
        if config["agent"]["cost_robustness"]["cost_penalty"] != 0.175:
            return
    if environment == "PointGoal2" and category == "ramu":
        if "may7" not in run.config["wandb"]["notes"]:
            return
    return metrics, seed, category, environment


def walk_wandb_runs(project, filters):
    runs = api.runs(project, filters=filters)
    print(f"Fetching {len(runs)} runs")
    for run in runs:
        out = handle_run(run)
        if out is None:
            continue
        yield out


filters = {
    "display_name": {
        "$regex": "apr01-nominal-aga$|apr01-dr-aga$|apr01-ptsd-aga|mar28-simple-aga|apr01-ramu-aga|may7-g2g-ramu|apr11-g2g-dr|apr11-g2g-nominal|apr11-g2g-simple-aga$|0.175.*apr16-g2g-tune$"
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

budgets = {
    "RaceCar": 5,
    "SafeCartpoleSwingup": 100,
    "SafeHumanoidWalk": 100,
    "SafeQuadrupedRun": 100,
    "SafeWalkerWalk": 100,
    "PointGoal2": 25,
}

aggregated_data["normalized_episode_cost"] = aggregated_data.apply(
    lambda row: row["eval/episode_cost"] / budgets[row["environment"]], axis=1
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


def sci_notation(
    num,
    decimal_digits=1,
    precision=None,
    exponent=None,
    disable_scaling=False,
    prefix="",
):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num == 0:
        return f"${prefix}0$"
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    if disable_scaling:
        return rf"${prefix}{coeff * 10**exponent:.{precision}f}$"
    return rf"${prefix}{coeff:.{precision}f}\times10^{{{exponent:d}}}$"


def fill_unsafe(ax, x):
    lim = ax.get_xlim()[1]
    ax.axvspan(x + lim / 200, lim, color="red", alpha=0.1)


theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=3, height_to_width_ratio=1.4))
plt.rcParams.update({"text.latex.preamble": r"\usepackage{xfrac}\usepackage{times}"})
fig = plt.figure()
marker_styles = ["o", "x", "^", "s", "*"]
so.Plot(
    aggregated_data[aggregated_data["category"] != "simple"],
    x="normalized_episode_cost",
    y="normalized_episode_reward",
    marker="category",
    color="category",
).facet(col="environment", wrap=3, order=list(budgets.keys())).share(
    y=True, x=False
).add(so.Range(), so.Est(errorbar="se")).add(
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
        order=["ptsd", "ramu", "dr", "nominal"],
    ),
    marker=so.Nominal(values=marker_styles, order=["ptsd", "ramu", "dr", "nominal"]),
).label(
    x=r"$\sfrac{\hat{C}(\tilde{\pi})}{d}$",
    y=r"$\hat{J}(\tilde{\pi})$",
    title=lambda x: x.strip("Safe"),
).theme(axes_style("ticks")).on(fig).plot()
axes = fig.get_axes()
optimum = (
    aggregated_data[aggregated_data["category"] == "simple"]
    .groupby("environment")[["normalized_episode_reward", "normalized_episode_cost"]]
    .mean()
)

opts = {
    k: (
        optimum.loc[k]["normalized_episode_cost"],
        optimum.loc[k]["normalized_episode_reward"],
    )
    for k in budgets.keys()
    if k in optimum.index
}
scale = lambda x: [y * 1 / 1.1 for y in x]
for i, (ax, env_name) in enumerate(zip(axes, budgets.keys())):
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    draw_optimum(ax, *opts[env_name])
    fill_unsafe(ax, 1.0)
    ax.axvline(
        x=1.0,
        color="black",
        alpha=0.2,
        linestyle=(0, (1, 1)),
        linewidth=1.25,
    )
    xticks = xlims = ax.get_xlim()
    xticks = ax.get_xticks()
    xticks = list(xticks) + [1.0]
    # if i == 0:
    #     xticks.remove(2.5)
    xticks = sorted(set(xticks))
    xtick_labels = [
        sci_notation(tick, disable_scaling=True, prefix=r"\times")
        if tick != 1.0
        else "Budget"
        for tick in xticks
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(xlims)
    ax.set_ylim(-0.05, 1.075)

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
    "ptsd": "\\textsf{{\\textbf{{SPiDR}}}}",
    "dr": "\sf Domain Randomization",
    "nominal": "\sf Nominal",
    "tightening": "\sf Tightening",
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
    bbox_to_anchor=(0.5, 1.025),
    ncol=6,
    frameon=False,
    handletextpad=0.25,
    handlelength=1.0,
)

fig.savefig("simulated.pdf")
