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
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=2))
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
    ax.axvline(0.6, color="black", linewidth=1.0, zorder=100)
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
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
            xytext=(0.3, 411),
            arrowprops=arrowprops,
            ha="center",
        )
    else:
        ax.annotate(
            "Constraint\nsatisfied",
            xy=(0.60, 25),
            xytext=(0.76, 206),
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

fig.savefig("tune-lambda-cartpole.pdf")
