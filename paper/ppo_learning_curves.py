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


def load_evaluation(data, seed, lambda_):
    if data.index.name is None or data.index.name != "step":
        data.rename(columns={"_step": "step"}, inplace=True)
        data.set_index("step", inplace=True)
    data = data[data.columns[data.columns.str.startswith("eval")]].dropna()
    data["seed"] = seed
    data["lambda"] = lambda_
    return data


def config_to_lambda(config):
    lambda_ = config["agent"]["penalizer"]["lambda_"]
    return lambda_


def handle_run(run):
    metrics = run.history(
        keys=["eval/episode_reward", "eval/episode_cost", "_step"],
        x_axis="_step",
        pandas=False,
    )
    safe = run.config["training"]["safe"]
    entropy = run.config["agent"]["entropy_cost"]
    if not safe or entropy != 0.025:
        return
    config = run.config
    metrics = pd.DataFrame(metrics)
    lambda_ = config_to_lambda(config)
    seed = config["training"]["seed"]
    return metrics, seed, lambda_


def walk_wandb_runs(project, filters):
    runs = api.runs(project, filters=filters)
    print(f"Fetching {len(runs)} runs")
    for run in runs:
        out = handle_run(run)
        if out is None:
            continue
        yield out


filters = {"display_name": {"$regex": "apr11-ppo-entropy"}}
data = pd.concat(
    [
        load_evaluation(*summary_files_data)
        for summary_files_data in walk_wandb_runs("yardas/ss2r", filters)
    ]
)
# %%

data = data[data["lambda"].isin([0.0, 0.01, 1.0, 0.001])]

# %%
metrics = ["eval/episode_reward", "eval/episode_cost"]
y_labels = {
    metrics[0]: r"$\hat{J}(\tilde{\pi})$",
    metrics[1]: r"$\hat{C}(\tilde{\pi})$",
}
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=2))
plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
fig, axes = plt.subplots(1, 2)
marker_styles = ["o", "x", "^", "s", "*"]
colors = [
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
]
text = {
    "1.0": r"$\lambda = 1$",
    "0.01": r"$\lambda = 0.01$",
    "0.001": r"$\lambda = 0.001$",
    "0.0": r"$\lambda = 0$",
}
for i, ax in enumerate(axes.flatten()):
    so.Plot(
        data,
        x="step",
        marker="lambda",
        color="lambda",
        y=metrics[i],
    ).add(
        so.Line(linewidth=1.0, pointsize=2.5, edgewidth=0.1),
        so.Agg("mean"),
    ).add(so.Band(alpha=0.15), so.Est("mean", errorbar="se"), legend=False).scale(
        color=so.Nominal(
            values=colors,
            order=[1.0, 0.01, 0.001, 0.0],
        ),
        marker=so.Nominal(values=marker_styles, order=[1.0, 0.01, 0.001, 0.0]),
    ).label(
        x="Training Steps",
        y=lambda name: y_labels[name],
    ).theme(axes_style("ticks")).on(ax).plot()

axes = fig.get_axes()
for i, ax in enumerate(axes):
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    if i == 1:
        ax.axhline(y=100, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
        ax.set_ylim(0, 400)
        yticks = ax.get_yticks()
        y_new = 100
        ax.set_yticks(list(yticks) + [y_new])
        ytick_labels = [f"${tick:.0f}$" for tick in yticks] + ["Budget"]
        ax.set_yticklabels(ytick_labels)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
legend = fig.legends.pop(0)
fig.legends = []
fig.legend(
    legend.legend_handles,
    [text[t.get_text()] for t in legend.texts],
    loc="center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=8,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.5,
    handlelength=1.5,
)

fig.savefig("ppo-curves.pdf")
