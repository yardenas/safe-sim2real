# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style
from tueplots import bundles, figsizes, fontsizes

warnings.filterwarnings("ignore")


# %%


def load_evaluation(data_path):
    data = pd.read_csv(data_path)
    data = data.drop(columns=["trial_id"])
    return data


data = load_evaluation("rccar_experiment.csv")
# data = data.groupby(["lambda", "policy"]).mean().reset_index()
# data = data.drop(columns=["policy"])
# data = data.groupby(["lambda"]).median().reset_index()


# %%
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

# %%
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=1, rel_width=0.5))
plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

# for ax in axes:
#     ax.grid(True, linewidth=0.5, c="gainsboro")
data["cumulative_cost"] = data["cumulative_cost"] + 0.01
plot = sns.violinplot(
    data,
    x="lambda",
    y="cumulative_reward",
    hue="lambda",
    palette={
        0: colors[0],
        45: colors[1],
        55: colors[2],
        60: colors[3],
        65: colors[4],
    },
    cut=0,
    inner=None,
    legend=False,
    ax=axes[0],
)
sns.swarmplot(
    data=data,
    x="lambda",
    y="cumulative_reward",
    color="k",
    size=1.5,
    ax=axes[0],
    alpha=0.7,
)
plot = sns.violinplot(
    data,
    x="lambda",
    y="cumulative_cost",
    hue="lambda",
    palette={
        0: colors[0],
        45: colors[1],
        55: colors[2],
        60: colors[3],
        65: colors[4],
    },
    cut=0,
    inner=None,
    ax=axes[1],
    legend=False,
    log_scale=(False, True),
)
sns.swarmplot(
    data=data,
    x="lambda",
    y="cumulative_cost",
    color="k",
    size=1.5,
    ax=axes[1],
    alpha=0.7,
)

axes[0].set_ylabel(r"$\hat{J}(\pi)$")
axes[0].set_xlabel(r"$\lambda$")
axes[1].set_ylabel(r"$\hat{C}(\pi)$")
axes[1].set_xlabel(r"$\lambda$")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=6,
    frameon=False,
    handletextpad=0.25,
    handlelength=1.0,
)
fig.savefig("rccar-sim-to-real-paper-violin.pdf")


# %%
constraint = data.groupby(["lambda", "policy"]).mean()["cumulative_cost"].reset_index()
constraint = constraint.drop(columns="policy")


# %%
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=1, rel_width=0.5))
plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
fig = plt.figure()
marker_styles = ["o", "x", "^", "s", "*"]
so.Plot(constraint, x="lambda", y="cumulative_cost").add(
    so.Line(linewidth=1.0, pointsize=3.5, edgewidth=0.5, marker="x", color="#5F4690"),
    so.Agg("median"),
).add(
    so.Band(alpha=0.15, color="#5F4690"),
    so.Est("median", errorbar=("ci", 68)),
    legend=False,
).label(
    x="$\lambda$",
    y=r"$\hat{C}_{p^\star}(\pi)$",
).theme(axes_style("ticks")).on(fig).plot()

ax = fig.get_axes()[0]
budget = 15
ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
ax.axhline(y=budget, xmax=1.0, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
ax.axhline(
    y=13.4, xmax=1.0, color="#1D6996", linewidth=1.25, zorder=0, label="Simulation"
)
yticks = ax.get_yticks()
ax.set_yticks(list(yticks) + [budget])
ytick_labels = [f"{tick:.0f}" for tick in yticks] + ["Budget"]
ax.set_yticklabels(ytick_labels)
ax.set_ylim(-2, 60)
xticks = ax.get_xticks()
xtick_labels = [f"{tick:.0f}" for tick in xticks]
xtick_labels[1] = f"{xticks[1]:.0f}\n No Pessimism"
ax.set_xticklabels(xtick_labels)
lambdas = constraint["lambda"]
costs = constraint["cumulative_cost"]
y_0 = np.median(costs[lambdas == 0])
y_01 = np.median(costs[lambdas == 0.1])
reduction_factor = y_01 / y_0
x_0 = 0.0
x_01 = 0.1
limits = ax.get_xlim()
ax.axhline(
    y=y_0,
    xmin=(x_0 - limits[0]) / limits[1],
    xmax=1.0,
    color="black",
    alpha=0.4,
    linewidth=1.25,
    linestyle="dashed",
)

base_font_size = fontsizes.neurips2024()["font.size"]


cost = r"\hat{C}_{p^\star}(\pi)"

ax.annotate(
    f"${cost} =\; ${y_0:.2f}",
    xy=(20, y_0),
    xytext=(20, y_0 + 4.5),
    fontsize=base_font_size,
    va="center",
)

ax.set_ylim(-1, 42)
ax.set_title("RaceCar")


fig.savefig("rccar-lambda-ablation.pdf")

# %%
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=2, rel_width=0.6))
fig, axes = plt.subplots(1, 2)
metrics = ["cumulative_reward", "cumulative_cost"]
y_labels = {
    metrics[0]: r"$\hat{J}(\pi)$",
    metrics[1]: r"$\hat{C}(\pi)$",
}
marker_styles = ["o", "x", "^", "s", "*"]
for i, ax in enumerate(axes.flatten()):
    so.Plot(data, x="lambda", y=metrics[i], color="lambda").add(
        so.Bar(),
        so.Agg("mean"),
        legend=True,
    ).add(
        so.Range(color="k", linewidth=0.75),
        so.Est("mean", errorbar="se"),
        legend=False,
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
        ),
        x=so.Nominal(values=[0, 45, 55, 60, 65], order=[0, 45, 55, 60, 65]),
    ).label(y=lambda name: y_labels[name], x="").theme(axes_style("ticks")).on(
        ax
    ).plot()

budget = 15

for i, ax in enumerate(axes):
    ax.grid(True, linewidth=0.5, c="gainsboro", axis="y", zorder=0)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    if i >= 1:
        ax.axhline(y=budget, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
        yticks = ax.get_yticks()
        y_new = budget
        ax.set_yticks(list(yticks) + [y_new])
        ytick_labels = [f"{tick:.0f}" for tick in yticks] + ["Budget"]
        ax.set_yticklabels(ytick_labels)
legend = fig.legends.pop(0)
fig.legends = []
text = {
    "0": r"$\lambda = 0$",
    "45": r"$\lambda = 45$",
    "55": r"$\lambda = 55$",
    "60": r"$\lambda = 60$",
    "65": r"$\lambda = 65$",
}
fig.legend(
    legend.legend_handles,
    [text[t.get_text()] for t in legend.texts],
    loc="center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=6,
    frameon=False,
    handletextpad=0.5,
    columnspacing=0.75,
    handlelength=1.0,
)

fig.savefig("rccar-sim-to-real.pdf")
