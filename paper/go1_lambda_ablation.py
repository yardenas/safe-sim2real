# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
from seaborn import axes_style
from tueplots import bundles, figsizes, fontsizes

warnings.filterwarnings("ignore")


# %%


def load_evaluation(data_path):
    data = pd.read_csv(data_path)
    data = data.fillna(20)
    return data


data = load_evaluation("joint_limits_experiment.csv")

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
).scale(y="log").label(
    x="$\lambda$",
    y=r"$\hat{C}_{p^\star}(\pi)$",
).theme(axes_style("ticks")).on(fig).plot()

ax = fig.get_axes()[0]
budget = 20
ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
ax.axhline(y=budget, xmax=1.0, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
ax.axhline(
    y=12, xmax=1.0, color="#1D6996", linewidth=1.25, zorder=0, label="Simulation"
)
yticks = ax.get_yticks()
ax.set_yticks(list(yticks) + [budget])
ytick_labels = [f"{tick:.0f}" for tick in yticks] + ["Budget"]
ax.set_yticklabels(ytick_labels)
ax.set_ylim(1, 450)
ax.set_xticks([0.0, 0.025, 0.05, 0.075, 0.1])
xticks = ax.get_xticks()
xtick_labels = [tick for tick in xticks]
xtick_labels[0] = f"{xticks[0]}\n No Pessimism"
# ax.yaxis.set_minor_locator(plt.NullLocator())
ax.tick_params(axis="y", which="minor", width=0.5, length=2)
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


def add_arrow(ax, x, min_val, max_val, times):
    midpoint = (min_val + max_val) / 4
    ax.annotate(
        "",
        xy=(x, max_val),
        xytext=(x, min_val),
        arrowprops=dict(arrowstyle="<->", shrinkA=2, shrinkB=2),
        textcoords="data",
        xycoords="data",
        color="black",
    )
    ax.annotate(
        r"$\times $" + f"{times:.2f}",
        xy=(x, midpoint),
        xytext=(0.5, 0),
        fontsize=base_font_size,
        xycoords="data",
        textcoords="offset points",
    )


cost = r"\hat{C}_{p^\star}(\pi)"

ax.annotate(
    f"${cost} =\; ${y_0:.2f}",
    xy=(0.05, y_0),
    xytext=(0.025, y_0 + 200),
    fontsize=base_font_size,
    va="center",
)
for i, lambda_val in enumerate([0.075]):
    cost = np.median(costs[lambdas == lambda_val])
    add_arrow(ax, lambda_val, cost, y_0, y_0 / cost)
add_arrow(ax, 0, budget, y_0, y_0 / budget)


ax.set_ylim(0.3, 800)
ax.set_title("Unitree Go1")
fig.savefig("go1-lambda-ablation.pdf")

# %%

tmp_constraint = constraint[constraint["lambda"].isin([0, 0.075])]
simulation_constraint = pd.DataFrame(
    {
        "trial_id": [0] * 5,
        "lambda": ["Simulation"] * 5,
        "cumulative_cost": [13.05, 20.20, 11.38, 9.56, 8.6],
    }
)
tmp_constraint = pd.concat([tmp_constraint, simulation_constraint])
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=1, rel_width=0.5))
fig = plt.figure()
metrics = ["cumulative_cost"]
y_labels = {
    metrics[0]: r"$\hat{C}(\pi)$",
}
marker_styles = ["o", "x", "^", "s", "*"]
so.Plot(tmp_constraint, x="lambda", y=metrics[0], color="lambda").add(
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
        order=[0.075, "Simulation", 0.0],
    ),
    x=so.Nominal(values=[0.075, "Simulation", 0.0], order=[0.075, "Simulation", 0.0]),
).label(y=lambda name: y_labels[name], x="").theme(axes_style("ticks")).on(fig).plot()

budget = 20
axes = fig.get_axes()
for i, ax in enumerate(axes):
    ax.grid(True, linewidth=0.5, c="gainsboro", axis="y", zorder=0)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.axhline(y=budget, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
    yticks = ax.get_yticks()
    y_new = budget
    ax.set_yticks(list(yticks) + [y_new])
    ytick_labels = [f"{tick:.0f}" for tick in yticks] + ["Budget"]
    ax.set_yticklabels(ytick_labels)
    ax.set_ylim(0, 360)
legend = fig.legends.pop(0)
fig.legends = []
text = {
    "0.0": "{{\sf Domain Randomization}} (real)",
    "0.075": "{{\sf PTSD}} (real)",
    "Simulation": "{{\sf Domain Randomization}} (simulated)",
}
hatches = ["", "//", "\\", "*", "o"]
for i, bar in enumerate(ax.patches):
    hatch_pattern = hatches[i]
    bar.set_hatch(hatch_pattern)
fig.legend(
    legend.legend_handles,
    [text[t.get_text()] for t in legend.texts],
    loc="center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=6,
    frameon=False,
    handletextpad=0.5,
    handlelength=1.0,
)
fig.savefig("go1-safety.pdf")
