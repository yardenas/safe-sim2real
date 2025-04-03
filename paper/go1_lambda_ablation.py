# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
from seaborn import axes_style
from tueplots import bundles, figsizes

warnings.filterwarnings("ignore")


# %%


def load_evaluation(data_path):
    data = pd.read_csv(data_path)
    data = data.drop(columns=["mean", "Unnamed: 6", "Unnamed: 7", "budget"])
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
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=1))
plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
fig = plt.figure()
marker_styles = ["o", "x", "^", "s", "*"]
so.Plot(constraint, x="lambda", y="cumulative_cost").add(
    so.Line(linewidth=1.0, pointsize=2.5, edgewidth=0.1, marker="x", color="#5F4690"),
    so.Agg("median"),
).add(
    so.Band(alpha=0.15, color="#5F4690"), so.Est("median", errorbar="ci"), legend=False
).scale(y="log").label(
    x="$\lambda$",
    y=r"$\hat{C}_{p^\star}(\pi)$",
).theme(axes_style("ticks")).on(fig).plot()

ax = fig.get_axes()[0]
budget = 20
ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
ax.axhline(y=budget, xmax=1.0, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
ax.axhline(
    y=12, xmax=1.0, color="#EDAD08", linewidth=1.25, zorder=0, label="Simulation"
)
yticks = ax.get_yticks()
ax.set_yticks(list(yticks) + [budget])
ytick_labels = [f"{tick:.0f}" for tick in yticks] + ["Budget"]
ax.set_yticklabels(ytick_labels)
ax.set_ylim(1, 400)
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
        fontsize=9,
        xycoords="data",
        textcoords="offset points",
    )


ax.annotate(
    "Constraint in simulation",
    xy=(0.025, 12),
    xytext=(0.01, 4),
    arrowprops=dict(
        arrowstyle="-|>", color="black", linewidth=0.75, connectionstyle="arc3,rad=0.2"
    ),
    fontsize=9,
    va="center",
)

for i, lambda_val in enumerate([0.05, 0.1]):
    cost = np.median(costs[lambdas == lambda_val])
    add_arrow(ax, lambda_val, cost, y_0, y_0 / cost)
add_arrow(ax, 0, budget, y_0, y_0 / budget)


fig.savefig("go1-lambda-ablation.pdf")
