# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd
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
    so.Agg("mean"),
).add(
    so.Band(alpha=0.15, color="#5F4690"),
    so.Est("mean", errorbar="se"),
    legend=False,
).label(
    x="$\lambda$",
    y=r"$\hat{C}(\tilde{\pi})$",
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

base_font_size = fontsizes.neurips2024()["font.size"]


cost = r"$\hat{C}(\tilde{\pi})$"


ax.set_ylim(-1, 42)
ax.set_title("RaceCar")

ax.annotate(
    "Constraint \nin simulation",
    xy=(30, 13),
    xytext=(0.01, 5.05),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#1D6996",
        linewidth=0.75,
        connectionstyle="arc3,rad=0.2",
    ),
    bbox=dict(pad=-10, facecolor="none", edgecolor="none"),
    fontsize=base_font_size,
    va="center",
    color="#1D6996",
)


fig.savefig("rccar-lambda-ablation.pdf")

# %%
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=2, rel_width=0.6))
fig, axes = plt.subplots(1, 2)
metrics = ["cumulative_reward", "cumulative_cost"]
y_labels = {
    metrics[0]: r"$\hat{J}(\tilde{\pi})$",
    metrics[1]: r"$\hat{C}(\tilde{\pi})$",
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
