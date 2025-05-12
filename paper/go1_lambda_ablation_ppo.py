# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so
from seaborn import axes_style
from tueplots import bundles, figsizes

warnings.filterwarnings("ignore")


# %%


def load_evaluation(data_path):
    data = pd.read_csv(data_path)
    return data


data = load_evaluation("joint_limits_experiment_ppo.csv")
simulation_constraint = pd.DataFrame(
    {
        "trial_id": [0] * 5,
        "policy": [""] * 5,
        "lambda": ["Simulation"] * 5,
        "cumulative_cost": [17.49721, 17.37938, 17.92791, 18.18569, 19.35582],
    }
)
data = pd.concat([data, simulation_constraint])

go1_dr_cost = data[(data["lambda"] == 0.0)]["cumulative_cost"].mean()
go1_ptsd_cost = data[(data["lambda"] == 0.125)]["cumulative_cost"].mean()
go1_ratio = go1_dr_cost / go1_ptsd_cost

# %%
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=1, rel_width=0.33))
fig = plt.figure()
metrics = ["cumulative_cost"]
y_labels = {
    metrics[0]: r"$\hat{C}(\tilde{\pi})$",
}
marker_styles = ["o", "x", "^", "s", "*"]
so.Plot(data, x="lambda", y=metrics[0], color="lambda").add(
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
        order=[0.125, "Simulation", 0.0],
    ),
    x=so.Nominal(values=[0.125, "Simulation", 0.0], order=[0.125, "Simulation", 0.0]),
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
    "0.125": "{{\\textsf{{\\textbf{{SPiDR}}}}}} (real)",
    "0.0": "{{\sf Domain Randomization}} (real)",
    "Simulation": "{{\sf Domain Randomization}} (simulation)",
}
hatches = ["", "//", "\\", "*", "o"]
for i, bar in enumerate(ax.patches):
    hatch_pattern = hatches[i]
    bar.set_hatch(hatch_pattern)
fig.legend(
    legend.legend_handles,
    [text[t.get_text()] for t in legend.texts],
    loc="center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=1,
    frameon=False,
    handletextpad=0.5,
    handlelength=1.0,
)
fig.savefig("go1-lambda-ablation-ppo.pdf")
