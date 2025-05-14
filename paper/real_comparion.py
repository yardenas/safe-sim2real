# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so
from seaborn import axes_style
from tueplots import bundles, figsizes

warnings.filterwarnings("ignore")


# %%


def load_evaluation_go1(data_path):
    data = pd.read_csv(data_path)
    data = data.fillna(20)
    data["environment"] = "Unitree Go1"
    return data


go1 = load_evaluation_go1("joint_limits_experiment.csv")


def load_evaluation_rccar(data_path):
    data = pd.read_csv(data_path)
    data = data.drop(columns=["trial_id"])
    data["environment"] = "RaceCar"
    return data


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

rccar = load_evaluation_rccar("rccar_experiment.csv")

# %%

go1_constraint = (
    go1.groupby(["lambda", "policy", "environment"])
    .mean()["cumulative_cost"]
    .reset_index()
)
go1_constraint = go1_constraint.drop(columns="policy")
go1_constraint = go1_constraint[go1_constraint["lambda"].isin([0, 0.075])]
go1_constraint["lambda"] = go1_constraint["lambda"].map({0: "dr", 0.075: "ptsd"})

go1_simulation_constraint = pd.DataFrame(
    {
        "trial_id": [0] * 5,
        "lambda": ["Simulation"] * 5,
        "cumulative_cost": [13.05, 20.20, 11.38, 9.56, 8.6],
        "environment": ["Unitree Go1"] * 5,
    }
)
go1_constraint = pd.concat([go1_constraint, go1_simulation_constraint])
rccar_constraint = (
    rccar.groupby(["lambda", "policy", "environment"])
    .mean()["cumulative_cost"]
    .reset_index()
)
rccar_simulation_constraint = pd.DataFrame(
    {
        "lambda": ["Simulation"] * 5,
        "cumulative_cost": [9.54, 8.37, 6.18, 6.42, 3.85],
        "environment": ["RaceCar"] * 5,
        "cumulative_reward": [159.73, 178.59, 175.76, 172.15, 179.195],
    }
)
rccar_constraint = rccar_constraint.drop(columns="policy")
rccar_constraint = rccar_constraint[rccar_constraint["lambda"].isin([0, 45])]
rccar_constraint["lambda"] = rccar_constraint["lambda"].map({0: "dr", 45: "ptsd"})
rccar_constraint = pd.concat([rccar_constraint, rccar_simulation_constraint])

constraint = pd.concat([go1_constraint, rccar_constraint])


# %%
rccar_plot = rccar[rccar["lambda"].isin([0, 45])]
rccar_plot["lambda"] = rccar_plot["lambda"].map({0: "dr", 45: "ptsd"})
rccar_plot = pd.concat([rccar_simulation_constraint, rccar_plot])


# %%
car_dr_cost = rccar_plot[(rccar_plot["lambda"] == "dr")]["cumulative_cost"].mean()
car_ptsd_cost = rccar_plot[(rccar_plot["lambda"] == "ptsd")]["cumulative_cost"].mean()
car_ratio = car_dr_cost / car_ptsd_cost
go1_dr_cost = go1_constraint[(go1_constraint["lambda"] == "dr")][
    "cumulative_cost"
].mean()
go1_ptsd_cost = go1_constraint[(go1_constraint["lambda"] == "ptsd")][
    "cumulative_cost"
].mean()
go1_ratio = go1_dr_cost / go1_ptsd_cost
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=3))
fig, axes = plt.subplots(1, 3)
metrics = ["cumulative_reward", "cumulative_cost"]
y_labels = {
    metrics[0]: r"$\hat{J}(\tilde{\pi})$",
    metrics[1]: r"$\hat{C}(\tilde{\pi})$",
}
marker_styles = ["o", "x", "^", "s", "*"]
for i, ax in enumerate(axes.flatten()[:2]):
    so.Plot(rccar_plot, x="lambda", y=metrics[i], color="lambda").add(
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
            order=["ptsd", "Simulation", "dr"],
        ),
        x=so.Nominal(
            values=["ptsd", "Simulation", "dr"], order=["ptsd", "Simulation", "dr"]
        ),
    ).label(y=lambda name: y_labels[name], x="").theme(axes_style("ticks")).on(
        ax
    ).plot()

so.Plot(go1_constraint, x="lambda", y=metrics[1], color="lambda").add(
    so.Bar(),
    so.Agg("mean"),
    legend=False,
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
        order=["ptsd", "Simulation", "dr"],
    ),
    x=so.Nominal(
        values=["ptsd", "Simulation", "dr"], order=["ptsd", "Simulation", "dr"]
    ),
).label(y=lambda name: y_labels[name], x="").theme(axes_style("ticks")).on(
    axes[2]
).plot()
car_budget = 15
go1_budget = 20

for i, ax in enumerate(axes):
    ax.grid(True, linewidth=0.5, c="gainsboro", axis="y", zorder=0)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    if i == 1:
        ax.axhline(y=car_budget, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
        yticks = ax.get_yticks()
        y_new = car_budget
        ax.set_yticks(list(yticks) + [y_new])
        ytick_labels = [f"{tick:.0f}" for tick in yticks] + ["Budget"]
        ax.set_yticklabels(ytick_labels)
    if i == 2:
        ax.axhline(y=go1_budget, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
        yticks = ax.get_yticks()
        y_new = go1_budget
        ax.set_yticks(list(yticks) + [y_new])
        ytick_labels = [f"{tick:.0f}" for tick in yticks] + ["Budget"]
        ax.set_yticklabels(ytick_labels)
        ax.set_title("Unitree Go1")
    ax.set_axisbelow(True)
    hatches = ["", "", "\\\\", "*", "o"]
    for i, bar in enumerate(ax.patches):
        hatch_pattern = hatches[i]
        bar.set_hatch(hatch_pattern)

bbox0 = axes[0].get_position()
bbox1 = axes[1].get_position()

x_center = (bbox0.x0 + bbox1.x1) / 2
y_top = max(bbox0.y1, bbox1.y1)
fig.text(
    x_center,
    y_top + 0.00,
    "RaceCar",
    ha="center",
    va="bottom",
)

legend = fig.legends.pop(0)
fig.legends = []
text = {
    "ptsd": "{{\\textsf{{\\textbf{{SPiDR}}}}}} (real)",
    "dr": "{{\sf Domain Randomization}} (real)",
    "Simulation": "{{\sf Domain Randomization}} (simulation)",
}
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
axes[1].annotate(
    f"{car_ratio:.1f}× higher\nthan \\textsf{{SPiDR}}",
    xy=(2, car_dr_cost),
    xytext=(0.50, car_dr_cost - 4),  # adjust as needed
    ha="center",
    arrowprops=dict(
        arrowstyle="-|>",
        color="black",
        linewidth=0.75,
        connectionstyle="arc3,rad=-0.2",
    ),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee", edgecolor="red"),
)

axes[2].annotate(
    f"{go1_ratio:.1f}× higher\nthan \\textsf{{SPiDR}}",
    xy=(2, go1_dr_cost),
    xytext=(0.54, go1_dr_cost - 10),  # adjust as needed
    ha="center",
    arrowprops=dict(
        arrowstyle="-|>",
        color="black",
        linewidth=0.75,
        connectionstyle="arc3,rad=-0.2",
    ),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee", edgecolor="red"),
)
fig.savefig("real-world-safety.pdf")
