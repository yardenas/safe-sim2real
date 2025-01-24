# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so
import wandb
from matplotlib.ticker import FuncFormatter
from seaborn import axes_style
from tueplots import bundles, figsizes

warnings.filterwarnings("ignore")


# %%
api = wandb.Api()


def load_evaluation(data):
    if data.index.name is None or data.index.name != "step":
        data.rename(columns={"_step": "step"}, inplace=True)
        data.set_index("step", inplace=True)
    data = data.dropna()
    return data


def walk_wandb_runs(project, filters):
    runs = api.runs(project, filters=filters)
    print(f"Fetching {len(runs)} runs")
    for run in runs:
        metrics = run.history(
            keys=["cost", "reward"],
            x_axis="_step",
            pandas=False,
        )
        policy_id = run.config["policy_id"]
        metrics = pd.DataFrame(metrics)
        metrics["category"] = policy_id
        yield metrics


filters = {
    "display_name": {
        "$regex": "environment.train_car_params.nominal.max_throttle=0.4,num_trajectories=10,out_path_name=yarden,policy_id=o0de6yo8,port_number=6,wandb.notes=dec3-car|environment.train_car_params.nominal.max_throttle=0.4,num_trajectories=10,out_path_name=yarden,policy_id=lryod0ak,port_number=6,wandb.notes=dec3-car"
    },
}

data = pd.concat(
    [
        load_evaluation(summary_files_data)
        for summary_files_data in walk_wandb_runs("yardas/ss2r", filters)
    ]
)


# %%


theme = bundles.jmlr2001()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(figsizes.jmlr2001(nrows=1, ncols=2, rel_width=0.5))
fig, axes = plt.subplots(1, 2)
metrics = ["reward", "cost"]
y_labels = {
    metrics[0]: r"$\hat{J}(\pi)$",
    metrics[1]: r"$\hat{C}(\pi)$",
}
marker_styles = ["o", "x", "^", "s", "*"]
for i, ax in enumerate(axes.flatten()):
    so.Plot(data, x="category", y=metrics[i], color="category").add(
        so.Bar(),
        so.Agg("mean"),
        legend=True,
    ).add(
        so.Range(color="k", linewidth=0.75),
        so.Est("mean", errorbar="se"),
        legend=False,
    ).add(so.Band(alpha=0.15), so.Est("mean", errorbar="se"), legend=False).scale(
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
            order=["lryod0ak", "o0de6yo8"],
        ),
        x=so.Nominal(values=["lryod0ak", "o0de6yo8"], order=["lryod0ak", "o0de6yo8"]),
    ).label(y=lambda name: y_labels[name], x="").theme(axes_style("ticks")).on(
        ax
    ).plot()

formatter = FuncFormatter(lambda x, _: {0: "", 1: ""}[x])

for i, ax in enumerate(axes):
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, linewidth=0.5, c="gainsboro", axis="y", zorder=0)
    if i >= 1:
        ax.axhline(y=8, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
        yticks = ax.get_yticks()
        y_new = 8
        ax.set_yticks(list(yticks) + [y_new])
        ytick_labels = [f"{tick:.0f}" for tick in yticks] + ["Budget"]
        ax.set_yticklabels(ytick_labels)
legend = fig.legends.pop(0)
fig.legends = []
labels = {"lryod0ak": "\sf Cost UCB", "o0de6yo8": "\sf Neutral"}
fig.legend(
    legend.legend_handles,
    [labels[t.get_text()] for t in legend.texts],
    loc="center",
    bbox_to_anchor=(0.5, 1.05),
    frameon=False,
    ncols=2,
)
fig.savefig("rccar-sim-to-real.pdf")
