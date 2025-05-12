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

budgets = {
    "RaceCar": 5,
    "SafeCartpoleSwingup": 100,
    "SafeHumanoidWalk": 100,
    "SafeQuadrupedRun": 100,
    "SafeWalkerWalk": 100,
    "PointGoal2": 25,
}

# %%
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=3, height_to_width_ratio=1.6))
plt.rcParams.update({"text.latex.preamble": r"\usepackage{xfrac}\usepackage{times}"})
fig = plt.figure()
metric = "eval/episode_cost"
y_label = r"$\hat{C}(\tilde{\pi})$"
marker_styles = ["o", "x", "^", "s", "*"]
so.Plot(
    data,
    x="step",
    y=metric,
    marker="category",
    color="category",
).facet(col="environment", wrap=3, order=list(budgets.keys())).share(
    y=False, x=False
).add(
    so.Line(linewidth=1.0, pointsize=2.5, edgewidth=0.1),
    so.Agg("median"),
).add(so.Band(alpha=0.15), so.Est("median", errorbar=("ci", 68)), legend=False).scale(
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
    x="",
    y=y_label,
    title=lambda x: x.strip("Safe"),
).theme(axes_style("ticks")).on(fig).plot()
axes = fig.get_axes()
limits = {
    "RaceCar": 30,
    "CartpoleSwingup": 250,
    "HumanoidWalk": 450,
    "QuadrupedRun": 350,
    "WalkerWalk": 250,
    "PointGoal2": 100,
}
tmp_budgets = {
    "RaceCar": 5,
    "CartpoleSwingup": 100,
    "HumanoidWalk": 100,
    "QuadrupedRun": 100,
    "WalkerWalk": 100,
    "PointGoal2": 25,
}
for i, ax in enumerate(fig.get_axes()):
    ax.set_ylim(-1, limits[ax.get_title()])
    budget = tmp_budgets[ax.get_title()]
    ax.axhline(y=budget, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
    ylim = ax.get_ylim()
    yticks = ax.get_yticks()
    y_new = budget
    ax.set_yticks(list(yticks) + [y_new])
    ytick_labels = [f"${tick:.0f}$" for tick in yticks] + ["Budget"]
    ax.set_yticklabels(ytick_labels)
    ax.set_ylim(-1, limits[ax.get_title()])
    ax.grid(True, linewidth=0.5, c="gainsboro", axis="both", zorder=0)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.yaxis.labelpad = -0.5

text = {
    "ramu": "\sf RAMU",
    "ptsd": "\\textsf{{\\textbf{{SPiDR}}}}",
    "dr": "\sf Domain Randomization",
    "nominal": "\sf Nominal",
}
legend = fig.legends.pop(0)
fig.legends = []
fig.legend(
    legend.legend_handles,
    [text[t.get_text()] for t in legend.texts],
    loc="center",
    bbox_to_anchor=(0.5, 1.015),
    ncol=4,
    frameon=False,
)
fig.supxlabel("Simulation Steps", fontsize=9, y=0.025)
fig.tight_layout(h_pad=0.0, pad=0.3)
fig.savefig("simulated-costs.pdf")
