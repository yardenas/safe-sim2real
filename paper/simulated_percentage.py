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
        keys=["eval/episode_safe", "_step", "eval/episode_reward"],
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
        "$regex": "apr01-nominal-aga$|apr01-dr-aga$|apr01-ptsd-aga|apr01-ramu-aga|apr11-g2g-ramu|apr11-g2g-ptsd|apr11-g2g-dr|apr11-g2g-nominal"
    }
}
data = pd.concat(
    [
        load_evaluation(*summary_files_data)
        for summary_files_data in walk_wandb_runs("yardas/ss2r", filters)
    ]
)
# %%

last_steps = data.groupby(["category", "environment", "seed"]).tail(1).reset_index()

budgets = {
    "RaceCar": 5,
    "SafeCartpoleSwingup": 100,
    "SafeHumanoidWalk": 100,
    "SafeQuadrupedRun": 100,
    "SafeWalkerWalk": 100,
    "PointGoal2": 25,
}

# %%

env_name_to_display = {
    "RaceCar": "RaceCar",
    "SafeCartpoleSwingup": "Cartpole\nSwingup",
    "SafeHumanoidWalk": "Humanoid\nWalk",
    "SafeQuadrupedRun": "Quadruped\nRun",
    "SafeWalkerWalk": "Walker\nWalk",
    "PointGoal2": "PointGoal2",
}

text = {
    "ramu": "\sf RAMU",
    "ptsd": "\sf PTSD",
    "dr": "\sf Domain Randomization",
    "nominal": "\sf Nominal",
    "tightening": "\sf Tightening",
}


theme = bundles.neurips2024()
metrics = ["eval/episode_reward", "eval/episode_safe"]
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(figsizes.neurips2024(nrows=2, ncols=6, height_to_width_ratio=1.1))
fig = plt.figure()
so.Plot(
    last_steps,
    x="category",
    color="category",
).facet(col="environment", order=list(budgets.keys())).pair(y=metrics).share(
    y=False, x=True
).add(
    so.Range(color="k", linewidth=0.75),
    so.Est("mean", errorbar="se"),
    legend=False,
).add(so.Bar(), so.Agg(), legend=True).scale(
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
    x=so.Nominal(order=["ptsd", "ramu", "dr", "nominal"]),
).label(x="", y=r"$\hat{J}(\pi)$", title=lambda x: env_name_to_display[x]).theme(
    axes_style("ticks")
).on(fig).plot()
axes = fig.get_axes()
for ax in axes:
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    ax.tick_params(axis="x", length=0, labelbottom=False)

legend = fig.legends.pop(0)

fig.legend(
    legend.legend_handles,
    [text[t.get_text()] for t in legend.texts],
    loc="center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=6,
    frameon=False,
    handletextpad=0.25,
    handlelength=1.0,
)

fig.savefig("simulated-percentage.pdf")
