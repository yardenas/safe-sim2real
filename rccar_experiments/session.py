from rccar_experiments.logger import Logger
from rccar_experiments.plot import RewardPlotter


class Session:
    def __init__(self, directory=".", filename_prefix="log", filename=None):
        self.logger = Logger(
            directory=directory, filename_prefix=filename_prefix, filename=filename
        )
        self.ongoing_steps = 0
        self.plotter = RewardPlotter()
        # Load existing data if any
        existing_data = self.logger.load_existing_data()
        for row in existing_data:
            try:
                reward = float(row["reward"])
                self.plotter.steps.append(int(row["ongoing_steps"]))
                self.plotter.rewards.append(reward)
            except (ValueError, KeyError):
                continue  # Skip malformed rows
        if len(existing_data) > 0:
            self.ongoing_steps = int(max(data["steps"] for data in existing_data))
        else:
            self.ongoing_steps = 0
        # Update plot with loaded data
        self.plotter.line.set_data(self.plotter.steps, self.plotter.rewards)
        self.plotter.ax.relim()
        self.plotter.ax.autoscale_view()
        self.plotter.fig.canvas.draw()
        self.plotter.fig.canvas.flush_events()

    def update(self, data: dict):
        if "reward" not in data or not isinstance(data["reward"], float):
            raise ValueError("Data must include a float 'reward'.")
        self.ongoing_steps += data["steps"]
        data["ongoing_steps"] = self.ongoing_steps
        self.logger.append_row(data)
        self.plotter.update(data)

    @property
    def steps(self):
        return self.plotter.steps
