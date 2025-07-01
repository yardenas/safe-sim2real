import matplotlib.pyplot as plt


class RewardPlotter:
    def __init__(self):
        self.rewards = []
        self.steps = []
        # Set up the interactive plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [], label="Reward")
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Reward Over Time")
        self.ax.legend()
        self.fig.show()

    def update(self, data: dict):
        if "reward" not in data:
            raise ValueError("Data must contain a 'reward' key.")
        if not isinstance(data["reward"], float):
            raise ValueError("'reward' must be a float.")
        self.steps.append(data["ongoing_steps"])
        self.rewards.append(data["reward"])
        self.line.set_data(self.steps, self.rewards)
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
