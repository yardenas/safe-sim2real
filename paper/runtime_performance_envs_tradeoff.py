import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Simulated data
np.random.seed(0)  # For reproducibility

# Number of environments
envs = np.array([2, 4, 8, 16, 32])

# Simulated runtime (increasing linearly until a memory threshold, then increases faster)
runtime = np.array([1, 2, 4, 8, 16]) + np.random.normal(0, 0.1, len(envs))

# Simulated performance (diminishing returns as environments increase)
performance = np.array([60, 75, 85, 90, 92]) + np.random.normal(0, 1, len(envs))

# Create a DataFrame for seaborn
data = pd.DataFrame(
    {"Environments": envs, "Runtime": runtime, "Performance": performance}
)

# Plot using seaborn's object interface
p = sns.scatterplot(
    data=data,
    x="Environments",
    y="Runtime",
    hue="Performance",
    size="Performance",
    palette="coolwarm",
)
p.set(yscale="log", xscale="log")
p.set(title="Performance vs. Number of Environments & Runtime")

plt.colorbar(p.collections[0], label="Performance")  # Add colorbar for reference
plt.show()
