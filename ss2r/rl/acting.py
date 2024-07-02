from ss2r.rl.epoch_summary import EpochSummary
from ss2r.rl.types import Agent, Simulator


def evaluate(
    agent: Agent,
    simulator: Simulator,
    num_steps: int,
    render_episodes: int = 0,
) -> tuple[EpochSummary, int]:
    summary = EpochSummary()
    _, trajectories = simulator.rollout(agent.policy, num_steps)
    summary.extend(trajectories)
    return summary
