from ss2r.rl.epoch_summary import EpochSummary
from ss2r.rl.types import Agent, Simulator


def evaluate(
    agent: Agent,
    simulator: Simulator,
    num_steps: int,
    seed: int,
    render_episodes: int = 0,
) -> EpochSummary:
    summary = EpochSummary()
    _, trajectories = simulator.rollout(agent.policy, num_steps, seed)
    summary.extend(trajectories)
    return summary
