import numpy as np
from tqdm import tqdm

from ss2r.rl.episodic_async_env import EpisodicAsync
from ss2r.rl.epoch_summary import EpochSummary
from ss2r.rl.trajectory import Trajectory, TrajectoryData, Transition
from ss2r.rl.types import Agent


def _summarize_episodes(
    trajectory: TrajectoryData,
) -> tuple[float, float]:
    reward = float(trajectory.reward.sum(1).mean())
    cost = float(trajectory.cost.sum(1).mean())
    return reward, cost


def interact(
    agent: Agent,
    environment: EpisodicAsync,
    num_episodes: int,
    train: bool,
    step: int,
    render_episodes: int = 0,
) -> tuple[list[Trajectory], int]:
    observations = environment.reset()
    episode_count = 0
    episodes: list[Trajectory] = []
    trajectory = Trajectory()
    with tqdm(
        total=num_episodes,
        unit=f"Episode (✕ {environment.num_envs} parallel)",
    ) as pbar:
        while episode_count < num_episodes:
            render = render_episodes > 0
            if render:
                trajectory.frames.append(environment.render())
            actions = agent(observations, train)
            next_observations, rewards, done, infos = environment.step(actions)
            costs = np.array([info.get("cost", 0) for info in infos])
            transition = Transition(
                observations, next_observations, actions, rewards, costs
            )
            trajectory.transitions.append(transition)
            if train:
                agent.observe(transition)
            observations = next_observations
            if done.any():
                assert (
                    done.all()
                ), "No support for environments with different ending conditions"
                np_trajectory = trajectory.as_numpy()
                step += int(np.prod(np_trajectory.reward.shape))
                reward, cost = _summarize_episodes(np_trajectory)
                pbar.set_postfix({"reward": reward, "cost": cost})
                if render:
                    render_episodes = max(render_episodes - 1, 0)
                episodes.append(trajectory)
                trajectory = Trajectory()
                pbar.update(1)
                episode_count += 1
                observations = environment.reset()
    return episodes, step


def epoch(
    agent: Agent,
    env: EpisodicAsync,
    num_episodes: int,
    train: bool,
    step: int,
    render_episodes: int = 0,
) -> tuple[EpochSummary, int]:
    summary = EpochSummary()
    samples, step = interact(
        agent,
        env,
        num_episodes,
        train,
        step,
        render_episodes,
    )
    summary.extend(samples)
    return summary, step
