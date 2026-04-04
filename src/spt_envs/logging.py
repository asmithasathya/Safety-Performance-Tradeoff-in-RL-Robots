"""Trajectory logging utilities for project-local env rollouts."""

import copy
from dataclasses import dataclass, field

import gymnasium


@dataclass
class StepRecord:
    """A single environment interaction record."""

    action: object
    reward: float
    cost: float
    terminated: bool
    truncated: bool
    info: dict
    frame: object = None


@dataclass
class EpisodeTrajectory:
    """A complete recorded episode."""

    variant: str
    split: str
    layout_seed: int
    initial_observation: object
    reset_info: dict
    steps: list = field(default_factory=list)
    episode_return: float = 0.0
    episode_cost: float = 0.0
    goals_achieved: int = 0
    episode_length: int = 0


class TrajectoryRecorderWrapper(gymnasium.Wrapper):
    """Optionally capture rollout traces for later analysis or preference data."""

    def __init__(self, env, capture_frames=False):
        super().__init__(env)
        self.capture_frames = bool(capture_frames)
        self.completed_trajectories = []
        self._current_trajectory = None

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._current_trajectory = EpisodeTrajectory(
            variant=info.get("variant", ""),
            split=info.get("split", ""),
            layout_seed=info.get("layout_seed", -1),
            initial_observation=copy.deepcopy(observation),
            reset_info=dict(info),
        )
        return observation, info

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        frame = None
        if self.capture_frames:
            try:
                frame = self.env.render()
            except Exception:
                frame = None

        if self._current_trajectory is not None:
            self._current_trajectory.steps.append(
                StepRecord(
                    action=copy.deepcopy(action),
                    reward=float(reward),
                    cost=float(cost),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=dict(info),
                    frame=copy.deepcopy(frame),
                )
            )

            if terminated or truncated:
                self._current_trajectory.episode_return = float(
                    info.get("episode_return", 0.0)
                )
                self._current_trajectory.episode_cost = float(
                    info.get("episode_cost", 0.0)
                )
                self._current_trajectory.goals_achieved = int(
                    info.get("goals_achieved", 0)
                )
                self._current_trajectory.episode_length = int(
                    info.get("episode_length", len(self._current_trajectory.steps))
                )
                self.completed_trajectories.append(self._current_trajectory)
                self._current_trajectory = None

        return observation, reward, cost, terminated, truncated, info

    def pop_completed_trajectories(self):
        """Return and clear the completed trajectory buffer."""
        completed = list(self.completed_trajectories)
        self.completed_trajectories.clear()
        return completed
