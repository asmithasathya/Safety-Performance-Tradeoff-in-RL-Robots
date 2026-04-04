"""Registration and task definitions for project-local Safety-Gymnasium envs."""

from gymnasium.envs.registration import registry as gym_registry

import safety_gymnasium
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.builder import Builder
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

from spt_envs.configs import (
    DEFAULT_AGENT_NAME,
    DEFAULT_EXTENTS,
    DEFAULT_HAZARD_KEEPOUT,
    DEFAULT_MAX_EPISODE_STEPS,
    ENV_ID_TO_VARIANT,
    VARIANT_SPECS,
)


class HazardOnlyPointGoalBase(GoalLevel0):
    """Goal task with hazards as the only active safety object."""

    HAZARD_COUNT = 0

    def __init__(self, config):
        super().__init__(config=config)
        self.placements_conf.extents = list(DEFAULT_EXTENTS)
        self._add_geoms(Hazards(num=self.HAZARD_COUNT, keepout=DEFAULT_HAZARD_KEEPOUT))
        self.num_steps = DEFAULT_MAX_EPISODE_STEPS
        self.cost_conf.constrain_indicator = True
        self.mechanism_conf.continue_goal = True
        self.mechanism_conf.randomize_layout = True
        self.mechanism_conf.terminate_resample_failure = True


class PointGoalEasy(HazardOnlyPointGoalBase):
    """Easy hazard-only PointGoal task."""

    HAZARD_COUNT = VARIANT_SPECS["easy"]["hazards"]


class PointGoalMedium(HazardOnlyPointGoalBase):
    """Medium hazard-only PointGoal task."""

    HAZARD_COUNT = VARIANT_SPECS["medium"]["hazards"]


class PointGoalHard(HazardOnlyPointGoalBase):
    """Hard hazard-only PointGoal task."""

    HAZARD_COUNT = VARIANT_SPECS["hard"]["hazards"]


TASK_ID_TO_CLASS = {
    VARIANT_SPECS["easy"]["env_id"]: PointGoalEasy,
    VARIANT_SPECS["medium"]["env_id"]: PointGoalMedium,
    VARIANT_SPECS["hard"]["env_id"]: PointGoalHard,
}

_REGISTERED = False


class SPTBuilder(Builder):
    """Builder that resolves project-local task IDs to local task classes."""

    def _get_task(self):
        try:
            task_class = TASK_ID_TO_CLASS[self.task_id]
        except KeyError as exc:
            raise ValueError(
                "Task ID {!r} is not registered. Known IDs: {}.".format(
                    self.task_id,
                    ", ".join(sorted(TASK_ID_TO_CLASS)),
                ),
            ) from exc
        task = task_class(config=self.config or {})
        task.build_observation_space()
        return task


def register_envs():
    """Register project-local safe env IDs with Safety-Gymnasium once."""
    global _REGISTERED
    if _REGISTERED:
        return

    for variant, spec in VARIANT_SPECS.items():
        env_id = spec["env_id"]
        if env_id in gym_registry:
            continue

        safety_gymnasium.register(
            id=env_id,
            entry_point="spt_envs.registry:SPTBuilder",
            kwargs={
                "task_id": env_id,
                "config": {
                    "agent_name": DEFAULT_AGENT_NAME,
                    "observation_flatten": True,
                },
            },
            max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
        )

    _REGISTERED = True


def get_env_id_for_variant(variant):
    """Translate a variant label to the registered local env ID."""
    try:
        return VARIANT_SPECS[variant]["env_id"]
    except KeyError as exc:
        raise ValueError(
            "Unknown variant {!r}. Expected one of {}.".format(
                variant,
                ", ".join(sorted(VARIANT_SPECS)),
            ),
        ) from exc


def get_variant_for_env_id(env_id):
    """Translate a local env ID back to its short variant label."""
    try:
        return ENV_ID_TO_VARIANT[env_id]
    except KeyError as exc:
        raise ValueError(
            "Unknown env_id {!r}. Expected one of {}.".format(
                env_id,
                ", ".join(sorted(ENV_ID_TO_VARIANT)),
            ),
        ) from exc
