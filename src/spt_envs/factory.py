"""Factory for creating project-local Safety-Gymnasium environments."""

from spt_envs.configs import DEFAULT_API, VALID_APIS, VALID_SPLITS, get_variant_spec
from spt_envs.logging import TrajectoryRecorderWrapper
from spt_envs.splits import validate_layout_seed
from spt_envs.wrappers import (
    FixedLayoutSeedWrapper,
    RewardPenaltyWrapper,
    SafetyToGymWrapper,
    StandardizeSafetyInfoWrapper,
)


def make_env(
    variant,
    split,
    layout_seed,
    api=DEFAULT_API,
    render_mode=None,
    record_trajectory=False,
    penalty_coeff=None,
):
    """Instantiate a local PointGoal environment with a stable project contract."""
    import safety_gymnasium

    from spt_envs.registry import get_env_id_for_variant, register_envs

    get_variant_spec(variant)
    if split not in VALID_SPLITS:
        raise ValueError(
            "Unknown split {!r}. Expected one of {}.".format(
                split,
                ", ".join(VALID_SPLITS),
            )
        )
    if api not in VALID_APIS:
        raise ValueError(
            "Unknown api {!r}. Expected one of {}.".format(
                api,
                ", ".join(VALID_APIS),
            )
        )

    layout_seed = validate_layout_seed(variant, split, layout_seed)
    register_envs()

    env_id = get_env_id_for_variant(variant)
    env = safety_gymnasium.make(env_id, render_mode=render_mode)
    env = FixedLayoutSeedWrapper(env, layout_seed=layout_seed)
    env = StandardizeSafetyInfoWrapper(
        env,
        variant=variant,
        split=split,
        layout_seed=layout_seed,
    )

    if penalty_coeff is not None:
        env = RewardPenaltyWrapper(env, penalty_coeff=penalty_coeff)

    if record_trajectory:
        env = TrajectoryRecorderWrapper(
            env,
            capture_frames=render_mode in ("rgb_array", "rgb_array_list"),
        )

    if api == "safe":
        return env

    return SafetyToGymWrapper(env)
