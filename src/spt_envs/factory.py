"""Factory for creating project-local Safety-Gymnasium environments."""

from spt_envs.configs import DEFAULT_API, VALID_APIS, VALID_SPLITS, get_variant_spec
from spt_envs.logging import TrajectoryRecorderWrapper
from spt_envs.splits import get_layout_seeds, validate_layout_seed
from spt_envs.wrappers import (
    FixedLayoutSeedWrapper,
    LagrangianWrapper,
    RewardPenaltyWrapper,
    RuleBasedShieldWrapper,
    SafetyToGymWrapper,
    StandardizeSafetyInfoWrapper,
    TrainLayoutSeedWrapper,
)


def _validate_api(split, api):
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


def _apply_baseline_wrappers(
    env,
    penalty_coeff=None,
    lagrangian_budget=None,
    lagrangian_lr=None,
    lagrangian_init_lambda=0.0,
    shield_warning_radius=None,
):
    active = sum([
        penalty_coeff is not None,
        lagrangian_budget is not None,
        shield_warning_radius is not None,
    ])
    if active > 1:
        raise ValueError(
            "penalty_coeff, lagrangian_budget, and shield_warning_radius are "
            "mutually exclusive; choose one baseline at a time."
        )

    if penalty_coeff is not None:
        env = RewardPenaltyWrapper(env, penalty_coeff=penalty_coeff)

    if lagrangian_budget is not None:
        if lagrangian_lr is None:
            raise ValueError(
                "lagrangian_lr is required when lagrangian_budget is set."
            )
        env = LagrangianWrapper(
            env,
            budget=lagrangian_budget,
            lr_lambda=lagrangian_lr,
            init_lambda=lagrangian_init_lambda,
        )

    if shield_warning_radius is not None:
        env = RuleBasedShieldWrapper(env, warning_radius=shield_warning_radius)

    return env


def make_env(
    variant,
    split,
    layout_seed,
    api=DEFAULT_API,
    render_mode=None,
    record_trajectory=False,
    penalty_coeff=None,
    lagrangian_budget=None,
    lagrangian_lr=None,
    lagrangian_init_lambda=0.0,
    shield_warning_radius=None,
):
    """Instantiate a local PointGoal environment with a stable project contract."""
    import safety_gymnasium

    from spt_envs.registry import get_env_id_for_variant, register_envs

    get_variant_spec(variant)
    _validate_api(split, api)

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
    env = _apply_baseline_wrappers(
        env,
        penalty_coeff=penalty_coeff,
        lagrangian_budget=lagrangian_budget,
        lagrangian_lr=lagrangian_lr,
        lagrangian_init_lambda=lagrangian_init_lambda,
        shield_warning_radius=shield_warning_radius,
    )

    if record_trajectory:
        env = TrajectoryRecorderWrapper(
            env,
            capture_frames=render_mode in ("rgb_array", "rgb_array_list"),
        )

    if api == "safe":
        return env

    return SafetyToGymWrapper(env)


def make_train_env(
    variant,
    seed,
    api=DEFAULT_API,
    render_mode=None,
    record_trajectory=False,
    penalty_coeff=None,
    lagrangian_budget=None,
    lagrangian_lr=None,
    lagrangian_init_lambda=0.0,
    shield_warning_radius=None,
):
    """Instantiate a training env that samples a train layout seed on each reset."""
    import safety_gymnasium

    from spt_envs.registry import get_env_id_for_variant, register_envs

    get_variant_spec(variant)
    _validate_api("train", api)
    register_envs()

    env_id = get_env_id_for_variant(variant)
    env = safety_gymnasium.make(env_id, render_mode=render_mode)
    env = TrainLayoutSeedWrapper(
        env,
        layout_seeds=get_layout_seeds(variant, "train"),
        rng_seed=seed,
    )
    env = StandardizeSafetyInfoWrapper(
        env,
        variant=variant,
        split="train",
        layout_seed=None,
    )
    env = _apply_baseline_wrappers(
        env,
        penalty_coeff=penalty_coeff,
        lagrangian_budget=lagrangian_budget,
        lagrangian_lr=lagrangian_lr,
        lagrangian_init_lambda=lagrangian_init_lambda,
        shield_warning_radius=shield_warning_radius,
    )

    if record_trajectory:
        env = TrajectoryRecorderWrapper(
            env,
            capture_frames=render_mode in ("rgb_array", "rgb_array_list"),
        )

    if api == "safe":
        return env

    return SafetyToGymWrapper(env)
