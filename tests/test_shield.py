"""Rule-based safety shield contract tests."""

import numpy as np
import pytest

pytest.importorskip("safety_gymnasium")

from spt_envs.factory import make_env
from spt_envs.splits import get_layout_seeds
from spt_envs.wrappers import RuleBasedShieldWrapper


WARNING_RADIUS = 0.4


@pytest.fixture
def env():
    layout_seed = get_layout_seeds("easy", "train")[0]
    e = make_env(
        variant="easy",
        split="train",
        layout_seed=layout_seed,
        api="safe",
        shield_warning_radius=WARNING_RADIUS,
    )
    yield e
    e.close()


def test_info_keys_present_each_step(env):
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _, _, _, _, _, info = env.step(action)
    assert "shield_intervened" in info
    assert "episode_shield_interventions" in info
    assert isinstance(info["shield_intervened"], bool)
    assert isinstance(info["episode_shield_interventions"], int)


def test_intervention_rate_at_episode_end(env):
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    done = False
    total_interventions = 0
    total_steps = 0
    while not done:
        _, _, _, terminated, truncated, info = env.step(action)
        total_interventions += int(info["shield_intervened"])
        total_steps += 1
        done = bool(terminated or truncated)

    assert "shield_intervention_rate" in info
    expected_rate = total_interventions / max(1, total_steps)
    assert info["shield_intervention_rate"] == pytest.approx(expected_rate)


def test_intervention_count_matches_intervened_flag(env):
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    counted = 0
    for _ in range(10):
        _, _, _, terminated, truncated, info = env.step(action)
        if info["shield_intervened"]:
            counted += 1
        assert info["episode_shield_interventions"] == counted
        if terminated or truncated:
            break


def test_reward_unmodified_by_shield(env):
    """The shield changes actions but never shapes the reward."""
    layout_seed = get_layout_seeds("easy", "train")[0]
    env_plain = make_env("easy", "train", layout_seed, api="safe")
    env_shield = make_env(
        "easy", "train", layout_seed, api="safe", shield_warning_radius=WARNING_RADIUS
    )
    env_plain.reset(seed=layout_seed)
    env_shield.reset(seed=layout_seed)

    # When the shield does NOT intervene, raw reward must match.
    action = np.zeros(env_plain.action_space.shape, dtype=env_plain.action_space.dtype)
    _, r_plain, _, _, _, _ = env_plain.step(action)
    _, r_shield, _, _, _, info = env_shield.step(action)

    if not info["shield_intervened"]:
        assert r_shield == pytest.approx(r_plain)

    env_plain.close()
    env_shield.close()


def test_invalid_warning_radius():
    with pytest.raises(ValueError, match="warning_radius"):
        RuleBasedShieldWrapper(None, warning_radius=0.0)
    with pytest.raises(ValueError, match="warning_radius"):
        RuleBasedShieldWrapper(None, warning_radius=-0.1)


def test_mutual_exclusion_with_penalty():
    layout_seed = get_layout_seeds("easy", "train")[0]
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_env(
            variant="easy",
            split="train",
            layout_seed=layout_seed,
            penalty_coeff=1.0,
            shield_warning_radius=WARNING_RADIUS,
        )


def test_mutual_exclusion_with_lagrangian():
    layout_seed = get_layout_seeds("easy", "train")[0]
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_env(
            variant="easy",
            split="train",
            layout_seed=layout_seed,
            lagrangian_budget=10.0,
            lagrangian_lr=0.05,
            shield_warning_radius=WARNING_RADIUS,
        )


def test_gym_api_with_shield():
    layout_seed = get_layout_seeds("easy", "train")[0]
    e = make_env(
        variant="easy",
        split="train",
        layout_seed=layout_seed,
        api="gym",
        shield_warning_radius=WARNING_RADIUS,
    )
    e.reset()
    action = np.zeros(e.action_space.shape, dtype=e.action_space.dtype)
    _, _, _, _, info = e.step(action)
    assert "cost" in info
    assert "shield_intervened" in info
    assert "episode_shield_interventions" in info
    e.close()
