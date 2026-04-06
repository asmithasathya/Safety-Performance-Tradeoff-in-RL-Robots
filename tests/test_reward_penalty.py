"""Reward-penalty baseline contract tests."""

import numpy as np
import pytest

pytest.importorskip("safety_gymnasium")

from spt_envs.factory import make_env
from spt_envs.splits import get_layout_seeds
from spt_envs.wrappers import RewardPenaltyWrapper


PENALTY_COEFF = 1.5


@pytest.fixture
def env():
    layout_seed = get_layout_seeds("easy", "train")[0]
    e = make_env(
        variant="easy",
        split="train",
        layout_seed=layout_seed,
        api="safe",
        penalty_coeff=PENALTY_COEFF,
    )
    yield e
    e.close()


def test_shaped_reward_equals_reward_minus_penalty(env):
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    obs, shaped_reward, cost, terminated, truncated, info = env.step(action)
    expected = info["reward_unpenalized"] - PENALTY_COEFF * float(cost)
    assert shaped_reward == pytest.approx(expected)


def test_reward_unpenalized_in_info(env):
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _, _, _, _, _, info = env.step(action)
    assert "reward_unpenalized" in info
    assert "penalty_coeff" in info
    assert info["penalty_coeff"] == pytest.approx(PENALTY_COEFF)


def test_episode_penalized_return_at_episode_end(env):
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    cumulative_penalized = 0.0
    done = False
    while not done:
        _, shaped_reward, _, terminated, truncated, info = env.step(action)
        cumulative_penalized += shaped_reward
        done = bool(terminated or truncated)

    assert "episode_penalized_return" in info
    assert info["episode_penalized_return"] == pytest.approx(cumulative_penalized)
    # Unpenalized return (tracked by StandardizeSafetyInfoWrapper) must also be present
    assert "episode_return" in info


def test_penalty_coeff_zero_preserves_reward(env):
    layout_seed = get_layout_seeds("easy", "train")[0]
    env_plain = make_env("easy", "train", layout_seed, api="safe")
    env_zero = make_env("easy", "train", layout_seed, api="safe", penalty_coeff=0.0)

    env_plain.reset(seed=layout_seed)
    env_zero.reset(seed=layout_seed)

    action = np.zeros(env_plain.action_space.shape, dtype=env_plain.action_space.dtype)
    _, r_plain, _, _, _, _ = env_plain.step(action)
    _, r_zero, _, _, _, _ = env_zero.step(action)

    assert r_plain == pytest.approx(r_zero)
    env_plain.close()
    env_zero.close()


def test_negative_penalty_coeff_raises():
    with pytest.raises(ValueError, match="penalty_coeff"):
        RewardPenaltyWrapper(None, penalty_coeff=-0.1)


def test_gym_api_cost_in_info_with_penalty():
    layout_seed = get_layout_seeds("easy", "train")[0]
    env = make_env(
        "easy", "train", layout_seed, api="gym", penalty_coeff=PENALTY_COEFF
    )
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _, shaped_reward, _, _, info = env.step(action)
    # cost must still be available in info after SafetyToGymWrapper
    assert "cost" in info
    assert "reward_unpenalized" in info
    expected = info["reward_unpenalized"] - PENALTY_COEFF * info["cost"]
    assert shaped_reward == pytest.approx(expected)
    env.close()
