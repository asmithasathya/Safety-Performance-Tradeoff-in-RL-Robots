"""Lagrangian constrained RL baseline contract tests."""

import numpy as np
import pytest

pytest.importorskip("safety_gymnasium")

from spt_envs.factory import make_env
from spt_envs.splits import get_layout_seeds
from spt_envs.wrappers import LagrangianWrapper


BUDGET = 10.0
LR_LAMBDA = 0.05


@pytest.fixture
def env():
    layout_seed = get_layout_seeds("easy", "train")[0]
    e = make_env(
        variant="easy",
        split="train",
        layout_seed=layout_seed,
        api="safe",
        lagrangian_budget=BUDGET,
        lagrangian_lr=LR_LAMBDA,
    )
    yield e
    e.close()


def test_shaped_reward_equals_reward_minus_lambda_times_cost(env):
    env.reset()
    lam = env.lambda_
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _, shaped_reward, cost, _, _, info = env.step(action)
    expected = info["reward_unpenalized"] - lam * float(cost)
    assert shaped_reward == pytest.approx(expected)


def test_info_keys_present_each_step(env):
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _, _, _, _, _, info = env.step(action)
    assert "reward_unpenalized" in info
    assert "lagrangian_lambda" in info
    assert "budget" in info
    assert info["budget"] == pytest.approx(BUDGET)


def test_lambda_increases_when_over_budget():
    layout_seed = get_layout_seeds("hard", "train")[0]
    # Hard variant (12 hazards) should rack up costs quickly.
    e = make_env(
        variant="hard",
        split="train",
        layout_seed=layout_seed,
        api="safe",
        lagrangian_budget=0.0,   # always violated
        lagrangian_lr=1.0,
    )
    e.reset()
    action = np.zeros(e.action_space.shape, dtype=e.action_space.dtype)
    lambda_before = e.lambda_
    done = False
    while not done:
        _, _, _, terminated, truncated, info = e.step(action)
        done = bool(terminated or truncated)
    e.close()

    episode_cost = info["episode_cost"]
    if episode_cost > 0.0:
        assert e.lambda_ > lambda_before, "λ should increase when episode_cost > budget"


def test_lambda_stays_nonnegative_when_under_budget():
    layout_seed = get_layout_seeds("easy", "train")[0]
    e = make_env(
        variant="easy",
        split="train",
        layout_seed=layout_seed,
        api="safe",
        lagrangian_budget=1000.0,  # never violated
        lagrangian_lr=LR_LAMBDA,
        lagrangian_init_lambda=0.0,
    )
    e.reset()
    action = np.zeros(e.action_space.shape, dtype=e.action_space.dtype)
    done = False
    while not done:
        _, _, _, terminated, truncated, _ = e.step(action)
        done = bool(terminated or truncated)
    e.close()
    assert e.lambda_ >= 0.0


def test_dual_update_formula(env):
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    lambda_before = env.lambda_
    done = False
    while not done:
        _, _, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

    episode_cost = info["episode_cost"]
    expected_lambda = max(0.0, lambda_before + LR_LAMBDA * (episode_cost - BUDGET))
    assert env.lambda_ == pytest.approx(expected_lambda)
    # info["lagrangian_lambda"] reflects the post-update value
    assert info["lagrangian_lambda"] == pytest.approx(expected_lambda)


def test_episode_penalized_return_consistent(env):
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    cumulative = 0.0
    done = False
    while not done:
        _, shaped_reward, _, terminated, truncated, info = env.step(action)
        cumulative += shaped_reward
        done = bool(terminated or truncated)

    assert "episode_penalized_return" in info
    assert info["episode_penalized_return"] == pytest.approx(cumulative)


def test_mutual_exclusion_with_penalty_coeff():
    layout_seed = get_layout_seeds("easy", "train")[0]
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_env(
            variant="easy",
            split="train",
            layout_seed=layout_seed,
            penalty_coeff=1.0,
            lagrangian_budget=10.0,
            lagrangian_lr=0.01,
        )


def test_lagrangian_lr_required_when_budget_set():
    layout_seed = get_layout_seeds("easy", "train")[0]
    with pytest.raises(ValueError, match="lagrangian_lr"):
        make_env(
            variant="easy",
            split="train",
            layout_seed=layout_seed,
            lagrangian_budget=10.0,
        )


def test_invalid_constructor_args():
    with pytest.raises(ValueError, match="budget"):
        LagrangianWrapper(None, budget=-1, lr_lambda=0.1)
    with pytest.raises(ValueError, match="lr_lambda"):
        LagrangianWrapper(None, budget=10, lr_lambda=0.0)
    with pytest.raises(ValueError, match="init_lambda"):
        LagrangianWrapper(None, budget=10, lr_lambda=0.1, init_lambda=-0.5)


def test_gym_api_with_lagrangian():
    layout_seed = get_layout_seeds("easy", "train")[0]
    e = make_env(
        variant="easy",
        split="train",
        layout_seed=layout_seed,
        api="gym",
        lagrangian_budget=BUDGET,
        lagrangian_lr=LR_LAMBDA,
    )
    e.reset()
    action = np.zeros(e.action_space.shape, dtype=e.action_space.dtype)
    _, shaped_reward, _, _, info = e.step(action)
    assert "cost" in info
    assert "reward_unpenalized" in info
    assert "lagrangian_lambda" in info
    e.close()
