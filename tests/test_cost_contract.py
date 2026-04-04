"""Cost and episode summary contract tests."""

import numpy as np
import pytest

pytest.importorskip("safety_gymnasium")

from spt_envs.factory import make_env
from spt_envs.splits import get_layout_seeds


def test_cost_keys_and_episode_summaries_are_consistent():
    layout_seed = get_layout_seeds("medium", "train")[0]
    env = make_env(variant="medium", split="train", layout_seed=layout_seed, api="safe")
    env.reset()

    cumulative_reward = 0.0
    cumulative_cost = 0.0
    step_count = 0
    done = False

    while not done:
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        _, reward, cost, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        cumulative_cost += cost
        step_count += 1

        assert "cost" in info
        assert "goal_achieved" in info
        assert "variant" in info
        assert "split" in info
        assert "layout_seed" in info

        done = bool(terminated or truncated)

    assert info["episode_length"] == step_count
    assert info["episode_cost"] == pytest.approx(cumulative_cost)
    assert info["episode_return"] == pytest.approx(cumulative_reward)
    assert info["goals_achieved"] >= 0

    env.close()
