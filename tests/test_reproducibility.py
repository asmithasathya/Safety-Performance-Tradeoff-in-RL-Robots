"""Deterministic rollout tests for fixed layout seeds."""

import numpy as np
import pytest

pytest.importorskip("safety_gymnasium")

from spt_envs.factory import make_env
from spt_envs.splits import get_layout_seeds


def test_fixed_layout_seed_reproduces_rollouts():
    layout_seed = get_layout_seeds("easy", "train")[0]
    env_a = make_env(variant="easy", split="train", layout_seed=layout_seed, api="safe")
    env_b = make_env(variant="easy", split="train", layout_seed=layout_seed, api="safe")

    obs_a, info_a = env_a.reset()
    obs_b, info_b = env_b.reset()
    assert np.allclose(obs_a, obs_b)
    assert info_a == info_b

    low = env_a.action_space.low
    high = env_a.action_space.high
    rng = np.random.RandomState(7)

    for _ in range(5):
        action = rng.uniform(low=low, high=high).astype(env_a.action_space.dtype)
        step_a = env_a.step(action)
        step_b = env_b.step(action)

        assert np.allclose(step_a[0], step_b[0])
        assert step_a[1] == pytest.approx(step_b[1])
        assert step_a[2] == pytest.approx(step_b[2])
        assert step_a[3] == step_b[3]
        assert step_a[4] == step_b[4]
        assert step_a[5] == step_b[5]

    env_a.close()
    env_b.close()
