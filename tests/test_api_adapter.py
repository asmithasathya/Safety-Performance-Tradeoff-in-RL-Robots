"""API compatibility tests for safe and gym env adapters."""

import numpy as np
import pytest

pytest.importorskip("safety_gymnasium")

from spt_envs.factory import make_env
from spt_envs.splits import get_layout_seeds


def test_safe_and_gym_apis_match_rollout_totals():
    layout_seed = get_layout_seeds("hard", "train")[0]
    safe_env = make_env(variant="hard", split="train", layout_seed=layout_seed, api="safe")
    gym_env = make_env(variant="hard", split="train", layout_seed=layout_seed, api="gym")

    safe_obs, safe_info = safe_env.reset()
    gym_obs, gym_info = gym_env.reset()

    assert np.allclose(safe_obs, gym_obs)
    assert safe_info == gym_info

    low = safe_env.action_space.low
    high = safe_env.action_space.high
    rng = np.random.RandomState(11)

    for _ in range(8):
        action = rng.uniform(low=low, high=high).astype(safe_env.action_space.dtype)
        safe_step = safe_env.step(action)
        gym_step = gym_env.step(action)

        assert np.allclose(safe_step[0], gym_step[0])
        assert safe_step[1] == pytest.approx(gym_step[1])
        assert safe_step[2] == pytest.approx(gym_step[4]["cost"])
        assert safe_step[3] == gym_step[2]
        assert safe_step[4] == gym_step[3]
        assert safe_step[5]["cost"] == pytest.approx(gym_step[4]["cost"])

        if safe_step[3] or safe_step[4]:
            break

    safe_env.close()
    gym_env.close()
