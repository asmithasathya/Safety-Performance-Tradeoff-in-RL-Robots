"""Training environment sampler tests."""

import pytest

pytest.importorskip("safety_gymnasium")

from spt_envs.factory import make_train_env
from spt_envs.splits import get_layout_seeds


def test_make_train_env_samples_only_train_seeds_and_is_reproducible():
    env_a = make_train_env(variant="easy", seed=11, api="safe")
    env_b = make_train_env(variant="easy", seed=11, api="safe")

    sampled_a = []
    sampled_b = []
    for _ in range(5):
        _, info_a = env_a.reset()
        _, info_b = env_b.reset()
        sampled_a.append(info_a["layout_seed"])
        sampled_b.append(info_b["layout_seed"])

    assert sampled_a == sampled_b
    assert set(sampled_a).issubset(set(get_layout_seeds("easy", "train")))

    env_a.close()
    env_b.close()
