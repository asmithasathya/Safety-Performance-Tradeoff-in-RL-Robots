"""Environment creation smoke tests."""

import pytest

pytest.importorskip("safety_gymnasium")

from spt_envs.factory import make_env
from spt_envs.splits import get_layout_seeds


@pytest.mark.parametrize("variant", ("easy", "medium", "hard"))
@pytest.mark.parametrize("split", ("train", "test"))
@pytest.mark.parametrize("api", ("safe", "gym"))
def test_env_creation_across_variants_and_apis(variant, split, api):
    layout_seed = get_layout_seeds(variant, split)[0]
    env = make_env(variant=variant, split=split, layout_seed=layout_seed, api=api)
    observation, info = env.reset()

    assert observation is not None
    assert info["variant"] == variant
    assert info["split"] == split
    assert info["layout_seed"] == layout_seed
    assert env.action_space is not None
    assert env.observation_space is not None

    env.close()
