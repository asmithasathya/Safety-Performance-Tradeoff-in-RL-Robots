"""Split manifest integrity tests."""

import pytest

from spt_envs.configs import VARIANT_SPECS
from spt_envs.splits import get_layout_seeds, validate_layout_seed


def test_split_manifests_are_disjoint_and_expected_size():
    for variant in VARIANT_SPECS:
        train = get_layout_seeds(variant, "train")
        test = get_layout_seeds(variant, "test")
        assert len(train) == 32
        assert len(test) == 8
        assert set(train).isdisjoint(test)


def test_invalid_layout_seed_is_rejected():
    with pytest.raises(ValueError):
        validate_layout_seed("easy", "train", 999)
