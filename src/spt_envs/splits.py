"""Reproducible layout split manifests for local PointGoal environments."""

from spt_envs.configs import VALID_SPLITS, VARIANT_SPECS, get_variant_spec

TRAIN_LAYOUT_SEEDS = tuple(range(24))
TEST_LAYOUT_SEEDS = tuple(range(100, 108))

SPLIT_MANIFESTS = {
    variant: {
        "train": TRAIN_LAYOUT_SEEDS,
        "test": TEST_LAYOUT_SEEDS,
    }
    for variant in VARIANT_SPECS
}


def get_split_manifest(variant):
    """Return the full train/test manifest for one variant."""
    get_variant_spec(variant)
    return SPLIT_MANIFESTS[variant]


def get_layout_seeds(variant, split):
    """Return the tuple of allowed layout seeds for a variant/split pair."""
    if split not in VALID_SPLITS:
        raise ValueError(
            "Unknown split {!r}. Expected one of {}.".format(
                split,
                ", ".join(VALID_SPLITS),
            ),
        )
    return get_split_manifest(variant)[split]


def validate_layout_seed(variant, split, layout_seed):
    """Validate and normalize a requested layout seed."""
    seeds = get_layout_seeds(variant, split)
    if layout_seed not in seeds:
        raise ValueError(
            "layout_seed {!r} is not valid for variant={!r}, split={!r}. "
            "Allowed seeds: {}.".format(layout_seed, variant, split, list(seeds))
        )
    return int(layout_seed)
