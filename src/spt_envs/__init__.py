"""Project-local Safety-Gymnasium environments for safety tradeoff experiments."""

from spt_envs.configs import DEFAULT_MAX_EPISODE_STEPS, VARIANT_SPECS
from spt_envs.splits import SPLIT_MANIFESTS, get_layout_seeds, validate_layout_seed
from spt_envs.wrappers import RewardPenaltyWrapper

__all__ = [
    "DEFAULT_MAX_EPISODE_STEPS",
    "RewardPenaltyWrapper",
    "SPLIT_MANIFESTS",
    "VARIANT_SPECS",
    "get_layout_seeds",
    "make_env",
    "register_envs",
    "validate_layout_seed",
]


def __getattr__(name):
    if name == "make_env":
        from spt_envs.factory import make_env

        return make_env
    if name == "register_envs":
        from spt_envs.registry import register_envs

        return register_envs
    raise AttributeError("module 'spt_envs' has no attribute {!r}".format(name))
