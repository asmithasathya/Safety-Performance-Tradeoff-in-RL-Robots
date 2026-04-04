"""Shared configuration for project-local Safety-Gymnasium environments."""

DEFAULT_MAX_EPISODE_STEPS = 300
DEFAULT_AGENT_NAME = "Point"
DEFAULT_API = "gym"
VALID_APIS = ("safe", "gym")
VALID_SPLITS = ("train", "test")
DEFAULT_EXTENTS = (-2.0, -2.0, 2.0, 2.0)
DEFAULT_HAZARD_KEEPOUT = 0.18

EPISODE_SUMMARY_KEYS = (
    "episode_return",
    "episode_cost",
    "goals_achieved",
    "episode_length",
)

STANDARD_STEP_INFO_KEYS = (
    "cost",
    "goal_achieved",
    "variant",
    "split",
    "layout_seed",
)

VARIANT_SPECS = {
    "easy": {
        "env_id": "SPTPointGoalEasy-v0",
        "label": "easy",
        "hazards": 4,
        "placements_extents": DEFAULT_EXTENTS,
    },
    "medium": {
        "env_id": "SPTPointGoalMedium-v0",
        "label": "medium",
        "hazards": 8,
        "placements_extents": DEFAULT_EXTENTS,
    },
    "hard": {
        "env_id": "SPTPointGoalHard-v0",
        "label": "hard",
        "hazards": 12,
        "placements_extents": DEFAULT_EXTENTS,
    },
}

ENV_ID_TO_VARIANT = {
    spec["env_id"]: variant for variant, spec in VARIANT_SPECS.items()
}


def get_variant_spec(variant):
    """Return the canonical configuration for a local task variant."""
    try:
        return VARIANT_SPECS[variant]
    except KeyError as exc:
        raise ValueError(
            "Unknown variant {!r}. Expected one of {}.".format(
                variant,
                ", ".join(sorted(VARIANT_SPECS)),
            ),
        ) from exc
