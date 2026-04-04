# Safety-Performance-Tradeoff-in-RL-Robots

This repository studies the true performance cost of different safety levels in robot learning. Instead of training one "safe agent" and stopping there, the project is designed to train policies across safety budgets `B ∈ {0, 1, 2, 5, 10}`, trace the full return-versus-violation frontier, and recommend an operating point.

The current milestone is environment setup only. No baselines, budget sweeps, Pareto plotting, or ablation code has been implemented yet.

## Why Safety-Gymnasium

The environment layer is built on top of Safety-Gymnasium because it gives the project:

- benchmark-recognizable safe-RL tasks
- native cost-aware step outputs
- CPU-friendly local experimentation
- a clear path to constrained RL, reward-penalty, shielding, and later RLHF-style comparisons

This repo does not use raw official `Goal0/1/2` unchanged.

- `Goal0` has no safety cost, which weakens the "different safety levels" story.
- `Goal2` mixes hazard and vase costs, which makes budget semantics less consistent across variants.

Instead, this repo registers three local hazard-only PointGoal variants that keep the same task family and reward mechanics while making the safety signal consistent across difficulty levels.

## Current Milestone

Implemented now:

- Conda-first project setup
- three local PointGoal variants: `easy`, `medium`, `hard`
- reproducible `train` and `test` layout split manifests
- a stable environment factory with safe and gym-style APIs
- episode-level logging of return, cumulative cost, goals achieved, and episode length
- optional in-memory trajectory recording
- smoke-check script and tests

Deferred for later:

- reward-penalty baseline
- Lagrangian constrained RL baseline
- rule-based shielding baseline
- RLHF-style baseline
- budget sweeps over `B ∈ {0, 1, 2, 5, 10}`
- Pareto frontier plotting
- ablation studies

## Environment Definitions

All variants use:

- Point agent only
- continuing PointGoal episodes
- `continue_goal=True`
- `constrain_indicator=True`
- 300-step episode horizon
- hazard-only safety semantics
- the same reward mechanics and observation/action spaces

Difficulty differs only by hazard count:

| Variant | Env ID | Hazards | Train Layout Seeds | Test Layout Seeds |
| --- | --- | ---: | ---: | ---: |
| easy | `SPTPointGoalEasy-v0` | 4 | 24 | 8 |
| medium | `SPTPointGoalMedium-v0` | 8 | 24 | 8 |
| hard | `SPTPointGoalHard-v0` | 12 | 24 | 8 |

The split manifests are explicit and checked into the repo in [src/spt_envs/splits.py](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/src/spt_envs/splits.py).

## Reward, Cost, and Budget Semantics

The project uses the benchmark reward unchanged and defines safety in terms of cumulative episode cost.

- `episode_cost` is the sum of per-step costs over one episode.
- This `episode_cost` is the quantity compared against budget `B`.
- Because `constrain_indicator=True`, each unsafe step contributes a binary cost signal before episode aggregation.
- `goals_achieved` is the success-style metric for this continuing task family.

This means later constrained-RL methods can treat `B` as an episode-level cost budget while still evaluating return and task progress in a benchmark-like navigation setting.

## Public Interface

The main entry point is `make_env(...)` in [src/spt_envs/factory.py](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/src/spt_envs/factory.py).

```python
from spt_envs.factory import make_env

env = make_env(
    variant="medium",
    split="train",
    layout_seed=0,
    api="gym",
    render_mode=None,
    record_trajectory=False,
)
```

Arguments:

- `variant`: one of `easy`, `medium`, `hard`
- `split`: one of `train`, `test`
- `layout_seed`: must belong to that variant's split manifest
- `api="safe"` returns `(obs, reward, cost, terminated, truncated, info)`
- `api="gym"` returns `(obs, reward, terminated, truncated, info)` with `info["cost"]`

Standardized `info` fields:

- per step: `cost`, `goal_achieved`, `variant`, `split`, `layout_seed`
- end of episode: `episode_return`, `episode_cost`, `goals_achieved`, `episode_length`

## Install

CPU-only setup is supported.

Recommended setup:

```bash
conda env create -f environment.yml
conda activate safe-rl-tradeoff
pip install -e .
```

Manual fallback:

```bash
conda create -n safe-rl-tradeoff python=3.10 -y
conda activate safe-rl-tradeoff
pip install -e .
```

Install test dependencies as well:

```bash
pip install -e ".[dev]"
```

## Verification

Run the environment smoke check:

```bash
python scripts/check_env.py --variant medium --split train --layout-seed 0 --episodes 2
```

Run the test suite:

```bash
pytest -q
```

The smoke-check script prints:

- env ID
- observation space
- action space
- episode return
- episode cost
- goals achieved
- episode length

## Optional Render

You can inspect one rollout with rendering:

```bash
python scripts/check_env.py --variant easy --split train --layout-seed 0 --episodes 1 --render human
```

If you want to record trajectories in memory for later analysis:

```bash
python scripts/check_env.py --variant easy --split train --layout-seed 0 --episodes 1 --record-trajectory
```

## Repo Layout

- [environment.yml](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/environment.yml): Conda environment definition
- [pyproject.toml](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/pyproject.toml): packaging and dependencies
- [src/spt_envs/registry.py](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/src/spt_envs/registry.py): local task definitions and env registration
- [src/spt_envs/factory.py](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/src/spt_envs/factory.py): main env factory
- [src/spt_envs/wrappers.py](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/src/spt_envs/wrappers.py): standardized metadata, cost, and API wrappers
- [src/spt_envs/logging.py](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/src/spt_envs/logging.py): optional trajectory capture
- [scripts/check_env.py](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/scripts/check_env.py): smoke-check utility
- [tests/test_env_creation.py](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/tests/test_env_creation.py): creation smoke tests

## Troubleshooting

- If `import safety_gymnasium` or MuJoCo setup fails, recreate the Conda environment and reinstall the package inside that environment.
- If rendering fails, first verify the non-rendered smoke check without `--render`.
- If a layout seed is rejected, make sure it belongs to the chosen variant and split manifest in [src/spt_envs/splits.py](/Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots/src/spt_envs/splits.py).
- If `pytest` fails before dependencies are installed, run `pip install -e ".[dev]"` inside the Conda environment.

## Notes for Later Milestones

This setup is intentionally baseline-agnostic.

- The safe API supports constrained RL methods directly.
- The gym-style adapter supports standard RL code that expects Gymnasium step signatures.
- The explicit split manifests support held-out layout robustness evaluation later.
- The trajectory recorder gives a starting point for future RLHF-style preference or demonstration data collection.
