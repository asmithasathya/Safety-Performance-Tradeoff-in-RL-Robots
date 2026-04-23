# Safety-Performance-Tradeoff-in-RL-Robots

This repository studies the true performance cost of different safety levels in robot learning. The current repo now supports:

- local Safety-Gymnasium PointGoal variants (`easy`, `medium`, `hard`)
- a hand-tuned reward-penalty baseline
- a Lagrangian constrained RL baseline with learned `lambda`
- PPO training, checkpointing, and machine-readable training logs
- automatic final-checkpoint evaluation on train and held-out test layouts
- post-processing to match reward-penalty runs to target budgets

The project target budgets are `B ∈ {0, 5, 10, 20, 35}`.

## Environment

All variants use:

- Point agent only
- continuing PointGoal episodes
- `continue_goal=True`
- `constrain_indicator=True`
- 1000-step episode horizon
- hazard-only safety semantics
- the same reward mechanics and observation/action spaces

Difficulty differs only by hazard count:

| Variant | Env ID | Hazards | Train Layout Seeds | Test Layout Seeds |
| --- | --- | ---: | ---: | ---: |
| easy | `SPTPointGoalEasy-v0` | 4 | 32 | 8 |
| medium | `SPTPointGoalMedium-v0` | 8 | 32 | 8 |
| hard | `SPTPointGoalHard-v0` | 12 | 32 | 8 |

Train and test splits are explicit in `src/spt_envs/splits.py`.

## Reward, Cost, and Budgets

The environment uses the benchmark reward unchanged.

- `episode_return`: raw task return over one full episode
- `episode_cost`: cumulative unsafe timesteps over one episode
- `goals_achieved`: number of goals completed during the episode
- `episode_length`: number of steps in the episode

The budget `B` constrains `episode_cost`.

Because `constrain_indicator=True`, each unsafe step contributes binary cost before aggregation. In this repo, a budget such as `B = 10` means the policy is trying to stay under about 10 unsafe timesteps per 1000-step episode.

## Baselines

### Reward-Penalty

Reward-penalty uses a fixed hand-tuned coefficient:

`r_shaped = r - lambda * cost`

- `lambda` is the fixed `penalty_coeff`
- there is no direct budget argument during training
- to compare against target budgets `{0, 5, 10, 20, 35}`, train a sweep over penalty coefficients and later match runs by realized `episode_cost`

Recommended starting coefficient sweep:

`lambda ∈ {0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0}`

### Lagrangian

Lagrangian training uses a learned dual variable:

`r_shaped = r - lambda * cost`

At the end of each episode:

`lambda <- max(0, lambda + lr * (episode_cost - B))`

That means:

- if `episode_cost > B`, `lambda` increases and the next episode penalizes unsafe behavior more heavily
- if `episode_cost < B`, `lambda` decreases and the policy can care more about task reward
- if `episode_cost ≈ B`, `lambda` stabilizes near the level that balances reward and safety

Unlike reward-penalty, Lagrangian training takes the target budget `B` directly.

## Public Interfaces

For fixed-layout envs and evaluation:

```python
from spt_envs.factory import make_env

env = make_env(
    variant="medium",
    split="test",
    layout_seed=100,
    api="gym",
)
```

For training across the full train split:

```python
from spt_envs.factory import make_train_env

env = make_train_env(
    variant="medium",
    seed=0,
    api="gym",
    penalty_coeff=1.0,
)
```

`make_train_env(...)` samples one approved `train` layout seed on each reset and preserves the sampled `layout_seed` in `info`.

## Install

CPU-only setup is supported.

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate safe-rl-tradeoff
```

Install the package plus training and test dependencies:

```bash
pip install -e ".[train]"
```

## Environment Smoke Check

Run a quick non-training environment check:

```bash
python scripts/check_env.py --variant medium --split train --layout-seed 0 --episodes 2
```

Optional render:

```bash
python scripts/check_env.py --variant easy --split train --layout-seed 0 --episodes 1 --render human
```

## Training

Training saves checkpoints and logs, then automatically evaluates the final checkpoint on both the `train` and `test` splits.

If you want to watch the simulator live for debugging or demos, add `--render human` to the training or manual evaluation command. Leave rendering off for normal experiments because it slows the run down significantly.

You will see console progress during:

- PPO training start
- each completed training episode
- each saved checkpoint
- automatic final-checkpoint evaluation on `train`
- automatic final-checkpoint evaluation on `test`

Each training run writes:

- `run_config.json`
- `train_metrics.csv`
- `evaluation_manifest.json`
- `checkpoints/`
- `final_model.zip`
- `final_model.json`
- automatic `eval_episodes.csv` and `eval_summary.csv` outputs for `train`
- automatic `eval_episodes.csv` and `eval_summary.csv` outputs for `test`

### Full Sweep Scripts

If you want to launch a full baseline sweep inside `tmux` and let every run execute one after another, use these batch scripts:

```bash
bash scripts/run_reward_penalty_sweep.sh
bash scripts/run_lagrangian_sweep.sh
```

Each script runs every combination it is responsible for and skips runs that already have both `final_model.zip` and `evaluation_manifest.json`.

During the sweep, each script prints the current run number, completed count, skipped count, remaining count, and elapsed time.

Default sweep coverage:

- reward-penalty: `3 variants × 3 seeds × 7 penalty coefficients = 63 runs`
- Lagrangian: `3 variants × 3 seeds × 5 budgets = 45 runs`

The scripts use these defaults unless you override them in the shell before launching:

- `TOTAL_TIMESTEPS=250000`
- `SAVE_FREQ=50000`
- `OUTPUT_ROOT=results`
- `RENDER_MODE=` left empty by default
- `LAGRANGIAN_LR=0.05` for the Lagrangian sweep
- `LAGRANGIAN_INIT_LAMBDA=0.0` for the Lagrangian sweep

Example `tmux` workflow:

```bash
tmux new -s saferl
conda activate safe-rl-tradeoff
cd /Users/asmitha/robot-learning/Safety-Performance-Tradeoff-in-RL-Robots
bash scripts/run_reward_penalty_sweep.sh
```

### First Experiment To Run

If you want to confirm the full training pipeline works before launching a large sweep, start with this small reward-penalty pilot run:

```bash
python scripts/train_baseline.py \
  --baseline reward_penalty \
  --variant easy \
  --seed 0 \
  --total-timesteps 25000 \
  --save-freq 5000 \
  --penalty-coeff 0.3 \
  --output-dir results/reward_penalty/easy/seed0_lambda0p3_pilot
```

What this command means:

- `--baseline reward_penalty`: train the fixed-penalty baseline, not Lagrangian
- `--variant easy`: use the easiest hazard configuration first
- `--seed 0`: use run seed `0` for this training run
- `--total-timesteps 25000`: give PPO a small pilot budget of 25,000 environment steps
- `--save-freq 5000`: save checkpoints every 5,000 training steps
- `--penalty-coeff 0.3`: use a mild fixed safety penalty `lambda = 0.3`
- `--output-dir ...`: write all checkpoints and logs into one dedicated results folder

This is a pilot run, not a final experiment. It is meant to verify that:

- training launches correctly
- checkpoints are saved
- `train_metrics.csv` is populated
- automatic evaluation runs after training completes
- the output folder structure looks right before you scale up to longer runs

If you want to watch that same run in the simulator, add:

```bash
--render human
```

### Reward-Penalty Example

```bash
python scripts/train_baseline.py \
  --baseline reward_penalty \
  --variant medium \
  --seed 0 \
  --total-timesteps 250000 \
  --save-freq 50000 \
  --penalty-coeff 1.0 \
  --output-dir results/reward_penalty/medium/seed0_lambda1p0
```

### Lagrangian Example

```bash
python scripts/train_baseline.py \
  --baseline lagrangian \
  --variant medium \
  --seed 0 \
  --total-timesteps 1000000 \
  --save-freq 50000 \
  --budget 10 \
  --lagrangian-lr 0.05 \
  --output-dir results/lagrangian/medium/seed0_budget10
```

### Manual Evaluation With Rendering

If you want to watch a trained checkpoint instead of training live, run:

```bash
python scripts/evaluate_baseline.py \
  --run-dir results/reward_penalty/easy/seed0_lambda0p3_up_pilot \
  --checkpoint final_model.zip \
  --split test \
  --render human
```

### Metrics Logged During Training

`train_metrics.csv` records:

- `episode_return`
- `episode_penalized_return`
- `episode_cost`
- `goals_achieved`
- `episode_length`
- training timestep
- elapsed time
- steps per second
- baseline name
- variant
- run seed
- sampled `layout_seed`
- reward-penalty `penalty_coeff`
- Lagrangian `budget`
- Lagrangian `lambda` values

This gives you enough information to analyze learning speed later without running evaluation first.

## Evaluation

Evaluation is automatically triggered at the end of each training run for the final checkpoint.

You can still run evaluation manually if you want to re-evaluate a saved checkpoint, change the split, or rerun with different options.

To evaluate a checkpoint on all held-out test layouts:

```bash
python scripts/evaluate_baseline.py \
  --run-dir results/lagrangian/medium/seed0_budget10 \
  --checkpoint final_model.zip \
  --split test
```

To evaluate on the train split instead:

```bash
python scripts/evaluate_baseline.py \
  --run-dir results/reward_penalty/medium/seed0_lambda1p0 \
  --checkpoint final_model.zip \
  --split train
```

Each evaluation run writes:

- `eval_episodes.csv`
- `eval_summary.csv`

By default these go under:

`<run-dir>/evaluations/<checkpoint-name-without-zip>/<split>/`

## Aggregation

Reward-penalty runs must be matched to budgets after evaluation. Lagrangian runs already carry their target budget.

Once you have evaluation summaries, aggregate them with:

```bash
python scripts/aggregate_budget_results.py \
  --results-root results \
  --split train \
  --output-dir results/aggregated/train
```

This writes:

- `reward_penalty_budget_matches.csv`
- `lagrangian_budget_summary.csv`

Use train-split aggregation to match reward-penalty coefficient sweeps to target budgets `{0, 5, 10, 20, 35}`. After that, run the corresponding held-out test evaluations for robustness reporting.

### Reward-Penalty Plots

Generate the main reward-penalty result figures with:

```bash
python scripts/plot_reward_penalty_results.py \
  --results-root results \
  --output-dir results/figures/reward_penalty
```

If Matplotlib is missing, install the analysis extra with `pip install -e ".[analysis]"`.

This verifies `eval_summary.csv` against `eval_episodes.csv`, matches fixed-penalty runs to target budgets on the train split, derives success as `goals_achieved > 0`, and writes:

- `reward_penalty_test_task_return_by_budget.png`
- `reward_penalty_test_success_rate_by_budget.png`
- `reward_penalty_train_lambda_matching_by_budget.png`
- `reward_penalty_test_mistakes_by_lambda.png`

### Lagrangian Plots

Generate the main Lagrangian result figures with:

```bash
python scripts/plot_lagrangian_results.py \
  --results-root results \
  --output-dir results/figures/lagrangian
```

This verifies `eval_summary.csv` against `eval_episodes.csv`, derives success as `goals_achieved > 0`, and writes:

- `lagrangian_test_task_return_by_budget.png`
- `lagrangian_test_success_rate_by_budget.png`
- `lagrangian_test_constraint_violations_by_budget.png`
- `lagrangian_learned_lambda_by_budget.png`

### Medium Comparison Plots

Compare reward-penalty and Lagrangian on the medium task with:

```bash
python scripts/plot_medium_comparison.py \
  --results-root results \
  --output-dir results/figures/medium_comparison \
  --variant medium
```

This writes:

- `medium_reward_penalty_vs_lagrangian_budget_task_return.png`
- `medium_reward_penalty_vs_lagrangian_learning_speed.png`
- `medium_reward_penalty_vs_lagrangian_robustness.png`

## Repo Layout

- `src/spt_envs/`: environment definitions, wrappers, and factories
- `src/spt_training/`: training, evaluation, and aggregation utilities
- `scripts/train_baseline.py`: train one baseline run
- `scripts/evaluate_baseline.py`: evaluate one checkpoint on a split
- `scripts/aggregate_budget_results.py`: budget-oriented result aggregation
- `tests/`: env, baseline, training, evaluation, and aggregation tests

## Troubleshooting

- If `safety_gymnasium` or MuJoCo fails to import, recreate the Conda env and reinstall with `pip install -e ".[train]"`.
- If `stable_baselines3` or `torch` is missing, the training and evaluation scripts will fail until the `train` extra is installed.
- If a layout seed is rejected in fixed-layout evaluation, check `src/spt_envs/splits.py`.
- If `output-dir` already exists and contains files, the training runner will refuse to overwrite it.
