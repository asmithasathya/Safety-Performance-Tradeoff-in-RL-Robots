"""Training runner validation and smoke tests."""

import csv

import pytest

pytest.importorskip("safety_gymnasium")

from spt_training.train import (
    TrainingRunConfig,
    build_train_parser,
    train_run,
    training_run_config_from_args,
)


def test_reward_penalty_requires_penalty_coeff():
    parser = build_train_parser()
    args = parser.parse_args(
        [
            "--baseline",
            "reward_penalty",
            "--variant",
            "easy",
            "--seed",
            "0",
            "--total-timesteps",
            "1000",
            "--output-dir",
            "tmp/run",
        ]
    )
    with pytest.raises(ValueError, match="penalty-coeff"):
        training_run_config_from_args(args)


def test_lagrangian_requires_budget_and_lr():
    parser = build_train_parser()
    args = parser.parse_args(
        [
            "--baseline",
            "lagrangian",
            "--variant",
            "easy",
            "--seed",
            "0",
            "--total-timesteps",
            "1000",
            "--output-dir",
            "tmp/run",
        ]
    )
    with pytest.raises(ValueError, match="budget"):
        training_run_config_from_args(args)


@pytest.mark.parametrize(
    ("baseline", "kwargs"),
    (
        ("reward_penalty", {"penalty_coeff": 0.3}),
        ("lagrangian", {"budget": 10.0, "lagrangian_lr": 0.05}),
    ),
)
def test_tiny_training_run_writes_artifacts(tmp_path, baseline, kwargs):
    pytest.importorskip("stable_baselines3")

    run_dir = tmp_path / baseline
    config = TrainingRunConfig(
        baseline=baseline,
        variant="easy",
        seed=3,
        total_timesteps=1024,
        save_freq=512,
        output_dir=str(run_dir),
        **kwargs,
    )
    outputs = train_run(
        config,
        ppo_kwargs={"n_steps": 256, "batch_size": 64},
    )

    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "train_metrics.csv").exists()
    assert (run_dir / "final_model.zip").exists()
    assert (run_dir / "evaluation_manifest.json").exists()
    assert outputs["run_dir"] == str(run_dir)

    with (run_dir / "train_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "expected at least one completed episode metric row"
    row = rows[-1]
    assert row["baseline"] == baseline
    assert row["variant"] == "easy"
    assert row["episode_return"] != ""
    assert row["episode_cost"] != ""
