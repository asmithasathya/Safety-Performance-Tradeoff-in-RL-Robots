"""Evaluation setup and aggregation smoke tests."""

import csv

import pytest

pytest.importorskip("safety_gymnasium")

from spt_training.aggregate import aggregate_results
from spt_training.evaluate import evaluate_run
from spt_training.train import TrainingRunConfig, train_run


def test_evaluate_run_writes_expected_outputs(tmp_path):
    pytest.importorskip("stable_baselines3")

    run_dir = tmp_path / "reward_penalty_run"
    config = TrainingRunConfig(
        baseline="reward_penalty",
        variant="easy",
        seed=5,
        total_timesteps=1024,
        save_freq=512,
        output_dir=str(run_dir),
        penalty_coeff=0.3,
    )
    train_run(config, ppo_kwargs={"n_steps": 256, "batch_size": 64})

    outputs = evaluate_run(
        run_dir=run_dir,
        checkpoint_name="final_model.zip",
        split="test",
    )

    assert (run_dir / "evaluations" / "final_model" / "test" / "eval_episodes.csv").exists()
    assert (run_dir / "evaluations" / "final_model" / "test" / "eval_summary.csv").exists()
    assert outputs["eval_summary_csv"].endswith("eval_summary.csv")


def test_aggregate_results_matches_reward_penalty_to_budgets(tmp_path):
    results_root = tmp_path / "results"
    summary_dir = results_root / "reward_run" / "evaluations" / "final_model" / "train"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "eval_summary.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "baseline",
                "variant",
                "run_seed",
                "split",
                "checkpoint_name",
                "checkpoint_timesteps",
                "episodes",
                "deterministic",
                "penalty_coeff",
                "budget",
                "lagrangian_lambda",
                "mean_episode_return",
                "std_episode_return",
                "mean_episode_cost",
                "std_episode_cost",
                "mean_goals_achieved",
                "std_goals_achieved",
                "mean_episode_length",
                "std_episode_length",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "baseline": "reward_penalty",
                "variant": "easy",
                "run_seed": "0",
                "split": "train",
                "checkpoint_name": "final_model.zip",
                "checkpoint_timesteps": "1000",
                "episodes": "8",
                "deterministic": "True",
                "penalty_coeff": "0.3",
                "budget": "",
                "lagrangian_lambda": "",
                "mean_episode_return": "12.0",
                "std_episode_return": "1.0",
                "mean_episode_cost": "9.5",
                "std_episode_cost": "0.2",
                "mean_goals_achieved": "2.0",
                "std_goals_achieved": "0.1",
                "mean_episode_length": "1000.0",
                "std_episode_length": "0.0",
            }
        )

    aggregate_dir = tmp_path / "aggregated"
    outputs = aggregate_results(
        results_root=results_root,
        split="train",
        output_dir=aggregate_dir,
        budgets=(10.0,),
    )

    matches_path = aggregate_dir / "reward_penalty_budget_matches.csv"
    assert matches_path.exists()
    assert outputs["reward_penalty_budget_matches"] == str(matches_path)

    with matches_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["target_budget"] == "10.0"
    assert rows[0]["matched_penalty_coeff"] == "0.3"
