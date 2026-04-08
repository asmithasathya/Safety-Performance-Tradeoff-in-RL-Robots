"""Aggregate evaluation summaries into budget-oriented experiment tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from spt_training.common import DEFAULT_BUDGETS, ensure_directory


def build_aggregate_parser():
    """Build the CLI parser used by the aggregation script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=list(DEFAULT_BUDGETS),
    )
    return parser


def _read_eval_summary_rows(results_root):
    rows = []
    for summary_path in Path(results_root).rglob("eval_summary.csv"):
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["summary_path"] = str(summary_path)
                rows.append(row)
    return rows


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_results(results_root, split, output_dir, budgets=DEFAULT_BUDGETS):
    """Aggregate eval summaries into reward-penalty and Lagrangian budget tables."""
    rows = [row for row in _read_eval_summary_rows(results_root) if row.get("split") == split]
    if not rows:
        raise FileNotFoundError(
            "No eval_summary.csv files found for split={!r} under {!r}.".format(
                split, str(results_root)
            )
        )

    budgets = tuple(float(budget) for budget in budgets)
    output_dir = ensure_directory(output_dir)

    reward_penalty_rows = []
    grouped_reward_penalty = defaultdict(list)
    lagrangian_rows = []

    for row in rows:
        baseline = row.get("baseline")
        if baseline == "reward_penalty":
            key = (row.get("variant"), row.get("split"), row.get("run_seed"))
            grouped_reward_penalty[key].append(row)
        elif baseline == "lagrangian":
            budget = float(row.get("budget"))
            if budget in budgets:
                lagrangian_rows.append(row)

    for (variant, split_name, run_seed), candidates in grouped_reward_penalty.items():
        for target_budget in budgets:
            best_row = min(
                candidates,
                key=lambda row: (
                    abs(float(row.get("mean_episode_cost")) - target_budget),
                    -float(row.get("mean_episode_return")),
                ),
            )
            reward_penalty_rows.append(
                {
                    "variant": variant,
                    "split": split_name,
                    "run_seed": run_seed,
                    "target_budget": float(target_budget),
                    "matched_penalty_coeff": best_row.get("penalty_coeff"),
                    "matched_mean_episode_cost": best_row.get("mean_episode_cost"),
                    "matched_mean_episode_return": best_row.get("mean_episode_return"),
                    "matched_checkpoint_name": best_row.get("checkpoint_name"),
                    "cost_gap": abs(
                        float(best_row.get("mean_episode_cost")) - float(target_budget)
                    ),
                    "summary_path": best_row.get("summary_path"),
                }
            )

    reward_penalty_path = output_dir / "reward_penalty_budget_matches.csv"
    reward_penalty_fieldnames = (
        "variant",
        "split",
        "run_seed",
        "target_budget",
        "matched_penalty_coeff",
        "matched_mean_episode_cost",
        "matched_mean_episode_return",
        "matched_checkpoint_name",
        "cost_gap",
        "summary_path",
    )
    _write_csv(reward_penalty_path, reward_penalty_fieldnames, reward_penalty_rows)

    lagrangian_path = output_dir / "lagrangian_budget_summary.csv"
    lagrangian_fieldnames = (
        "variant",
        "split",
        "run_seed",
        "budget",
        "lagrangian_lambda",
        "mean_episode_cost",
        "mean_episode_return",
        "mean_goals_achieved",
        "checkpoint_name",
        "summary_path",
    )
    lagrangian_output_rows = [
        {
            "variant": row.get("variant"),
            "split": row.get("split"),
            "run_seed": row.get("run_seed"),
            "budget": row.get("budget"),
            "lagrangian_lambda": row.get("lagrangian_lambda"),
            "mean_episode_cost": row.get("mean_episode_cost"),
            "mean_episode_return": row.get("mean_episode_return"),
            "mean_goals_achieved": row.get("mean_goals_achieved"),
            "checkpoint_name": row.get("checkpoint_name"),
            "summary_path": row.get("summary_path"),
        }
        for row in lagrangian_rows
    ]
    _write_csv(lagrangian_path, lagrangian_fieldnames, lagrangian_output_rows)

    return {
        "reward_penalty_budget_matches": str(reward_penalty_path),
        "lagrangian_budget_summary": str(lagrangian_path),
    }
