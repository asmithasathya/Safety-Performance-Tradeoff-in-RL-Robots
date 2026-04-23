"""Evaluation utilities for trained PPO baseline checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from spt_envs.factory import make_env
from spt_envs.splits import get_layout_seeds
from spt_training.common import ensure_directory, read_json


EVAL_EPISODE_FIELDNAMES = (
    "baseline",
    "variant",
    "run_seed",
    "split",
    "layout_seed",
    "checkpoint_name",
    "checkpoint_timesteps",
    "episode_index",
    "episode_return",
    "episode_cost",
    "goals_achieved",
    "episode_length",
    "penalty_coeff",
    "budget",
    "lagrangian_lambda",
    "shield_warning_radius",
    "episode_shield_interventions",
    "shield_intervention_rate",
)

EVAL_SUMMARY_FIELDNAMES = (
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
    "shield_warning_radius",
    "mean_episode_return",
    "std_episode_return",
    "mean_episode_cost",
    "std_episode_cost",
    "mean_goals_achieved",
    "std_goals_achieved",
    "mean_episode_length",
    "std_episode_length",
    "mean_shield_intervention_rate",
    "std_shield_intervention_rate",
)


def build_evaluate_parser():
    """Build the CLI parser used by the evaluation script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--render", dest="render_mode", choices=("human",), default=None)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run evaluation/rendering without writing eval_episodes.csv or eval_summary.csv.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of deterministic PPO inference.",
    )
    return parser


def _resolve_run_artifacts(run_dir, checkpoint_name):
    run_dir = Path(run_dir)
    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint {!r} does not exist.".format(str(checkpoint_path))
        )
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    return run_dir, checkpoint_path, metadata


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_run(
    run_dir,
    checkpoint_name,
    split,
    output_dir=None,
    episodes_per_seed=1,
    deterministic=True,
    show_progress=True,
    render_mode=None,
    no_save=False,
):
    """Evaluate one checkpoint across every layout seed for a chosen split."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - import path only exercised locally
        raise ImportError(
            "Evaluation requires the 'train' dependencies. "
            "Install them with: pip install -e \".[train]\""
        ) from exc

    run_dir, checkpoint_path, checkpoint_metadata = _resolve_run_artifacts(
        run_dir, checkpoint_name
    )
    run_config = read_json(run_dir / "run_config.json")

    if episodes_per_seed <= 0:
        raise ValueError("episodes_per_seed must be > 0.")
    if render_mode not in (None, "human"):
        raise ValueError("render_mode must be None or 'human'.")

    if output_dir is None:
        output_dir = run_dir / "evaluations" / checkpoint_path.stem / split
    if not no_save:
        output_dir = ensure_directory(output_dir)

    if show_progress:
        print(
            "[eval] loading checkpoint={} baseline={} variant={} run_seed={} split={} deterministic={} render_mode={}".format(
                checkpoint_path.name,
                run_config["baseline"],
                run_config["variant"],
                int(run_config["seed"]),
                split,
                bool(deterministic),
                render_mode,
            )
        )
        if render_mode == "human":
            print("[eval] human rendering is enabled; evaluation will run more slowly.")

    model = PPO.load(str(checkpoint_path), device="auto")
    rows = []
    seeds = get_layout_seeds(run_config["variant"], split)
    total_eval_episodes = len(seeds) * int(episodes_per_seed)
    completed_eval_episodes = 0

    # For the shield baseline the shield must be active during evaluation too,
    # because the policy was trained with it and expects its dynamics.
    baseline = run_config.get("baseline")
    eval_env_kwargs = {}
    if baseline == "shield":
        eval_env_kwargs["shield_warning_radius"] = run_config.get("shield_warning_radius")

    for seed_index, layout_seed in enumerate(seeds, start=1):
        for episode_index in range(int(episodes_per_seed)):
            if show_progress:
                print(
                    "[eval] split={} layout_seed={} ({}/{}) episode {}/{}".format(
                        split,
                        int(layout_seed),
                        seed_index,
                        len(seeds),
                        episode_index + 1,
                        int(episodes_per_seed),
                    )
                )
            env = make_env(
                variant=run_config["variant"],
                split=split,
                layout_seed=layout_seed,
                api="gym",
                render_mode=render_mode,
                **eval_env_kwargs,
            )
            observation, info = env.reset()
            _ = info
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                _ = reward
                done = bool(terminated or truncated)
            env.close()

            rows.append(
                {
                    "baseline": run_config["baseline"],
                    "variant": run_config["variant"],
                    "run_seed": int(run_config["seed"]),
                    "split": split,
                    "layout_seed": int(layout_seed),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_timesteps": checkpoint_metadata.get("timesteps"),
                    "episode_index": int(episode_index),
                    "episode_return": float(info["episode_return"]),
                    "episode_cost": float(info["episode_cost"]),
                    "goals_achieved": int(info.get("goals_achieved", 0)),
                    "episode_length": int(info.get("episode_length", 0)),
                    "penalty_coeff": checkpoint_metadata.get("penalty_coeff"),
                    "budget": checkpoint_metadata.get("budget"),
                    "lagrangian_lambda": checkpoint_metadata.get("lagrangian_lambda"),
                    "shield_warning_radius": checkpoint_metadata.get("shield_warning_radius"),
                    "episode_shield_interventions": info.get("episode_shield_interventions"),
                    "shield_intervention_rate": info.get("shield_intervention_rate"),
                }
            )
            completed_eval_episodes += 1
            if show_progress:
                print(
                    "[eval] completed {}/{}: return={:.3f} cost={:.3f} goals={} length={}".format(
                        completed_eval_episodes,
                        total_eval_episodes,
                        float(info["episode_return"]),
                        float(info["episode_cost"]),
                        int(info.get("goals_achieved", 0)),
                        int(info.get("episode_length", 0)),
                    )
                )

    eval_episodes_path = None
    if not no_save:
        eval_episodes_path = output_dir / "eval_episodes.csv"
        _write_csv(eval_episodes_path, EVAL_EPISODE_FIELDNAMES, rows)

    returns = np.array([row["episode_return"] for row in rows], dtype=float)
    costs = np.array([row["episode_cost"] for row in rows], dtype=float)
    goals = np.array([row["goals_achieved"] for row in rows], dtype=float)
    lengths = np.array([row["episode_length"] for row in rows], dtype=float)
    shield_rates_raw = [row.get("shield_intervention_rate") for row in rows]
    shield_rates = np.array(
        [r for r in shield_rates_raw if r is not None], dtype=float
    )
    summary_row = {
        "baseline": run_config["baseline"],
        "variant": run_config["variant"],
        "run_seed": int(run_config["seed"]),
        "split": split,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_timesteps": checkpoint_metadata.get("timesteps"),
        "episodes": int(len(rows)),
        "deterministic": bool(deterministic),
        "penalty_coeff": checkpoint_metadata.get("penalty_coeff"),
        "budget": checkpoint_metadata.get("budget"),
        "lagrangian_lambda": checkpoint_metadata.get("lagrangian_lambda"),
        "shield_warning_radius": checkpoint_metadata.get("shield_warning_radius"),
        "mean_episode_return": float(returns.mean()),
        "std_episode_return": float(returns.std(ddof=0)),
        "mean_episode_cost": float(costs.mean()),
        "std_episode_cost": float(costs.std(ddof=0)),
        "mean_goals_achieved": float(goals.mean()),
        "std_goals_achieved": float(goals.std(ddof=0)),
        "mean_episode_length": float(lengths.mean()),
        "std_episode_length": float(lengths.std(ddof=0)),
        "mean_shield_intervention_rate": float(shield_rates.mean()) if len(shield_rates) else None,
        "std_shield_intervention_rate": float(shield_rates.std(ddof=0)) if len(shield_rates) else None,
    }

    eval_summary_path = None
    if not no_save:
        eval_summary_path = output_dir / "eval_summary.csv"
        _write_csv(eval_summary_path, EVAL_SUMMARY_FIELDNAMES, [summary_row])
    if show_progress:
        print(
            "[eval] finished split={} mean_return={:.3f} mean_cost={:.3f} outputs={}".format(
                split,
                float(summary_row["mean_episode_return"]),
                float(summary_row["mean_episode_cost"]),
                "not_saved" if no_save else output_dir,
            )
        )
    return {
        "eval_episodes_csv": None if eval_episodes_path is None else str(eval_episodes_path),
        "eval_summary_csv": None if eval_summary_path is None else str(eval_summary_path),
    }
