"""Training utilities for reward-penalty and Lagrangian PPO baselines."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from spt_envs.factory import make_train_env
from spt_training.common import DEFAULT_PPO_KWARGS, ensure_directory, write_json


TRAIN_METRIC_FIELDNAMES = (
    "timesteps",
    "elapsed_seconds",
    "steps_per_second",
    "baseline",
    "variant",
    "run_seed",
    "split",
    "layout_seed",
    "episode_return",
    "episode_penalized_return",
    "episode_cost",
    "goals_achieved",
    "episode_length",
    "penalty_coeff",
    "budget",
    "lagrangian_lambda",
    "lagrangian_lambda_before_update",
    "lagrangian_lambda_after_update",
)


@dataclass
class TrainingRunConfig:
    """Concrete training configuration for one baseline run."""

    baseline: str
    variant: str
    seed: int
    total_timesteps: int
    save_freq: int
    output_dir: str
    penalty_coeff: float | None = None
    budget: float | None = None
    lagrangian_lr: float | None = None
    lagrangian_init_lambda: float = 0.0

    def validate(self):
        """Validate the baseline-specific argument contract."""
        if self.baseline not in ("reward_penalty", "lagrangian"):
            raise ValueError(
                "baseline must be 'reward_penalty' or 'lagrangian', got {!r}".format(
                    self.baseline
                )
            )
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be > 0.")
        if self.save_freq <= 0:
            raise ValueError("save_freq must be > 0.")
        if self.baseline == "reward_penalty":
            if self.penalty_coeff is None:
                raise ValueError(
                    "reward_penalty baseline requires --penalty-coeff."
                )
            if self.penalty_coeff < 0:
                raise ValueError("penalty_coeff must be >= 0.")
            if (
                self.budget is not None
                or self.lagrangian_lr is not None
                or self.lagrangian_init_lambda != 0.0
            ):
                raise ValueError(
                    "reward_penalty baseline does not accept Lagrangian arguments."
                )
        if self.baseline == "lagrangian":
            if self.budget is None:
                raise ValueError("lagrangian baseline requires --budget.")
            if self.lagrangian_lr is None:
                raise ValueError("lagrangian baseline requires --lagrangian-lr.")
            if self.budget < 0:
                raise ValueError("budget must be >= 0.")
            if self.lagrangian_lr <= 0:
                raise ValueError("lagrangian_lr must be > 0.")
            if self.lagrangian_init_lambda < 0:
                raise ValueError("lagrangian_init_lambda must be >= 0.")
            if self.penalty_coeff is not None:
                raise ValueError(
                    "lagrangian baseline does not accept --penalty-coeff."
                )

    def to_dict(self):
        """Serialize the run config for JSON output."""
        return asdict(self)


def build_train_parser():
    """Build the CLI parser used by the training script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=("reward_penalty", "lagrangian"),
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=("easy", "medium", "hard"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--penalty-coeff", type=float, default=None)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--lagrangian-lr", type=float, default=None)
    parser.add_argument("--lagrangian-init-lambda", type=float, default=0.0)
    return parser


def training_run_config_from_args(args):
    """Create and validate a run config from argparse output."""
    config = TrainingRunConfig(
        baseline=args.baseline,
        variant=args.variant,
        seed=int(args.seed),
        total_timesteps=int(args.total_timesteps),
        save_freq=int(args.save_freq),
        output_dir=args.output_dir,
        penalty_coeff=args.penalty_coeff,
        budget=args.budget,
        lagrangian_lr=args.lagrangian_lr,
        lagrangian_init_lambda=args.lagrangian_init_lambda,
    )
    config.validate()
    return config


def _baseline_env_kwargs(config):
    if config.baseline == "reward_penalty":
        return {"penalty_coeff": config.penalty_coeff}
    return {
        "lagrangian_budget": config.budget,
        "lagrangian_lr": config.lagrangian_lr,
        "lagrangian_init_lambda": config.lagrangian_init_lambda,
    }


def _extract_baseline_state(training_env, config):
    if config.baseline == "reward_penalty":
        penalty_coeff = training_env.get_attr("penalty_coeff")[0]
        return {"penalty_coeff": float(penalty_coeff)}
    lambda_value = training_env.get_attr("lambda_")[0]
    return {
        "budget": float(config.budget),
        "lagrangian_lambda": float(lambda_value),
    }


def _checkpoint_metadata(
    config,
    checkpoint_path,
    timesteps,
    elapsed_seconds,
    steps_per_second,
    training_env,
):
    payload = {
        "baseline": config.baseline,
        "variant": config.variant,
        "run_seed": int(config.seed),
        "timesteps": int(timesteps),
        "checkpoint_path": str(checkpoint_path),
        "elapsed_seconds": float(elapsed_seconds),
        "steps_per_second": float(steps_per_second),
    }
    payload.update(_baseline_env_kwargs(config))
    payload.update(_extract_baseline_state(training_env, config))
    return payload


def _write_train_metric_row(metrics_path, row):
    with Path(metrics_path).open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_METRIC_FIELDNAMES)
        writer.writerow(row)


def train_run(config, ppo_kwargs=None):
    """Train one PPO baseline run and save checkpoints plus machine-readable metrics."""
    config.validate()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - import path only exercised locally
        raise ImportError(
            "Training requires the 'train' dependencies. "
            "Install them with: pip install -e \".[train,dev]\""
        ) from exc

    run_dir = Path(config.output_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(
            "output_dir {!r} already exists and is not empty.".format(str(run_dir))
        )
    ensure_directory(run_dir)
    checkpoints_dir = ensure_directory(run_dir / "checkpoints")
    metrics_path = run_dir / "train_metrics.csv"
    resolved_ppo_kwargs = dict(DEFAULT_PPO_KWARGS)
    if ppo_kwargs:
        resolved_ppo_kwargs.update(ppo_kwargs)
    run_config_payload = config.to_dict()
    run_config_payload["ppo_kwargs"] = resolved_ppo_kwargs
    write_json(run_dir / "run_config.json", run_config_payload)

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_METRIC_FIELDNAMES)
        writer.writeheader()

    env = make_train_env(
        variant=config.variant,
        seed=config.seed,
        api="gym",
        **_baseline_env_kwargs(config),
    )

    class EpisodeMetricsCallback(BaseCallback):
        """Record episode metrics and save checkpoints during PPO training."""

        def __init__(self):
            super().__init__()
            self.start_time = None
            self.last_checkpoint_timestep = 0
            self.checkpoint_records = []

        def _elapsed(self):
            return max(0.0, time.monotonic() - self.start_time)

        def _steps_per_second(self):
            elapsed = self._elapsed()
            if elapsed <= 0:
                return 0.0
            return float(self.num_timesteps) / elapsed

        def _save_checkpoint(self, checkpoint_name):
            checkpoint_path = checkpoints_dir / checkpoint_name
            self.model.save(str(checkpoint_path))
            metadata = _checkpoint_metadata(
                config=config,
                checkpoint_path=checkpoint_path,
                timesteps=self.num_timesteps,
                elapsed_seconds=self._elapsed(),
                steps_per_second=self._steps_per_second(),
                training_env=self.training_env,
            )
            write_json(checkpoint_path.with_suffix(".json"), metadata)
            self.checkpoint_records.append(metadata)

        def _on_training_start(self):
            self.start_time = time.monotonic()

        def _on_step(self):
            for info in self.locals.get("infos", ()):
                if "episode_return" not in info:
                    continue
                row = {
                    "timesteps": int(self.num_timesteps),
                    "elapsed_seconds": round(self._elapsed(), 6),
                    "steps_per_second": round(self._steps_per_second(), 6),
                    "baseline": config.baseline,
                    "variant": config.variant,
                    "run_seed": int(config.seed),
                    "split": info.get("split", "train"),
                    "layout_seed": int(info.get("layout_seed", -1)),
                    "episode_return": float(info["episode_return"]),
                    "episode_penalized_return": float(
                        info.get("episode_penalized_return", info["episode_return"])
                    ),
                    "episode_cost": float(info["episode_cost"]),
                    "goals_achieved": int(info.get("goals_achieved", 0)),
                    "episode_length": int(info.get("episode_length", 0)),
                    "penalty_coeff": info.get("penalty_coeff", config.penalty_coeff),
                    "budget": info.get("budget", config.budget),
                    "lagrangian_lambda": info.get("lagrangian_lambda"),
                    "lagrangian_lambda_before_update": info.get(
                        "lagrangian_lambda_before_update"
                    ),
                    "lagrangian_lambda_after_update": info.get(
                        "lagrangian_lambda_after_update"
                    ),
                }
                _write_train_metric_row(metrics_path, row)

            if self.num_timesteps - self.last_checkpoint_timestep >= config.save_freq:
                self.last_checkpoint_timestep = int(self.num_timesteps)
                self._save_checkpoint(
                    "checkpoint_{}.zip".format(int(self.num_timesteps))
                )
            return True

    model = PPO(
        env=env,
        seed=config.seed,
        device="auto",
        **resolved_ppo_kwargs,
    )
    callback = EpisodeMetricsCallback()
    model.learn(total_timesteps=config.total_timesteps, callback=callback)

    final_checkpoint_path = run_dir / "final_model.zip"
    model.save(str(final_checkpoint_path))
    final_metadata = _checkpoint_metadata(
        config=config,
        checkpoint_path=final_checkpoint_path,
        timesteps=model.num_timesteps,
        elapsed_seconds=callback._elapsed(),
        steps_per_second=callback._steps_per_second(),
        training_env=model.get_env(),
    )
    write_json(run_dir / "final_model.json", final_metadata)

    evaluation_manifest = {
        "baseline": config.baseline,
        "variant": config.variant,
        "run_seed": int(config.seed),
        "run_dir": str(run_dir),
        "train_metrics_csv": str(metrics_path),
        "supported_splits": ["train", "test"],
        "metrics": [
            "episode_return",
            "episode_cost",
            "goals_achieved",
            "episode_length",
        ],
        "checkpoints": callback.checkpoint_records + [final_metadata],
        "example_commands": {
            "train": "python scripts/evaluate_baseline.py --run-dir {} --checkpoint final_model.zip --split train".format(
                run_dir
            ),
            "test": "python scripts/evaluate_baseline.py --run-dir {} --checkpoint final_model.zip --split test".format(
                run_dir
            ),
        },
    }
    write_json(run_dir / "evaluation_manifest.json", evaluation_manifest)

    env.close()
    return {
        "run_dir": str(run_dir),
        "train_metrics_csv": str(metrics_path),
        "final_checkpoint": str(final_checkpoint_path),
        "evaluation_manifest": str(run_dir / "evaluation_manifest.json"),
    }
