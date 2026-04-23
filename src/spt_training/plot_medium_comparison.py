"""Compare reward-penalty and Lagrangian results on one task variant."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

from spt_training.common import DEFAULT_BUDGETS, ensure_directory, float_token
from spt_training import plot_lagrangian as lag_plot
from spt_training import plot_reward_penalty as rp_plot


METHOD_COLORS = {
    "reward_penalty": "#1971c2",
    "lagrangian": "#e8590c",
}
METHOD_LABELS = {
    "reward_penalty": "Reward penalty",
    "lagrangian": "Lagrangian",
}


def build_medium_comparison_parser():
    """Build the CLI parser for medium-task comparison plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--output-dir",
        default="results/figures/medium_comparison",
    )
    parser.add_argument(
        "--variant",
        choices=("easy", "medium", "hard"),
        default="medium",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=list(DEFAULT_BUDGETS),
    )
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--skip-verification", action="store_true")
    return parser


def generate_medium_comparison_plots(
    results_root,
    output_dir,
    variant="medium",
    budgets=DEFAULT_BUDGETS,
    rolling_window=10,
    success_threshold=0.8,
    verify=True,
):
    """Generate comparison figures for reward-penalty and Lagrangian."""
    output_dir = ensure_directory(output_dir)
    budgets = tuple(float(budget) for budget in budgets)

    rp_rows = _reward_penalty_budget_results(results_root, budgets, verify)
    lag_rows = _lagrangian_budget_results(results_root, verify)
    budget_rows = [
        row
        for row in rp_rows + lag_rows
        if row["variant"] == variant and float(row["budget"]) in budgets
    ]
    learning_rows = _learning_speed_rows(
        results_root=Path(results_root),
        variant=variant,
        budgets=budgets,
        reward_penalty_rows=budget_rows,
        rolling_window=rolling_window,
        success_threshold=success_threshold,
    )

    outputs = _plot_all(
        output_dir=output_dir,
        variant=variant,
        budgets=budgets,
        budget_rows=budget_rows,
        learning_rows=learning_rows,
        rolling_window=rolling_window,
        success_threshold=success_threshold,
    )
    return outputs


def _reward_penalty_budget_results(results_root, budgets, verify):
    summaries = rp_plot._read_reward_penalty_summaries(results_root)
    episode_cache = {}
    if verify:
        rp_plot._verify_eval_summaries(summaries, episode_cache)
    success_by_summary = rp_plot._compute_success_rates(summaries, episode_cache)
    average_lambda_matches = rp_plot._match_average_train_budgets(summaries, budgets)
    rows = rp_plot._average_lambda_budget_results(
        summaries,
        success_by_summary=success_by_summary,
        average_lambda_matches=average_lambda_matches,
    )
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "method": "reward_penalty",
                "variant": row["variant"],
                "budget": float(row["target_budget"]),
                "selected_parameter": float(row["matched_penalty_coeff"]),
                "train_return": float(row["train_mean_episode_return"]),
                "train_return_std": float(row["train_std_episode_return_across_seeds"]),
                "test_return": float(row["test_mean_episode_return"]),
                "test_return_std": float(row["test_std_episode_return_across_seeds"]),
                "train_cost": float(row["train_mean_episode_cost"]),
                "train_cost_std": float(row["train_std_episode_cost_across_seeds"]),
                "test_cost": float(row["test_mean_episode_cost"]),
                "test_cost_std": float(row["test_std_episode_cost_across_seeds"]),
                "train_success": float(row["train_success_rate"]),
                "train_success_std": float(row["train_std_success_rate_across_seeds"]),
                "test_success": float(row["test_success_rate"]),
                "test_success_std": float(row["test_std_success_rate_across_seeds"]),
            }
        )
    return output_rows


def _lagrangian_budget_results(results_root, verify):
    summaries = lag_plot._read_lagrangian_summaries(results_root)
    episode_cache = {}
    if verify:
        lag_plot._verify_eval_summaries(summaries, episode_cache)
    success_by_summary = lag_plot._compute_success_rates(summaries, episode_cache)
    rows = lag_plot._summarize_by_budget(summaries, success_by_summary)
    by_key = {
        (row["variant"], row["split"], float(row["budget"])): row
        for row in rows
    }
    output_rows = []
    for (variant, split, budget), row in sorted(by_key.items()):
        if split != "test":
            continue
        train = by_key[(variant, "train", budget)]
        output_rows.append(
            {
                "method": "lagrangian",
                "variant": variant,
                "budget": float(budget),
                "selected_parameter": float(row["lagrangian_lambda"]),
                "train_return": float(train["mean_episode_return"]),
                "train_return_std": float(train["std_episode_return_across_seeds"]),
                "test_return": float(row["mean_episode_return"]),
                "test_return_std": float(row["std_episode_return_across_seeds"]),
                "train_cost": float(train["mean_episode_cost"]),
                "train_cost_std": float(train["std_episode_cost_across_seeds"]),
                "test_cost": float(row["mean_episode_cost"]),
                "test_cost_std": float(row["std_episode_cost_across_seeds"]),
                "train_success": float(train["success_rate"]),
                "train_success_std": float(train["std_success_rate_across_seeds"]),
                "test_success": float(row["success_rate"]),
                "test_success_std": float(row["std_success_rate_across_seeds"]),
            }
        )
    return output_rows


def _learning_speed_rows(
    results_root,
    variant,
    budgets,
    reward_penalty_rows,
    rolling_window,
    success_threshold,
):
    rows = []
    rp_lambda_by_budget = {
        float(row["budget"]): float(row["selected_parameter"])
        for row in reward_penalty_rows
        if row["method"] == "reward_penalty"
    }
    for method in ("reward_penalty", "lagrangian"):
        for budget in budgets:
            timesteps = []
            totals = []
            for seed in (0, 1, 2):
                if method == "reward_penalty":
                    penalty_coeff = rp_lambda_by_budget[float(budget)]
                    path = (
                        results_root
                        / "reward_penalty"
                        / variant
                        / "seed{}_lambda{}".format(seed, _penalty_token(penalty_coeff))
                        / "train_metrics.csv"
                    )
                else:
                    path = (
                        results_root
                        / "lagrangian"
                        / variant
                        / "seed{}_budget{}".format(seed, float_token(budget))
                        / "train_metrics.csv"
                    )
                metric_rows = _load_train_metrics(path)
                totals.append(metric_rows[-1]["timesteps"])
                timestep = _first_rolling_success_timestep(
                    metric_rows,
                    rolling_window=rolling_window,
                    success_threshold=success_threshold,
                )
                if timestep is not None:
                    timesteps.append(timestep)
            rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "budget": float(budget),
                    "reached": len(timesteps),
                    "runs": 3,
                    "mean_timestep": _mean(timesteps) if timesteps else None,
                    "total_timesteps": max(totals),
                }
            )
    return rows


def _load_train_metrics(path):
    if not path.exists():
        raise FileNotFoundError("Missing train_metrics.csv: {}".format(path))
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "timesteps": int(row["timesteps"]),
                    "success": 1 if int(row["goals_achieved"]) > 0 else 0,
                }
            )
    if not rows:
        raise ValueError("No rows in {}".format(path))
    rows.sort(key=lambda row: row["timesteps"])
    return rows


def _first_rolling_success_timestep(metric_rows, rolling_window, success_threshold):
    for index in range(rolling_window - 1, len(metric_rows)):
        window = metric_rows[index - rolling_window + 1 : index + 1]
        if _mean([row["success"] for row in window]) >= float(success_threshold):
            return int(metric_rows[index]["timesteps"])
    return None


def _plot_all(
    output_dir,
    variant,
    budgets,
    budget_rows,
    learning_rows,
    rolling_window,
    success_threshold,
):
    plt = _load_pyplot()
    outputs = {
        "budget_task_return_plot": output_dir
        / "{}_reward_penalty_vs_lagrangian_budget_task_return.png".format(variant),
        "learning_speed_plot": output_dir
        / "{}_reward_penalty_vs_lagrangian_learning_speed.png".format(variant),
        "robustness_plot": output_dir
        / "{}_reward_penalty_vs_lagrangian_robustness.png".format(variant),
    }
    _plot_budget_task_return(plt, outputs["budget_task_return_plot"], variant, budgets, budget_rows)
    _plot_learning_speed(
        plt,
        outputs["learning_speed_plot"],
        variant,
        budgets,
        learning_rows,
        rolling_window,
        success_threshold,
    )
    _plot_robustness(plt, outputs["robustness_plot"], variant, budgets, budget_rows)
    return {key: str(value) for key, value in outputs.items()}


def _load_pyplot():
    cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "spt_matplotlib"
    ensure_directory(cache_root)
    ensure_directory(cache_root / "xdg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install -e \".[analysis]\""
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )
    return plt


def _plot_budget_task_return(plt, output_path, variant, budgets, rows):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for method in ("reward_penalty", "lagrangian"):
        ordered = _ordered_method_rows(rows, method, budgets)
        ax.errorbar(
            budgets,
            [row["test_return"] for row in ordered],
            yerr=[row["test_return_std"] for row in ordered],
            marker="o",
            linewidth=2,
            capsize=3,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Target budget")
    ax.set_ylabel("Held-out test mean episode return")
    ax.set_xticks(list(budgets))
    ax.legend()
    fig.suptitle("{} task: budget vs task return".format(variant.capitalize()))
    fig.text(
        0.5,
        0.01,
        "Reward-penalty lambdas are selected on train cost; both methods report held-out test return.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_learning_speed(
    plt,
    output_path,
    variant,
    budgets,
    rows,
    rolling_window,
    success_threshold,
):
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    methods = ("reward_penalty", "lagrangian")
    width = 0.34
    x_positions = list(range(len(budgets)))
    max_total = max(int(row["total_timesteps"]) for row in rows)
    offsets = {"reward_penalty": -width / 2.0, "lagrangian": width / 2.0}
    for method in methods:
        ordered = _ordered_method_rows(rows, method, budgets)
        heights = [
            row["mean_timestep"] if row["mean_timestep"] is not None else max_total
            for row in ordered
        ]
        bars = ax.bar(
            [x + offsets[method] for x in x_positions],
            heights,
            width=width,
            color=METHOD_COLORS[method],
            alpha=0.85,
            label=METHOD_LABELS[method],
        )
        for bar, row in zip(bars, ordered):
            if row["mean_timestep"] is None:
                bar.set_alpha(0.25)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max_total * 0.025,
                "{}/{}".format(row["reached"], row["runs"]),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.axhline(max_total, color="#868e96", linestyle=":", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_format_float(budget) for budget in budgets])
    ax.set_xlabel("Target budget")
    ax.set_ylabel("First timestep reaching rolling success")
    ax.set_ylim(0, max_total * 1.2)
    ax.legend()
    fig.suptitle(
        "{} task: learning speed, rolling {}-episode success >= {:.0f}%".format(
            variant.capitalize(), rolling_window, 100.0 * float(success_threshold)
        )
    )
    fig.text(
        0.5,
        0.01,
        "Labels show reached runs / 3. Faded bars indicate no run reached the threshold.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_robustness(plt, output_path, variant, budgets, rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    specs = (
        ("Task return", lambda row: row["test_return"] - row["train_return"], "Test - train return"),
        ("Success rate", lambda row: 100.0 * (row["test_success"] - row["train_success"]), "Test - train success (pp)"),
        ("Safety cost", lambda row: row["test_cost"] - row["train_cost"], "Test - train unsafe timesteps"),
    )
    for ax, (title, fn, ylabel) in zip(axes, specs):
        for method in ("reward_penalty", "lagrangian"):
            ordered = _ordered_method_rows(rows, method, budgets)
            ax.plot(
                budgets,
                [fn(row) for row in ordered],
                marker="o",
                linewidth=2,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.axhline(0.0, color="#495057", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Target budget")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(budgets))
    axes[-1].legend()
    fig.suptitle("{} task: train-to-test robustness".format(variant.capitalize()))
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _ordered_method_rows(rows, method, budgets):
    ordered = []
    for budget in budgets:
        matches = [
            row
            for row in rows
            if row["method"] == method and float(row["budget"]) == float(budget)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one row for method={} budget={}, found {}.".format(
                    method, budget, len(matches)
                )
            )
        ordered.append(matches[0])
    return ordered


def _mean(values):
    values = [float(value) for value in values]
    if not values:
        return math.nan
    return sum(values) / len(values)


def _format_float(value):
    return "{:g}".format(float(value))


def _penalty_token(value):
    text = "{:.1f}".format(float(value))
    return text.replace("-", "m").replace(".", "p")
