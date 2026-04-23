"""Plot Lagrangian result summaries from saved experiment CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

from spt_training.common import DEFAULT_BUDGETS, ensure_directory


VARIANT_ORDER = ("easy", "medium", "hard")
VARIANT_COLORS = {
    "easy": "#2f9e44",
    "medium": "#1971c2",
    "hard": "#e8590c",
}
VARIANT_MARKERS = {
    "easy": "o",
    "medium": "s",
    "hard": "^",
}


def build_plot_lagrangian_parser():
    """Build the CLI parser for Lagrangian plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--output-dir",
        default="results/figures/lagrangian",
        help="Directory where figures should be written.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=list(DEFAULT_BUDGETS),
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip recomputing eval summaries from eval_episodes.csv.",
    )
    return parser


def generate_lagrangian_plots(
    results_root,
    output_dir,
    budgets=DEFAULT_BUDGETS,
    verify=True,
):
    """Generate Lagrangian figures."""
    output_dir = ensure_directory(output_dir)
    budgets = tuple(float(budget) for budget in budgets)
    summaries = _read_lagrangian_summaries(results_root)
    if not summaries:
        raise FileNotFoundError(
            "No lagrangian eval_summary.csv files found under {!r}.".format(
                str(results_root)
            )
        )

    episode_cache = {}
    if verify:
        _verify_eval_summaries(summaries, episode_cache)

    success_by_summary = _compute_success_rates(summaries, episode_cache)
    budget_rows = _summarize_by_budget(summaries, success_by_summary)
    outputs = _plot_all(output_dir, budget_rows, budgets)
    return outputs


def _read_lagrangian_summaries(results_root):
    results_root = Path(results_root)
    search_root = results_root / "lagrangian"
    if not search_root.exists():
        search_root = results_root

    summaries = []
    for summary_path in sorted(search_root.rglob("eval_summary.csv")):
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(
                "Expected exactly one row in {!r}, found {}.".format(
                    str(summary_path), len(rows)
                )
            )
        row = dict(rows[0])
        if row.get("baseline") != "lagrangian":
            continue
        summaries.append(_normalize_summary_row(row, summary_path))
    return summaries


def _normalize_summary_row(row, summary_path):
    return {
        "baseline": row["baseline"],
        "variant": row["variant"],
        "run_seed": int(row["run_seed"]),
        "split": row["split"],
        "checkpoint_name": row["checkpoint_name"],
        "checkpoint_timesteps": int(float(row["checkpoint_timesteps"])),
        "episodes": int(row["episodes"]),
        "budget": float(row["budget"]),
        "lagrangian_lambda": float(row["lagrangian_lambda"]),
        "mean_episode_return": float(row["mean_episode_return"]),
        "std_episode_return": float(row["std_episode_return"]),
        "mean_episode_cost": float(row["mean_episode_cost"]),
        "std_episode_cost": float(row["std_episode_cost"]),
        "mean_goals_achieved": float(row["mean_goals_achieved"]),
        "std_goals_achieved": float(row["std_goals_achieved"]),
        "mean_episode_length": float(row["mean_episode_length"]),
        "std_episode_length": float(row["std_episode_length"]),
        "summary_path": Path(summary_path),
    }


def _summary_key(summary):
    return (
        summary["variant"],
        int(summary["run_seed"]),
        summary["split"],
        float(summary["budget"]),
    )


def _load_eval_episodes(summary, episode_cache):
    path = summary["summary_path"].parent / "eval_episodes.csv"
    if path in episode_cache:
        return episode_cache[path]
    if not path.exists():
        raise FileNotFoundError("Missing episode file: {}".format(path))

    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "episode_return": float(row["episode_return"]),
                    "episode_cost": float(row["episode_cost"]),
                    "goals_achieved": int(row["goals_achieved"]),
                    "episode_length": int(row["episode_length"]),
                }
            )
    if not rows:
        raise ValueError("No evaluation episodes found in {}".format(path))
    episode_cache[path] = rows
    return rows


def _verify_eval_summaries(summaries, episode_cache, tolerance=1e-9):
    checks = (
        ("episode_return", "mean_episode_return", "std_episode_return"),
        ("episode_cost", "mean_episode_cost", "std_episode_cost"),
        ("goals_achieved", "mean_goals_achieved", "std_goals_achieved"),
        ("episode_length", "mean_episode_length", "std_episode_length"),
    )
    for summary in summaries:
        episodes = _load_eval_episodes(summary, episode_cache)
        if len(episodes) != int(summary["episodes"]):
            raise ValueError(
                "{} reports {} episodes but {} rows were found.".format(
                    summary["summary_path"], summary["episodes"], len(episodes)
                )
            )
        for episode_key, mean_key, std_key in checks:
            values = [float(row[episode_key]) for row in episodes]
            mean_value = _mean(values)
            std_value = _std(values)
            if abs(mean_value - float(summary[mean_key])) > tolerance:
                raise ValueError(
                    "{} mismatch for {}: summary={} recomputed={}".format(
                        summary["summary_path"],
                        mean_key,
                        summary[mean_key],
                        mean_value,
                    )
                )
            if abs(std_value - float(summary[std_key])) > tolerance:
                raise ValueError(
                    "{} mismatch for {}: summary={} recomputed={}".format(
                        summary["summary_path"],
                        std_key,
                        summary[std_key],
                        std_value,
                    )
                )


def _compute_success_rates(summaries, episode_cache):
    success_by_summary = {}
    for summary in summaries:
        episodes = _load_eval_episodes(summary, episode_cache)
        success_by_summary[_summary_key(summary)] = _mean(
            [1.0 if row["goals_achieved"] > 0 else 0.0 for row in episodes]
        )
    return success_by_summary


def _summarize_by_budget(summaries, success_by_summary):
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[(summary["variant"], summary["split"], float(summary["budget"]))].append(
            summary
        )

    rows = []
    for (variant, split, budget), group_rows in sorted(
        grouped.items(),
        key=lambda item: (_variant_rank(item[0][0]), item[0][1], item[0][2]),
    ):
        rows.append(
            {
                "variant": variant,
                "split": split,
                "budget": float(budget),
                "runs": len(group_rows),
                "mean_episode_return": _mean(
                    [row["mean_episode_return"] for row in group_rows]
                ),
                "std_episode_return_across_seeds": _std(
                    [row["mean_episode_return"] for row in group_rows]
                ),
                "mean_episode_cost": _mean(
                    [row["mean_episode_cost"] for row in group_rows]
                ),
                "std_episode_cost_across_seeds": _std(
                    [row["mean_episode_cost"] for row in group_rows]
                ),
                "success_rate": _mean(
                    [success_by_summary[_summary_key(row)] for row in group_rows]
                ),
                "std_success_rate_across_seeds": _std(
                    [success_by_summary[_summary_key(row)] for row in group_rows]
                ),
                "lagrangian_lambda": _mean(
                    [row["lagrangian_lambda"] for row in group_rows]
                ),
                "std_lagrangian_lambda_across_seeds": _std(
                    [row["lagrangian_lambda"] for row in group_rows]
                ),
            }
        )
    return rows


def _plot_all(output_dir, rows, budgets):
    plt = _load_pyplot(output_dir)
    figures = {
        "test_task_return_by_budget_plot": output_dir
        / "lagrangian_test_task_return_by_budget.png",
        "test_success_rate_by_budget_plot": output_dir
        / "lagrangian_test_success_rate_by_budget.png",
        "test_constraint_violations_by_budget_plot": output_dir
        / "lagrangian_test_constraint_violations_by_budget.png",
        "learned_lambda_by_budget_plot": output_dir
        / "lagrangian_learned_lambda_by_budget.png",
    }
    _plot_metric_by_budget(
        plt,
        figures["test_task_return_by_budget_plot"],
        rows,
        budgets,
        split="test",
        metric_key="mean_episode_return",
        std_key="std_episode_return_across_seeds",
        ylabel="Mean episode return",
        title="Lagrangian test task return by budget",
        footer="Budgets are explicit training constraints; points show held-out test return.",
    )
    _plot_metric_by_budget(
        plt,
        figures["test_success_rate_by_budget_plot"],
        rows,
        budgets,
        split="test",
        metric_key="success_rate",
        std_key="std_success_rate_across_seeds",
        ylabel="Success rate (%)",
        title="Lagrangian test success rate by budget",
        scale=100.0,
        ylim=(0, 105),
        footer="Success is the percentage of held-out test episodes with goals_achieved > 0.",
    )
    _plot_constraint_violations(
        plt,
        figures["test_constraint_violations_by_budget_plot"],
        rows,
        budgets,
    )
    _plot_metric_by_budget(
        plt,
        figures["learned_lambda_by_budget_plot"],
        rows,
        budgets,
        split="test",
        metric_key="lagrangian_lambda",
        std_key="std_lagrangian_lambda_across_seeds",
        ylabel="Final learned lambda",
        title="Lagrangian final learned lambda by budget",
        footer="Lambda is learned during training and saved with the final checkpoint.",
    )
    return {key: str(path) for key, path in figures.items()}


def _load_pyplot(output_dir):
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


def _plot_metric_by_budget(
    plt,
    output_path,
    rows,
    budgets,
    split,
    metric_key,
    std_key,
    ylabel,
    title,
    scale=1.0,
    ylim=None,
    footer=None,
):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for variant in _ordered_variants(rows):
        ordered = _ordered_budget_rows(rows, variant, split, budgets)
        means = [scale * float(row[metric_key]) for row in ordered]
        stds = [scale * float(row[std_key]) for row in ordered]
        ax.errorbar(
            budgets,
            means,
            yerr=stds,
            marker=VARIANT_MARKERS.get(variant, "o"),
            color=VARIANT_COLORS.get(variant),
            capsize=3,
            linewidth=2,
            label=variant,
        )
    ax.set_xlabel("Target budget")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(budgets))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(title="Variant")
    fig.suptitle(title)
    if footer:
        fig.text(0.5, 0.01, footer, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    else:
        fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_constraint_violations(plt, output_path, rows, budgets):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    max_cost = max(
        float(row["mean_episode_cost"]) for row in rows if row["split"] == "test"
    )
    for variant in _ordered_variants(rows):
        ordered = _ordered_budget_rows(rows, variant, "test", budgets)
        means = [float(row["mean_episode_cost"]) for row in ordered]
        stds = [float(row["std_episode_cost_across_seeds"]) for row in ordered]
        ax.errorbar(
            budgets,
            means,
            yerr=stds,
            marker=VARIANT_MARKERS.get(variant, "o"),
            color=VARIANT_COLORS.get(variant),
            capsize=3,
            linewidth=2,
            label=variant,
        )
    ax.plot(
        budgets,
        budgets,
        linestyle="--",
        color="#495057",
        linewidth=1.5,
        label="Target budget",
    )
    ax.set_xlabel("Target budget")
    ax.set_ylabel("Mean unsafe timesteps per episode")
    ax.set_xticks(list(budgets))
    ax.set_ylim(0, max(max_cost, max(budgets)) * 1.12)
    ax.legend(title="Variant")
    fig.suptitle("Lagrangian test constraint violations by budget")
    fig.text(
        0.5,
        0.01,
        "Dashed line marks exact budget equality; below it means average test cost is under budget.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _ordered_budget_rows(rows, variant, split, budgets):
    ordered = []
    for budget in budgets:
        matches = [
            row
            for row in rows
            if row["variant"] == variant
            and row["split"] == split
            and float(row["budget"]) == float(budget)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one row for variant={} split={} budget={}, found {}.".format(
                    variant, split, budget, len(matches)
                )
            )
        ordered.append(matches[0])
    return ordered


def _ordered_variants(rows):
    found = {row["variant"] for row in rows}
    return sorted(found, key=_variant_rank)


def _variant_rank(variant):
    try:
        return VARIANT_ORDER.index(variant)
    except ValueError:
        return len(VARIANT_ORDER)


def _mean(values):
    values = [float(value) for value in values]
    if not values:
        return math.nan
    return sum(values) / len(values)


def _std(values):
    values = [float(value) for value in values]
    if not values:
        return math.nan
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
