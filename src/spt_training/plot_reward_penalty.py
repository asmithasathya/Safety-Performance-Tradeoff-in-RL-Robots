"""Plot reward-penalty result summaries from saved experiment CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
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


def build_plot_reward_penalty_parser():
    """Build the CLI parser for reward-penalty plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root containing reward_penalty experiment outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures/reward_penalty",
        help="Directory where figures and derived CSVs should be written.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=list(DEFAULT_BUDGETS),
        help="Target budgets used for post-hoc reward-penalty matching.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip recomputing eval summaries from eval_episodes.csv.",
    )
    return parser


def generate_reward_penalty_plots(
    results_root,
    output_dir,
    budgets=DEFAULT_BUDGETS,
    verify=True,
):
    """Generate reward-penalty figures."""
    output_dir = ensure_directory(output_dir)
    budgets = tuple(float(budget) for budget in budgets)

    summaries = _read_reward_penalty_summaries(results_root)
    if not summaries:
        raise FileNotFoundError(
            "No reward_penalty eval_summary.csv files found under {!r}.".format(
                str(results_root)
            )
        )

    episode_cache = {}
    if verify:
        _verify_eval_summaries(summaries, episode_cache)

    success_by_summary = _compute_success_rates(summaries, episode_cache)
    average_lambda_matches = _match_average_train_budgets(summaries, budgets)
    lambda_summary_rows = _summarize_cost_by_lambda(summaries)
    budget_results = _average_lambda_budget_results(
        summaries,
        success_by_summary=success_by_summary,
        average_lambda_matches=average_lambda_matches,
    )

    figure_paths = _plot_all(
        output_dir=output_dir,
        budget_results=budget_results,
        average_lambda_matches=average_lambda_matches,
        lambda_summary_rows=lambda_summary_rows,
        budgets=budgets,
    )

    return figure_paths


def _read_reward_penalty_summaries(results_root):
    results_root = Path(results_root)
    search_root = results_root / "reward_penalty"
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
        if row.get("baseline") != "reward_penalty":
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
        "checkpoint_timesteps": _int_or_none(row.get("checkpoint_timesteps")),
        "episodes": int(row["episodes"]),
        "deterministic": row.get("deterministic"),
        "penalty_coeff": _float_required(row.get("penalty_coeff"), "penalty_coeff"),
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


def _float_required(value, name):
    if value in (None, ""):
        raise ValueError("{} is required.".format(name))
    return float(value)


def _int_or_none(value):
    if value in (None, ""):
        return None
    return int(float(value))


def _summary_key(summary):
    return (
        summary["variant"],
        int(summary["run_seed"]),
        summary["split"],
        float(summary["penalty_coeff"]),
    )


def _episodes_path(summary):
    return summary["summary_path"].parent / "eval_episodes.csv"


def _load_eval_episodes(summary, episode_cache):
    path = _episodes_path(summary)
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


def _match_train_budgets_to_test(summaries, success_by_summary, budgets):
    by_key = {_summary_key(summary): summary for summary in summaries}
    train_by_run = defaultdict(list)
    for summary in summaries:
        if summary["split"] == "train":
            train_by_run[(summary["variant"], int(summary["run_seed"]))].append(summary)

    matches = []
    for (variant, run_seed), candidates in sorted(
        train_by_run.items(), key=lambda item: (_variant_rank(item[0][0]), item[0][1])
    ):
        for target_budget in budgets:
            best_train = min(
                candidates,
                key=lambda row: (
                    abs(float(row["mean_episode_cost"]) - float(target_budget)),
                    -float(row["mean_episode_return"]),
                ),
            )
            test_key = (
                variant,
                run_seed,
                "test",
                float(best_train["penalty_coeff"]),
            )
            if test_key not in by_key:
                raise FileNotFoundError(
                    "Missing test eval summary for variant={} seed={} lambda={}.".format(
                        variant, run_seed, best_train["penalty_coeff"]
                    )
                )
            test_summary = by_key[test_key]
            train_key = _summary_key(best_train)

            matches.append(
                {
                    "variant": variant,
                    "run_seed": run_seed,
                    "target_budget": float(target_budget),
                    "matched_penalty_coeff": float(best_train["penalty_coeff"]),
                    "checkpoint_timesteps": best_train["checkpoint_timesteps"],
                    "train_mean_episode_cost": float(best_train["mean_episode_cost"]),
                    "train_std_episode_cost": float(best_train["std_episode_cost"]),
                    "train_mean_episode_return": float(best_train["mean_episode_return"]),
                    "train_std_episode_return": float(best_train["std_episode_return"]),
                    "train_success_rate": float(success_by_summary[train_key]),
                    "train_mean_goals_achieved": float(best_train["mean_goals_achieved"]),
                    "test_mean_episode_cost": float(test_summary["mean_episode_cost"]),
                    "test_std_episode_cost": float(test_summary["std_episode_cost"]),
                    "test_mean_episode_return": float(test_summary["mean_episode_return"]),
                    "test_std_episode_return": float(test_summary["std_episode_return"]),
                    "test_success_rate": float(success_by_summary[test_key]),
                    "test_mean_goals_achieved": float(test_summary["mean_goals_achieved"]),
                    "train_cost_gap": abs(
                        float(best_train["mean_episode_cost"]) - float(target_budget)
                    ),
                    "test_budget_excess": max(
                        0.0,
                        float(test_summary["mean_episode_cost"]) - float(target_budget),
                    ),
                    "train_summary_path": str(best_train["summary_path"]),
                    "test_summary_path": str(test_summary["summary_path"]),
                }
            )
    return matches


def _match_average_train_budgets(summaries, budgets):
    """Choose one lambda per variant/budget after averaging train cost across seeds."""
    grouped = defaultdict(list)
    for summary in summaries:
        if summary["split"] != "train":
            continue
        grouped[(summary["variant"], float(summary["penalty_coeff"]))].append(summary)

    candidates_by_variant = defaultdict(list)
    for (variant, penalty_coeff), rows in grouped.items():
        candidates_by_variant[variant].append(
            {
                "variant": variant,
                "matched_penalty_coeff": float(penalty_coeff),
                "runs": len(rows),
                "run_seeds": ";".join(
                    str(seed) for seed in sorted({int(row["run_seed"]) for row in rows})
                ),
                "train_mean_episode_cost": _mean(
                    [float(row["mean_episode_cost"]) for row in rows]
                ),
                "train_std_episode_cost_across_seeds": _std(
                    [float(row["mean_episode_cost"]) for row in rows]
                ),
                "train_mean_episode_return": _mean(
                    [float(row["mean_episode_return"]) for row in rows]
                ),
            }
        )

    matches = []
    for variant in sorted(candidates_by_variant, key=_variant_rank):
        candidates = candidates_by_variant[variant]
        for target_budget in budgets:
            best_row = min(
                candidates,
                key=lambda row: (
                    abs(float(row["train_mean_episode_cost"]) - float(target_budget)),
                    -float(row["train_mean_episode_return"]),
                ),
            )
            matches.append(
                {
                    **best_row,
                    "target_budget": float(target_budget),
                    "train_cost_gap": abs(
                        float(best_row["train_mean_episode_cost"])
                        - float(target_budget)
                    ),
                }
            )
    return matches


def _summarize_cost_by_lambda(summaries):
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[
            (
                summary["variant"],
                summary["split"],
                float(summary["penalty_coeff"]),
            )
        ].append(summary)

    rows = []
    for (variant, split, penalty_coeff), group_rows in sorted(
        grouped.items(),
        key=lambda item: (_variant_rank(item[0][0]), item[0][1], item[0][2]),
    ):
        costs = [float(row["mean_episode_cost"]) for row in group_rows]
        returns = [float(row["mean_episode_return"]) for row in group_rows]
        rows.append(
            {
                "variant": variant,
                "split": split,
                "penalty_coeff": float(penalty_coeff),
                "runs": len(group_rows),
                "run_seeds": ";".join(
                    str(seed)
                    for seed in sorted({int(row["run_seed"]) for row in group_rows})
                ),
                "mean_episode_cost": _mean(costs),
                "std_episode_cost_across_seeds": _std(costs),
                "mean_episode_return": _mean(returns),
                "std_episode_return_across_seeds": _std(returns),
            }
        )
    return rows


def _average_lambda_budget_results(
    summaries,
    success_by_summary,
    average_lambda_matches,
):
    by_key = defaultdict(list)
    for summary in summaries:
        by_key[
            (
                summary["variant"],
                summary["split"],
                float(summary["penalty_coeff"]),
            )
        ].append(summary)

    rows = []
    for match in average_lambda_matches:
        variant = match["variant"]
        target_budget = float(match["target_budget"])
        penalty_coeff = float(match["matched_penalty_coeff"])
        output_row = {
            "variant": variant,
            "target_budget": target_budget,
            "matched_penalty_coeff": penalty_coeff,
            "train_matched_mean_episode_cost": float(match["train_mean_episode_cost"]),
            "train_matched_cost_gap": float(match["train_cost_gap"]),
        }

        for split in ("train", "test"):
            split_summaries = by_key[(variant, split, penalty_coeff)]
            if not split_summaries:
                raise FileNotFoundError(
                    "Missing {} summaries for variant={} lambda={}.".format(
                        split, variant, _format_float(penalty_coeff)
                    )
                )
            costs = [float(row["mean_episode_cost"]) for row in split_summaries]
            returns = [float(row["mean_episode_return"]) for row in split_summaries]
            successes = [
                float(success_by_summary[_summary_key(row)])
                for row in split_summaries
            ]
            output_row["{}_mean_episode_cost".format(split)] = _mean(costs)
            output_row[
                "{}_std_episode_cost_across_seeds".format(split)
            ] = _std(costs)
            output_row["{}_mean_episode_return".format(split)] = _mean(returns)
            output_row[
                "{}_std_episode_return_across_seeds".format(split)
            ] = _std(returns)
            output_row["{}_success_rate".format(split)] = _mean(successes)
            output_row[
                "{}_std_success_rate_across_seeds".format(split)
            ] = _std(successes)
        rows.append(output_row)
    return rows


def _compute_learning_speed_rows(matches, rolling_window, success_threshold):
    rows = []
    for match in matches:
        train_summary_path = Path(match["train_summary_path"])
        train_metrics_path = _find_run_dir(train_summary_path) / "train_metrics.csv"
        metric_rows = _load_train_metrics(train_metrics_path)
        threshold_timestep = _first_rolling_success_timestep(
            metric_rows,
            rolling_window=rolling_window,
            success_threshold=success_threshold,
        )
        total_timesteps = metric_rows[-1]["timesteps"] if metric_rows else None
        rows.append(
            {
                "variant": match["variant"],
                "run_seed": match["run_seed"],
                "target_budget": match["target_budget"],
                "matched_penalty_coeff": match["matched_penalty_coeff"],
                "rolling_window": rolling_window,
                "success_threshold": success_threshold,
                "reached_threshold": threshold_timestep is not None,
                "timestep_to_threshold": threshold_timestep,
                "total_logged_timesteps": total_timesteps,
                "train_metrics_path": str(train_metrics_path),
            }
        )
    return rows


def _find_run_dir(path):
    path = Path(path)
    for parent in path.parents:
        if (parent / "train_metrics.csv").exists():
            return parent
    raise FileNotFoundError("Could not find train_metrics.csv above {}".format(path))


def _load_train_metrics(path):
    if not path.exists():
        raise FileNotFoundError("Missing training metrics file: {}".format(path))
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "timesteps": int(row["timesteps"]),
                    "goals_achieved": int(row["goals_achieved"]),
                }
            )
    rows.sort(key=lambda row: row["timesteps"])
    if not rows:
        raise ValueError("No training metrics found in {}".format(path))
    return rows


def _first_rolling_success_timestep(metric_rows, rolling_window, success_threshold):
    successes = [
        1.0 if int(row["goals_achieved"]) > 0 else 0.0 for row in metric_rows
    ]
    for index in range(rolling_window - 1, len(metric_rows)):
        window_values = successes[index - rolling_window + 1 : index + 1]
        if _mean(window_values) >= float(success_threshold):
            return int(metric_rows[index]["timesteps"])
    return None


def _summarize_budget_matches(matches):
    grouped = defaultdict(list)
    for row in matches:
        grouped[(row["variant"], float(row["target_budget"]))].append(row)

    summary_rows = []
    for (variant, target_budget), rows in sorted(
        grouped.items(), key=lambda item: (_variant_rank(item[0][0]), item[0][1])
    ):
        lambda_counts = Counter(
            float(row["matched_penalty_coeff"]) for row in rows
        )
        lambda_counts_text = ";".join(
            "{}x{}".format(_format_float(value), count)
            for value, count in sorted(lambda_counts.items())
        )
        out = {
            "variant": variant,
            "target_budget": float(target_budget),
            "runs": len(rows),
            "matched_penalty_coeff_counts": lambda_counts_text,
        }
        for key in (
            "train_mean_episode_cost",
            "train_mean_episode_return",
            "train_success_rate",
            "test_mean_episode_cost",
            "test_mean_episode_return",
            "test_success_rate",
            "train_cost_gap",
            "test_budget_excess",
        ):
            values = [float(row[key]) for row in rows]
            out["mean_" + key] = _mean(values)
            out["std_" + key] = _std(values)
        summary_rows.append(out)
    return summary_rows


def _plot_all(
    output_dir,
    budget_results,
    average_lambda_matches,
    lambda_summary_rows,
    budgets,
):
    plt = _load_pyplot(output_dir)
    figures = {
        "test_task_return_by_budget_plot": output_dir
        / "reward_penalty_test_task_return_by_budget.png",
        "test_success_rate_by_budget_plot": output_dir
        / "reward_penalty_test_success_rate_by_budget.png",
        "train_lambda_matching_by_budget_plot": output_dir
        / "reward_penalty_train_lambda_matching_by_budget.png",
        "test_mistakes_by_lambda_plot": output_dir
        / "reward_penalty_test_mistakes_by_lambda.png",
    }
    _plot_task_return_success(
        plt,
        figures["test_task_return_by_budget_plot"],
        budget_results,
        budgets,
    )
    _plot_success_rate(
        plt,
        figures["test_success_rate_by_budget_plot"],
        budget_results,
        budgets,
    )
    _plot_train_cost_lambda(
        plt, figures["train_lambda_matching_by_budget_plot"], average_lambda_matches, budgets
    )
    _plot_mistakes_by_lambda(
        plt,
        figures["test_mistakes_by_lambda_plot"],
        lambda_summary_rows,
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
    except ImportError as exc:  # pragma: no cover - depends on local environment
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


def _plot_task_return_success(plt, output_path, matches, budgets):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for variant in _ordered_variants(matches):
        rows = _ordered_budget_rows(matches, variant, budgets)
        return_means = [float(row["test_mean_episode_return"]) for row in rows]
        return_stds = [
            float(row["test_std_episode_return_across_seeds"]) for row in rows
        ]
        color = VARIANT_COLORS.get(variant)
        marker = VARIANT_MARKERS.get(variant, "o")
        ax.errorbar(
            budgets,
            return_means,
            yerr=return_stds,
            marker=marker,
            color=color,
            capsize=3,
            linewidth=2,
            label=variant,
        )

    ax.set_xlabel("Target budget")
    ax.set_ylabel("Mean episode return")
    ax.set_xticks(list(budgets))
    ax.legend(title="Variant")
    fig.suptitle("Reward-penalty task return by train-matched budget")
    fig.text(
        0.5,
        0.01,
        "Lambdas are chosen by averaged train cost; points show held-out test return.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_success_rate(plt, output_path, matches, budgets):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for variant in _ordered_variants(matches):
        rows = _ordered_budget_rows(matches, variant, budgets)
        success_means = [
            100.0 * float(row["test_success_rate"]) for row in rows
        ]
        success_stds = [
            100.0 * float(row["test_std_success_rate_across_seeds"])
            for row in rows
        ]
        ax.errorbar(
            budgets,
            success_means,
            yerr=success_stds,
            marker=VARIANT_MARKERS.get(variant, "o"),
            color=VARIANT_COLORS.get(variant),
            capsize=3,
            linewidth=2,
            label=variant,
        )

    ax.set_xlabel("Target budget")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(list(budgets))
    ax.set_ylim(0, 105)
    ax.legend(title="Variant")
    fig.suptitle("Reward-penalty success rate by train-matched budget")
    fig.text(
        0.5,
        0.01,
        "Success is the percentage of held-out test episodes with goals_achieved > 0.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_constraint_violations(plt, output_path, matches, budgets):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    split_specs = (
        ("train", "train_mean_episode_cost", "Train split"),
        ("test", "test_mean_episode_cost", "Held-out test split"),
    )
    max_cost = max(
        max(float(row["train_mean_episode_cost"]), float(row["test_mean_episode_cost"]))
        for row in matches
    )
    for ax, (_, metric_key, title) in zip(axes, split_specs):
        for variant in _ordered_variants(matches):
            rows = _rows_for_variant_budget(matches, variant, budgets)
            means = [_mean([row[metric_key] for row in group]) for group in rows]
            stds = [_std([row[metric_key] for row in group]) for group in rows]
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
        ax.set_title(title)
        ax.set_xlabel("Target budget")
        ax.set_xticks(list(budgets))
    axes[0].set_ylabel("Mean unsafe timesteps per episode")
    axes[0].set_ylim(0, max(max_cost, max(budgets)) * 1.12)
    axes[1].legend(title="Variant")
    fig.suptitle("Constraint violations under reward-penalty budget matching")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_train_cost_lambda(plt, output_path, matches, budgets):
    variants = _ordered_variants(matches)
    variant_offsets = _variant_offsets(variants)
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    max_cost = max(float(row["train_mean_episode_cost"]) for row in matches)
    max_budget = max(float(budget) for budget in budgets)
    y_top = max(max_cost, max_budget) * 1.22

    for variant in variants:
        variant_rows = sorted(
            [row for row in matches if row["variant"] == variant],
            key=lambda row: float(row["target_budget"]),
        )
        x_values = [
            float(row["target_budget"]) + variant_offsets[variant]
            for row in variant_rows
        ]
        y_values = [float(row["train_mean_episode_cost"]) for row in variant_rows]
        ax.scatter(
            x_values,
            y_values,
            s=72,
            color=VARIANT_COLORS.get(variant),
            edgecolor="#212529",
            linewidth=0.8,
            alpha=0.9,
            label=variant,
        )
        for x_value, y_value, row in zip(x_values, y_values, variant_rows):
            lambda_label = _format_float(row["matched_penalty_coeff"])
            ax.annotate(
                r"$\lambda={}$".format(lambda_label),
                xy=(x_value, y_value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=18,
            )

    ax.set_xlabel("Target budget")
    ax.set_ylabel("Train mean unsafe timesteps per episode")
    ax.set_xticks(list(budgets))
    ax.set_ylim(0, y_top)
    ax.legend(title="Variant", loc="upper left")
    fig.suptitle("Reward-penalty chosen lambda by target budget")
    fig.text(
        0.5,
        0.01,
        "Each dot averages train mean cost across seeds before choosing the lambda closest to the budget.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_mistakes_by_lambda(plt, output_path, lambda_summary_rows):
    plot_rows = [row for row in lambda_summary_rows if row["split"] == "test"]
    lambdas = sorted({float(row["penalty_coeff"]) for row in plot_rows})
    x_positions = list(range(len(lambdas)))
    lambda_to_x = {value: index for index, value in enumerate(lambdas)}

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for variant in _ordered_variants(plot_rows):
        variant_rows = sorted(
            [row for row in plot_rows if row["variant"] == variant],
            key=lambda row: float(row["penalty_coeff"]),
        )
        x_values = [lambda_to_x[float(row["penalty_coeff"])] for row in variant_rows]
        means = [float(row["mean_episode_cost"]) for row in variant_rows]
        stds = [float(row["std_episode_cost_across_seeds"]) for row in variant_rows]
        ax.errorbar(
            x_values,
            means,
            yerr=stds,
            marker=VARIANT_MARKERS.get(variant, "o"),
            color=VARIANT_COLORS.get(variant),
            capsize=3,
            linewidth=2,
            label=variant,
        )

    ax.set_xlabel("Penalty coefficient lambda")
    ax.set_ylabel("Mean mistakes per episode")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_format_float(value) for value in lambdas])
    ax.legend(title="Variant")
    fig.suptitle("Reward-penalty mistakes by lambda")
    fig.text(
        0.5,
        0.01,
        "Mistakes are mean unsafe timesteps per held-out test episode, averaged across training seeds.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_learning_speed(
    plt,
    output_path,
    learning_rows,
    budgets,
    rolling_window,
    success_threshold,
):
    variants = _ordered_variants(learning_rows)
    fig, axes = plt.subplots(1, len(variants), figsize=(4.6 * len(variants), 4.8), sharey=True)
    if len(variants) == 1:
        axes = [axes]
    max_timestep = max(
        int(row["total_logged_timesteps"])
        for row in learning_rows
        if row["total_logged_timesteps"] is not None
    )

    for ax, variant in zip(axes, variants):
        heights = []
        labels = []
        colors = []
        alphas = []
        for budget in budgets:
            rows = [
                row
                for row in learning_rows
                if row["variant"] == variant and float(row["target_budget"]) == float(budget)
            ]
            reached = [
                int(row["timestep_to_threshold"])
                for row in rows
                if row["timestep_to_threshold"] not in (None, "")
            ]
            if reached:
                heights.append(_mean(reached))
                colors.append(VARIANT_COLORS.get(variant))
                alphas.append(0.9)
            else:
                heights.append(max_timestep)
                colors.append("#adb5bd")
                alphas.append(0.35)
            labels.append("{}/{}".format(len(reached), len(rows)))

        bars = ax.bar([str(_format_float(budget)) for budget in budgets], heights, color=colors)
        for bar, label, alpha in zip(bars, labels, alphas):
            bar.set_alpha(alpha)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max_timestep * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.axhline(max_timestep, color="#868e96", linestyle=":", linewidth=1)
        ax.set_title(variant)
        ax.set_xlabel("Target budget")
        ax.set_ylim(0, max_timestep * 1.2)
    axes[0].set_ylabel("First timestep reaching rolling success")
    fig.suptitle(
        "Learning speed: rolling {}-episode success >= {:.0f}%".format(
            rolling_window, 100.0 * float(success_threshold)
        )
    )
    fig.text(
        0.5,
        0.01,
        "Bar labels show reached runs / total runs. Gray bars indicate no reached runs.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_robustness(plt, output_path, matches, budgets):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    specs = (
        (
            "Task return",
            lambda row: float(row["test_mean_episode_return"])
            - float(row["train_mean_episode_return"]),
            "Test - train return",
        ),
        (
            "Constraint violations",
            lambda row: float(row["test_mean_episode_cost"])
            - float(row["train_mean_episode_cost"]),
            "Test - train unsafe timesteps",
        ),
        (
            "Success rate",
            lambda row: 100.0
            * (float(row["test_success_rate"]) - float(row["train_success_rate"])),
            "Test - train success (pp)",
        ),
    )
    for ax, (title, metric_fn, ylabel) in zip(axes, specs):
        for variant in _ordered_variants(matches):
            groups = _rows_for_variant_budget(matches, variant, budgets)
            means = [_mean([metric_fn(row) for row in group]) for group in groups]
            stds = [_std([metric_fn(row) for row in group]) for group in groups]
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
        ax.axhline(0.0, color="#495057", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Target budget")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(budgets))
    axes[-1].legend(title="Variant")
    fig.suptitle("Reward-penalty robustness from train layouts to held-out test layouts")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _rows_for_variant_budget(rows, variant, budgets):
    grouped = []
    for budget in budgets:
        budget_rows = [
            row
            for row in rows
            if row["variant"] == variant and float(row["target_budget"]) == float(budget)
        ]
        if not budget_rows:
            raise ValueError(
                "No rows found for variant={} budget={}.".format(variant, budget)
            )
        grouped.append(budget_rows)
    return grouped


def _ordered_budget_rows(rows, variant, budgets):
    ordered = []
    for budget in budgets:
        budget_rows = [
            row
            for row in rows
            if row["variant"] == variant and float(row["target_budget"]) == float(budget)
        ]
        if len(budget_rows) != 1:
            raise ValueError(
                "Expected one row for variant={} budget={}, found {}.".format(
                    variant, budget, len(budget_rows)
                )
            )
        ordered.append(budget_rows[0])
    return ordered


def _ordered_variants(rows):
    found = {row["variant"] for row in rows}
    return sorted(found, key=_variant_rank)


def _variant_offsets(variants):
    if len(variants) == 1:
        return {variants[0]: 0.0}
    span = 1.8
    step = span / float(len(variants) - 1)
    return {
        variant: -span / 2.0 + index * step
        for index, variant in enumerate(variants)
    }


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


def _format_float(value):
    return "{:g}".format(float(value))


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _match_fieldnames():
    return (
        "variant",
        "run_seed",
        "target_budget",
        "matched_penalty_coeff",
        "checkpoint_timesteps",
        "train_mean_episode_cost",
        "train_std_episode_cost",
        "train_mean_episode_return",
        "train_std_episode_return",
        "train_success_rate",
        "train_mean_goals_achieved",
        "test_mean_episode_cost",
        "test_std_episode_cost",
        "test_mean_episode_return",
        "test_std_episode_return",
        "test_success_rate",
        "test_mean_goals_achieved",
        "train_cost_gap",
        "test_budget_excess",
        "train_summary_path",
        "test_summary_path",
    )


def _average_lambda_match_fieldnames():
    return (
        "variant",
        "target_budget",
        "matched_penalty_coeff",
        "runs",
        "run_seeds",
        "train_mean_episode_cost",
        "train_std_episode_cost_across_seeds",
        "train_mean_episode_return",
        "train_cost_gap",
    )


def _lambda_summary_fieldnames():
    return (
        "variant",
        "split",
        "penalty_coeff",
        "runs",
        "run_seeds",
        "mean_episode_cost",
        "std_episode_cost_across_seeds",
        "mean_episode_return",
        "std_episode_return_across_seeds",
    )


def _budget_summary_fieldnames():
    value_keys = (
        "train_mean_episode_cost",
        "train_mean_episode_return",
        "train_success_rate",
        "test_mean_episode_cost",
        "test_mean_episode_return",
        "test_success_rate",
        "train_cost_gap",
        "test_budget_excess",
    )
    fieldnames = [
        "variant",
        "target_budget",
        "runs",
        "matched_penalty_coeff_counts",
    ]
    for key in value_keys:
        fieldnames.append("mean_" + key)
        fieldnames.append("std_" + key)
    return tuple(fieldnames)


def _learning_fieldnames():
    return (
        "variant",
        "run_seed",
        "target_budget",
        "matched_penalty_coeff",
        "rolling_window",
        "success_threshold",
        "reached_threshold",
        "timestep_to_threshold",
        "total_logged_timesteps",
        "train_metrics_path",
    )
