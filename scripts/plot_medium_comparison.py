"""Generate medium-task comparison figures for reward-penalty vs Lagrangian."""

from spt_training.plot_medium_comparison import (
    build_medium_comparison_parser,
    generate_medium_comparison_plots,
)


def main():
    args = build_medium_comparison_parser().parse_args()
    outputs = generate_medium_comparison_plots(
        results_root=args.results_root,
        output_dir=args.output_dir,
        variant=args.variant,
        budgets=args.budgets,
        rolling_window=args.rolling_window,
        success_threshold=args.success_threshold,
        verify=not args.skip_verification,
    )
    for key, value in outputs.items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
