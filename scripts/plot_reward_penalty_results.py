"""Generate reward-penalty result figures from saved experiment outputs."""

from spt_training.plot_reward_penalty import (
    build_plot_reward_penalty_parser,
    generate_reward_penalty_plots,
)


def main():
    args = build_plot_reward_penalty_parser().parse_args()
    outputs = generate_reward_penalty_plots(
        results_root=args.results_root,
        output_dir=args.output_dir,
        budgets=args.budgets,
        verify=not args.skip_verification,
    )
    for key, value in outputs.items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
