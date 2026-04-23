"""Generate Lagrangian result figures from saved experiment outputs."""

from spt_training.plot_lagrangian import (
    build_plot_lagrangian_parser,
    generate_lagrangian_plots,
)


def main():
    args = build_plot_lagrangian_parser().parse_args()
    outputs = generate_lagrangian_plots(
        results_root=args.results_root,
        output_dir=args.output_dir,
        budgets=args.budgets,
        verify=not args.skip_verification,
    )
    for key, value in outputs.items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
