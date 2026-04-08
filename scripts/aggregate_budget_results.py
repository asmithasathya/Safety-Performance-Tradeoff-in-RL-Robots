"""Aggregate evaluation summaries into budget-oriented baseline tables."""

from spt_training.aggregate import aggregate_results, build_aggregate_parser


def main():
    args = build_aggregate_parser().parse_args()
    outputs = aggregate_results(
        results_root=args.results_root,
        split=args.split,
        output_dir=args.output_dir,
        budgets=args.budgets,
    )
    for key, value in outputs.items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
