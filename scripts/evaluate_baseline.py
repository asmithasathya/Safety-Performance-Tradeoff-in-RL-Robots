"""Evaluate a trained PPO checkpoint on the train or test layout split."""

from spt_training.evaluate import build_evaluate_parser, evaluate_run


def main():
    args = build_evaluate_parser().parse_args()
    outputs = evaluate_run(
        run_dir=args.run_dir,
        checkpoint_name=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        episodes_per_seed=args.episodes_per_seed,
        deterministic=not args.stochastic,
        render_mode=args.render_mode,
        no_save=args.no_save,
    )
    for key, value in outputs.items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
