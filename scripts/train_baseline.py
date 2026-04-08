"""Train one reward-penalty or Lagrangian PPO baseline run."""

from spt_training.train import build_train_parser, train_run, training_run_config_from_args


def main():
    args = build_train_parser().parse_args()
    config = training_run_config_from_args(args)
    outputs = train_run(config)
    for key, value in outputs.items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
