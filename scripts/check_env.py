"""Smoke-check utility for project-local Safety-Gymnasium environments."""

import argparse

from spt_envs.factory import make_env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("easy", "medium", "hard"), required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--layout-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--api", choices=("gym", "safe"), default="gym")
    parser.add_argument("--render", dest="render_mode", default=None)
    parser.add_argument("--record-trajectory", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_env(
        variant=args.variant,
        split=args.split,
        layout_seed=args.layout_seed,
        api=args.api,
        render_mode=args.render_mode,
        record_trajectory=args.record_trajectory,
    )

    print("env_id:", getattr(getattr(env.unwrapped, "spec", None), "id", "unknown"))
    print("api:", args.api)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)

    for episode_idx in range(args.episodes):
        observation, info = env.reset()
        _ = observation
        episode_reward = 0.0
        episode_cost = 0.0
        done = False

        while not done:
            action = env.action_space.sample()
            if args.api == "safe":
                observation, reward, cost, terminated, truncated, info = env.step(action)
            else:
                observation, reward, terminated, truncated, info = env.step(action)
                cost = info["cost"]

            _ = observation
            episode_reward += float(reward)
            episode_cost += float(cost)
            done = bool(terminated or truncated)

        print(
            "episode={} reward={:.3f} cost={:.3f} goals={} length={}".format(
                episode_idx,
                info.get("episode_return", episode_reward),
                info.get("episode_cost", episode_cost),
                info.get("goals_achieved", "n/a"),
                info.get("episode_length", "n/a"),
            )
        )

    if args.record_trajectory and hasattr(env, "get_wrapper_attr"):
        try:
            completed = env.get_wrapper_attr("completed_trajectories")
        except AttributeError:
            completed = None
        if completed is not None:
            print("recorded_trajectories:", len(completed))

    env.close()


if __name__ == "__main__":
    main()
