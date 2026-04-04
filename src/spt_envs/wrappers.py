"""Project-local wrappers for standardizing Safety-Gymnasium env behavior."""

import gymnasium


class FixedLayoutSeedWrapper(gymnasium.Wrapper):
    """Bind an environment instance to one approved layout seed."""

    def __init__(self, env, layout_seed):
        super().__init__(env)
        self.layout_seed = int(layout_seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None and int(seed) != self.layout_seed:
            raise ValueError(
                "This environment instance is bound to layout_seed={!r}. "
                "Received reset(seed={!r}).".format(self.layout_seed, seed)
            )
        return self.env.reset(seed=self.layout_seed, options=options)


class StandardizeSafetyInfoWrapper(gymnasium.Wrapper):
    """Expose stable metadata and episode statistics on every rollout."""

    def __init__(self, env, variant, split, layout_seed):
        super().__init__(env)
        self.variant = variant
        self.split = split
        self.layout_seed = int(layout_seed)
        self._episode_return = 0.0
        self._episode_cost = 0.0
        self._goals_achieved = 0
        self._episode_length = 0

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._episode_return = 0.0
        self._episode_cost = 0.0
        self._goals_achieved = 0
        self._episode_length = 0
        info = dict(info)
        info["variant"] = self.variant
        info["split"] = self.split
        info["layout_seed"] = self.layout_seed
        return observation, info

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        goal_achieved = bool(info.get("goal_achieved", info.get("goal_met", False)))
        cost = float(cost)
        reward = float(reward)

        self._episode_return += reward
        self._episode_cost += cost
        self._goals_achieved += int(goal_achieved)
        self._episode_length += 1

        info["cost"] = cost
        info["goal_achieved"] = goal_achieved
        info["variant"] = self.variant
        info["split"] = self.split
        info["layout_seed"] = self.layout_seed

        if terminated or truncated:
            info["episode_return"] = self._episode_return
            info["episode_cost"] = self._episode_cost
            info["goals_achieved"] = self._goals_achieved
            info["episode_length"] = self._episode_length

        return observation, reward, cost, terminated, truncated, info


class SafetyToGymWrapper(gymnasium.Wrapper):
    """Convert a Safety-Gymnasium step API into a standard Gymnasium one."""

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["cost"] = float(cost)
        return observation, reward, terminated, truncated, info
