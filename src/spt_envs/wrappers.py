"""Project-local wrappers for standardizing Safety-Gymnasium env behavior."""

import gymnasium
import numpy as np


class FixedLayoutSeedWrapper(gymnasium.Wrapper):
    """Bind an environment instance to one approved layout seed."""

    def __init__(self, env, layout_seed):
        super().__init__(env)
        self.layout_seed = int(layout_seed)
        self.current_layout_seed = self.layout_seed

    def reset(self, *, seed=None, options=None):
        if seed is not None and int(seed) != self.layout_seed:
            raise ValueError(
                "This environment instance is bound to layout_seed={!r}. "
                "Received reset(seed={!r}).".format(self.layout_seed, seed)
            )
        self.current_layout_seed = self.layout_seed
        return self.env.reset(seed=self.layout_seed, options=options)


class TrainLayoutSeedWrapper(gymnasium.Wrapper):
    """Sample an approved training layout seed on every environment reset."""

    def __init__(self, env, layout_seeds, rng_seed):
        super().__init__(env)
        self.layout_seeds = tuple(int(seed) for seed in layout_seeds)
        if not self.layout_seeds:
            raise ValueError("layout_seeds must contain at least one seed.")
        self._rng = np.random.RandomState(int(rng_seed))
        self.current_layout_seed = int(self.layout_seeds[0])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng.seed(int(seed))
        self.current_layout_seed = int(self._rng.choice(self.layout_seeds))
        return self.env.reset(seed=self.current_layout_seed, options=options)


class StandardizeSafetyInfoWrapper(gymnasium.Wrapper):
    """Expose stable metadata and episode statistics on every rollout."""

    def __init__(self, env, variant, split, layout_seed):
        super().__init__(env)
        self.variant = variant
        self.split = split
        self.layout_seed = None if layout_seed is None else int(layout_seed)
        self._episode_return = 0.0
        self._episode_cost = 0.0
        self._goals_achieved = 0
        self._episode_length = 0

    def _resolve_layout_seed(self):
        current_layout_seed = getattr(self.env, "current_layout_seed", self.layout_seed)
        if current_layout_seed is None:
            return -1
        return int(current_layout_seed)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._episode_return = 0.0
        self._episode_cost = 0.0
        self._goals_achieved = 0
        self._episode_length = 0
        layout_seed = self._resolve_layout_seed()
        info = dict(info)
        info["variant"] = self.variant
        info["split"] = self.split
        info["layout_seed"] = layout_seed
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
        layout_seed = self._resolve_layout_seed()

        info["cost"] = cost
        info["goal_achieved"] = goal_achieved
        info["variant"] = self.variant
        info["split"] = self.split
        info["layout_seed"] = layout_seed

        if terminated or truncated:
            info["episode_return"] = self._episode_return
            info["episode_cost"] = self._episode_cost
            info["goals_achieved"] = self._goals_achieved
            info["episode_length"] = self._episode_length

        return observation, reward, cost, terminated, truncated, info


class RewardPenaltyWrapper(gymnasium.Wrapper):
    """Reward-penalty baseline: r_shaped = r - penalty_coeff * cost.

    Operates on the safe (6-tuple) API.  The unpenalized reward is stored in
    ``info["reward_unpenalized"]`` on every step so callers can distinguish
    task performance from safety penalties.  At episode end, the cumulative
    penalized return is reported as ``info["episode_penalized_return"]``.
    """

    def __init__(self, env, penalty_coeff):
        if penalty_coeff < 0:
            raise ValueError(
                "penalty_coeff must be >= 0, got {!r}".format(penalty_coeff)
            )
        super().__init__(env)
        self.penalty_coeff = float(penalty_coeff)
        self._episode_penalized_return = 0.0

    def reset(self, *, seed=None, options=None):
        self._episode_penalized_return = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        shaped_reward = float(reward) - self.penalty_coeff * float(cost)
        self._episode_penalized_return += shaped_reward
        info["reward_unpenalized"] = float(reward)
        info["penalty_coeff"] = self.penalty_coeff
        if terminated or truncated:
            info["episode_penalized_return"] = self._episode_penalized_return
        return observation, shaped_reward, cost, terminated, truncated, info


class LagrangianWrapper(gymnasium.Wrapper):
    """Lagrangian constrained RL baseline.

    Maintains a dual variable λ ≥ 0 that is updated at the end of every
    episode via a dual gradient ascent step:

        λ ← max(0, λ + lr_lambda * (episode_cost − budget))

    During each rollout the reward is shaped with the *current* λ:

        r_shaped = r − λ * cost

    This drives λ up when the cost budget is exceeded and down when the policy
    is safely under budget, automatically finding the Lagrange multiplier that
    enforces the constraint at the boundary.

    Operates on the safe (6-tuple) API.  Per-step info keys added:
        ``reward_unpenalized``   – raw task reward before penalty
        ``lagrangian_lambda``    – λ value used for this step
        ``budget``               – the cost budget B

    Episode-end info keys added:
        ``episode_penalized_return`` – cumulative shaped return for this episode
        ``lagrangian_lambda``        – λ *after* the dual update (updated value)
    """

    def __init__(self, env, budget, lr_lambda, init_lambda=0.0):
        if budget < 0:
            raise ValueError("budget must be >= 0, got {!r}".format(budget))
        if lr_lambda <= 0:
            raise ValueError("lr_lambda must be > 0, got {!r}".format(lr_lambda))
        if init_lambda < 0:
            raise ValueError("init_lambda must be >= 0, got {!r}".format(init_lambda))
        super().__init__(env)
        self.budget = float(budget)
        self.lr_lambda = float(lr_lambda)
        self.lambda_ = float(init_lambda)
        self._episode_penalized_return = 0.0

    def reset(self, *, seed=None, options=None):
        self._episode_penalized_return = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        lambda_before_update = self.lambda_
        shaped_reward = float(reward) - lambda_before_update * float(cost)
        self._episode_penalized_return += shaped_reward
        info["reward_unpenalized"] = float(reward)
        info["lagrangian_lambda"] = lambda_before_update
        info["budget"] = self.budget
        if terminated or truncated:
            info["episode_penalized_return"] = self._episode_penalized_return
            episode_cost = info.get("episode_cost", 0.0)
            self.lambda_ = max(
                0.0, lambda_before_update + self.lr_lambda * (episode_cost - self.budget)
            )
            info["lagrangian_lambda_before_update"] = lambda_before_update
            info["lagrangian_lambda_after_update"] = self.lambda_
            # Overwrite with the post-update value so callers see the new λ.
            info["lagrangian_lambda"] = self.lambda_
        return observation, shaped_reward, cost, terminated, truncated, info


class SafetyToGymWrapper(gymnasium.Wrapper):
    """Convert a Safety-Gymnasium step API into a standard Gymnasium one."""

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["cost"] = float(cost)
        return observation, reward, terminated, truncated, info
