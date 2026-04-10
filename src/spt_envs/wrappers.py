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

    λ (penalty_coeff) is a fixed hyperparameter chosen before training.
    The policy is trained to maximise the shaped reward; there is no mechanism
    to enforce a cost budget — the resulting cost is whatever the policy learns.

    Operates on the safe (6-tuple) API.

    Per-step info keys added:
        ``reward_unpenalized`` – raw task reward r (before penalty)
        ``penalty_coeff``      – the fixed λ used for shaping

    Episode-end info keys added:
        ``episode_penalized_return`` – sum of shaped rewards over the episode
                                       (what the policy actually optimised)
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
        reward = float(reward)
        cost = float(cost)
        shaped_reward = reward - self.penalty_coeff * cost
        self._episode_penalized_return += shaped_reward
        info["reward_unpenalized"] = reward
        info["penalty_coeff"] = self.penalty_coeff
        if terminated or truncated:
            info["episode_penalized_return"] = self._episode_penalized_return
        return observation, shaped_reward, cost, terminated, truncated, info


class LagrangianWrapper(gymnasium.Wrapper):
    """Lagrangian constrained RL baseline.

    Maintains a dual variable λ ≥ 0 that is updated between episodes via a
    dual gradient ascent step to automatically enforce E[C] ≤ budget:

        λ_new = max(0, λ + lr_lambda * (episode_cost − budget))

    λ increases when the episode cost exceeds the budget (making the penalty
    heavier, discouraging constraint violations) and decreases when the policy
    is safely under budget (relaxing the penalty).  Over training λ converges
    to the Lagrange multiplier that enforces the constraint at the boundary.

    The reward shaped during each episode uses the λ from the START of that
    episode (i.e. λ is held constant within an episode and only updated at the
    end):

        r_shaped = r − λ * cost

    Operates on the safe (6-tuple) API.

    Per-step info keys added:
        ``reward_unpenalized``   – raw task reward r (before penalty)
        ``lagrangian_lambda``    – λ used to shape THIS step's reward
        ``budget``               – the cost budget B

    Episode-end info keys added (in addition to the above):
        ``episode_penalized_return``      – sum of shaped rewards this episode
        ``lagrangian_lambda_before_update`` – λ used throughout this episode
        ``lagrangian_lambda_after_update``  – new λ for the next episode
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
        # λ intentionally NOT reset — it accumulates across the full training run.
        self._episode_penalized_return = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        reward = float(reward)
        cost = float(cost)

        # Capture λ for this episode before any update.
        lambda_this_episode = self.lambda_
        shaped_reward = reward - lambda_this_episode * cost
        self._episode_penalized_return += shaped_reward

        # lagrangian_lambda always reflects the λ used for THIS step's shaping.
        info["reward_unpenalized"] = reward
        info["lagrangian_lambda"] = lambda_this_episode
        info["budget"] = self.budget

        if terminated or truncated:
            episode_cost = info.get("episode_cost", 0.0)
            # Dual update: λ ← max(0, λ + α * (C_ep − B))
            self.lambda_ = max(
                0.0,
                lambda_this_episode + self.lr_lambda * (episode_cost - self.budget),
            )
            info["episode_penalized_return"] = self._episode_penalized_return
            info["lagrangian_lambda_before_update"] = lambda_this_episode
            info["lagrangian_lambda_after_update"] = self.lambda_

        return observation, shaped_reward, cost, terminated, truncated, info


class RuleBasedShieldWrapper(gymnasium.Wrapper):
    """Rule-based safety shield: intercepts actions that would move the agent
    too close to a hazard and replaces them with a repulsive action.

    The shield reads the agent and hazard positions directly from the
    Safety-Gymnasium task state *before* each step.  If the agent is within
    ``warning_radius`` metres of any hazard centre it replaces the proposed
    action with a unit repulsive vector (weighted sum away from all nearby
    hazards), scaled to the action-space limits.  Otherwise the original
    action passes through unchanged.

    This is a purely reactive, learning-free baseline.  It operates on the
    safe (6-tuple) API and adds no reward shaping — the policy trains on the
    unmodified task reward.  The cost it sees is lower than without the shield
    because fewer hazard collisions occur.

    Operates on the safe (6-tuple) API.

    Per-step info keys added:
        ``shield_intervened``            – True if the action was replaced
        ``episode_shield_interventions`` – running intervention count

    Episode-end info keys added:
        ``shield_intervention_rate`` – interventions / episode_length
    """

    # Default warning distance: keepout radius + a reaction margin.
    _DEFAULT_WARNING_RADIUS = 0.28  # DEFAULT_HAZARD_KEEPOUT (0.18) + 0.10

    def __init__(self, env, warning_radius=None):
        super().__init__(env)
        if warning_radius is None:
            warning_radius = self._DEFAULT_WARNING_RADIUS
        if warning_radius <= 0:
            raise ValueError(
                "warning_radius must be > 0, got {!r}".format(warning_radius)
            )
        self.warning_radius = float(warning_radius)
        self._episode_interventions = 0
        self._episode_steps = 0

    def reset(self, *, seed=None, options=None):
        self._episode_interventions = 0
        self._episode_steps = 0
        return self.env.reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # State access helpers
    # ------------------------------------------------------------------

    def _get_agent_xy(self):
        """Return the agent's (x, y) position from the underlying task, or None."""
        try:
            pos = self.env.unwrapped.task.agent.pos
            return np.asarray(pos, dtype=float)[:2]
        except AttributeError:
            return None

    def _get_hazard_xys(self):
        """Return (n_hazards, 2) hazard positions from the underlying task, or None."""
        try:
            pos = self.env.unwrapped.task.hazards.pos  # (n_hazards, 3)
            return np.asarray(pos, dtype=float)[:, :2]
        except AttributeError:
            return None

    # ------------------------------------------------------------------
    # Safe action computation
    # ------------------------------------------------------------------

    def _repulsive_action(self, agent_xy, hazard_xys):
        """Unit repulsive vector away from all nearby hazards, scaled to action limits."""
        diffs = agent_xy - hazard_xys          # (n_hazards, 2)
        dists = np.linalg.norm(diffs, axis=1)  # (n_hazards,)
        dists = np.maximum(dists, 1e-6)        # avoid division by zero

        # Inverse-square weighting: closer hazards contribute more force.
        weights = 1.0 / (dists ** 2)
        repulsive = np.sum(weights[:, None] * (diffs / dists[:, None]), axis=0)

        norm = np.linalg.norm(repulsive)
        if norm < 1e-8:
            # Degenerate case (equidistant or on top of hazard): move in +x.
            repulsive = np.array([1.0, 0.0])
        else:
            repulsive = repulsive / norm

        # Scale to the action-space magnitude.
        high = np.asarray(self.action_space.high, dtype=float)
        scale = float(np.min(high[:2]))
        return np.clip(repulsive * scale, self.action_space.low, self.action_space.high)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, action):
        action = np.asarray(action, dtype=float)
        intervened = False

        agent_xy = self._get_agent_xy()
        if agent_xy is not None:
            haz_xys = self._get_hazard_xys()
            if haz_xys is not None and len(haz_xys) > 0:
                dists = np.linalg.norm(haz_xys - agent_xy, axis=1)
                if np.any(dists < self.warning_radius):
                    action = self._repulsive_action(agent_xy, haz_xys)
                    intervened = True

        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        self._episode_steps += 1
        if intervened:
            self._episode_interventions += 1

        info["shield_intervened"] = intervened
        info["episode_shield_interventions"] = self._episode_interventions

        if terminated or truncated:
            info["shield_intervention_rate"] = (
                self._episode_interventions / max(1, self._episode_steps)
            )

        return observation, reward, cost, terminated, truncated, info


class SafetyToGymWrapper(gymnasium.Wrapper):
    """Convert a Safety-Gymnasium step API into a standard Gymnasium one."""

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["cost"] = float(cost)
        return observation, reward, terminated, truncated, info
