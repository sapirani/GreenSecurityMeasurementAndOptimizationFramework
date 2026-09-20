import math
from logging import Logger
from typing import cast, Any
import torch as th
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from DTOs.hadoop.drl.training.training_config import TrainingConfig, UserDefinedTrainingParams, PPOModelConfig


# todo: make it more generic (not tailored to ppo with actor critic)
class PPODebugCallback(BaseCallback):
    def __init__(
            self,
            logger: Logger,
            train_id: str,
            user_defined_training_params: UserDefinedTrainingParams,
            verbose: int = 0,
    ):
        super().__init__(verbose)

        self.debugging_logger = logger
        self.train_id = train_id
        self._last_logged_update = -1
        self._rollout_num = 0
        self.user_defined_training_params = user_defined_training_params

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        """
        Runs once, when starting training
        """

        model = cast(PPO, self.model)

        training_config = TrainingConfig(
            training_id=self.train_id,
            drl_config=PPOModelConfig.from_model(model),
            user_defined_params=self.user_defined_training_params,
        )

        self.debugging_logger.info(
            "Training Hyperparameters",
            extra=training_config.model_dump(by_alias=True, mode="json")
        )

        for handler in self.debugging_logger.handlers:
            handler.flush()

    def _on_rollout_start(self):
        """
        Runs in the beginning of a new batch of n_steps steps.
        This function is used to output the metrics from the end of the previous rollout
        """
        model = cast(PPO, self.model)
        current_update = self.model._n_updates

        if current_update <= 0 or current_update == self._last_logged_update:
            return

        metrics = {
            key: value for key, value in self.model.logger.name_to_value.items()
            if key.startswith("train/")
        }

        if metrics:
            # ------------------------------------------------------------
            # Critic explained variance
            #
            # 1.0  → excellent value prediction - critic predictions explain the variation in the return targets very well
            # 0.0  → no better than predicting the mean of the target values
            # < 0  → worse than predicting the mean
            # ------------------------------------------------------------
            if math.isnan(metrics["train/explained_variance"]):
                metrics["train/explained_variance"] = 10    # max value is 1 - so now we know it was nan

            del metrics["train/clip_range"]

            self.debugging_logger.info(
                "PPO Training - Update Completed",
                extra={
                    "training_id": self.train_id,
                    "global_step": self.num_timesteps,
                    "update_num": current_update,
                    "rollout_num": self._rollout_num,
                    **metrics,
                },
            )

            for handler in self.debugging_logger.handlers:
                handler.flush()

            self._last_logged_update = current_update

    def _on_rollout_end(self):
        """
        Runs after n_steps (before gradients update)
        """

        self._rollout_num += 1

        model = cast(PPO, self.model)
        policy = cast(ActorCriticPolicy, model.policy)
        rollout_buffer = model.rollout_buffer

        observations = th.as_tensor(rollout_buffer.observations, device=policy.device)
        observations = observations.reshape(-1, *observations.shape[2:])

        actions = th.as_tensor(rollout_buffer.actions, device=policy.device)
        actions = actions.reshape(-1, actions.shape[-1])

        with th.no_grad():
            distribution = policy.get_distribution(observations)

            entropy = distribution.entropy()
            action_mean = distribution.distribution.mean
            action_std = distribution.distribution.stddev

            # Log-probability for each action dimension separately
            per_dim_log_prob = distribution.distribution.log_prob(actions)

        # Restore [n_steps, n_envs, ...]
        if entropy is not None:
            entropy = entropy.reshape(rollout_buffer.buffer_size, rollout_buffer.n_envs)
        action_mean = action_mean.reshape(rollout_buffer.buffer_size, rollout_buffer.n_envs, -1)
        action_std = action_std.reshape(rollout_buffer.buffer_size, rollout_buffer.n_envs, -1)
        per_dim_log_prob = per_dim_log_prob.reshape(rollout_buffer.buffer_size, rollout_buffer.n_envs, -1)

        # rollout_buffer vals are indexed as: [n_steps, n_envs]
        for step in range(rollout_buffer.buffer_size):
            for env_idx in range(rollout_buffer.n_envs):
                global_step = (
                        self.num_timesteps
                        - (rollout_buffer.buffer_size - step) * rollout_buffer.n_envs
                        + env_idx
                )

                self.debugging_logger.info(
                    "PPO Training Transition",
                    extra={
                        "training_id": self.train_id,
                        "env_index": env_idx,
                        "rollout_step": step,

                        # ------------------------------------------------
                        # Transition
                        # ------------------------------------------------
                        "state_t": rollout_buffer.observations[step, env_idx].tolist(),
                        "action_t": rollout_buffer.actions[step, env_idx].tolist(),
                        "reward_t": float(rollout_buffer.rewards[step, env_idx]),

                        # ------------------------------------------------
                        # PPO quantities
                        # ------------------------------------------------
                        "state_t_value": float(rollout_buffer.values[step, env_idx]),   # V(s_t)
                        # Joint log-probability assigned to the sampled action.
                        # Higher value means that the sampled action was more likely under the policy.
                        # Interpret alongside action_mean and action_std.
                        "action_log_prob": float(rollout_buffer.log_probs[step, env_idx]),
                        "advantage": float(rollout_buffer.advantages[step, env_idx]),
                        "critic_target_value": float(rollout_buffer.returns[step, env_idx]),

                        # ------------------------------------------------
                        # Policy distribution
                        # ------------------------------------------------
                        # mean and std form normal distribution parameters that the action is taken from
                        "action_mean": action_mean[step, env_idx].cpu().tolist(),
                        "action_std": action_std[step, env_idx].cpu().tolist(),
                        # Entropy measures the spread/uncertainty of the policy distribution.
                        # Higher entropy generally means more exploration.
                        "entropy": None if entropy is None else float(entropy[step, env_idx]),

                        # ------------------------------------------------
                        # Training position
                        # ------------------------------------------------
                        "global_step": global_step,
                    }
                )


        # ============================================================
        # Rollout-level aggregate statistics
        # ============================================================

        statistics: dict[str, Any] = {
            "training_id": self.train_id,
            "global_step": self.num_timesteps,
            "rollout_num": self._rollout_num,
        }

        # ------------------------------------------------------------
        # Aggregate PPO quantities across steps and environments
        # ------------------------------------------------------------
        self._add_statistics(statistics, "reward", rollout_buffer.rewards)
        # Whether the actions taken were better (positive number) or worse (negative number) than what critic expected
        self._add_statistics(statistics, "advantage", rollout_buffer.advantages)
        # Critic's predicted state value V(s) for each state in the rollout
        self._add_statistics(statistics, "critic_value_estimate", rollout_buffer.values)
        # GAE-based return target to train the critic toward the expected future return from each state
        self._add_statistics(statistics, "critic_target_value", rollout_buffer.returns)
        # Whether actions taken were likely to be chosen by the policy. Higher = more likely
        # Basically, it may be inferred from the action mean, action std, and the actual action taken
        self._add_statistics(statistics, "joint_action_log_prob", rollout_buffer.log_probs)

        # ------------------------------------------------------------
        # Actions statistics' are separate for each action dimension.
        # ------------------------------------------------------------
        actions = rollout_buffer.actions
        action_dim = actions.shape[-1]

        for action_idx in range(action_dim):
            self._add_statistics(statistics, f"action_{action_idx}", actions[..., action_idx])
            self._add_statistics(
                statistics,
                f"action_{action_idx}_log_prob",
                per_dim_log_prob[..., action_idx].detach().cpu().numpy()
            )

        # ------------------------------------------------------------
        # Advantage properties
        # Roughly say: whether the actions taken were better or worse than what critic expected
        # ------------------------------------------------------------
        advantages = rollout_buffer.advantages
        statistics["advantage_positive_fraction"] = float(np.mean(advantages > 0))  # actions were better than expected
        statistics["advantage_negative_fraction"] = float(np.mean(advantages < 0))  # actions were worse than expected
        statistics["advantage_zero_fraction"] = float(np.mean(advantages == 0))     # actions were as good as expected

        if entropy is not None:
            entropy_np = entropy.cpu().numpy()
            self._add_statistics(statistics, "entropy", entropy_np)

        self.debugging_logger.info(
            "PPO Training - Rollout Statistics",
            extra=statistics,
        )

        for handler in self.debugging_logger.handlers:
            handler.flush()

    @staticmethod
    def _add_statistics(output, prefix, values):
        values = np.asarray(values)

        if values.size == 0:
            return

        output[f"{prefix}_mean"] = float(np.mean(values))
        output[f"{prefix}_std"] = float(np.std(values))
        output[f"{prefix}_min"] = float(np.min(values))
        output[f"{prefix}_max"] = float(np.max(values))
        output[f"{prefix}_p01"] = float(np.percentile(values, 1))
        output[f"{prefix}_p25"] = float(np.percentile(values, 25))
        output[f"{prefix}_p50"] = float(np.percentile(values, 50))
        output[f"{prefix}_p75"] = float(np.percentile(values, 75))
        output[f"{prefix}_p99"] = float(np.percentile(values, 99))
