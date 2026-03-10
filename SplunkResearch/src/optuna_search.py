"""
Optuna-based multi-objective hyperparameter optimization for DRL experiments.

Uses NSGA-II to search for Pareto-optimal configurations across three
objectives: maximize energy_reward, minimize alert_gap, minimize
distribution_value (KL divergence). Each trial runs a fixed-budget
experiment via ExperimentManager, and the best configuration can be
automatically retrained at full budget.

Usage (via run_experiment.py):
    python -m SplunkResearch.src.run_experiment \
        --mode optuna_search --n-trials 50 --ip 1
"""

import datetime
import gc
import logging
import os
import csv
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from config import config

logger = logging.getLogger(__name__)

# Fixed trial budget (20k episodes = 240k steps with 12 steps per episode)
TRIAL_EPISODES = 20_000


def define_search_space(trial: optuna.Trial) -> dict:
    """Sample hyperparameters and return a config override dict.

    Simplified 5D search space focusing on the most impactful parameters:
    - Model type: {sac, td3, ppo, a2c} (4 choices)
    - Learning rate: continuous (1e-4 to 5e-3, log scale)
    - State type: {StateWrapper7, StateWrapper8} (2 choices)
    - Alert sensitivity: continuous (0.05 to 0.5, log scale)
    - KL sensitivity: continuous (0.05 to 0.2, log scale)

    Network architecture, algorithm-specific params, and power-law exponents
    are fixed to default config values to reduce search dimensionality.

    Args:
        trial: Optuna trial object used for sampling.

    Returns:
        Dictionary of config overrides keyed by dot-notation config paths.
    """
    overrides: Dict[str, Any] = {}

    # ---- Model type (4 choices) ----
    model_type = trial.suggest_categorical(
        'model_type', ['sac', 'td3', 'ppo', 'a2c'])
    overrides['experiment.model_type'] = model_type

    # ---- Learning rate (log scale) ----
    overrides['training.learning_rate'] = trial.suggest_float(
        'learning_rate', 1e-4, 5e-3, log=True)

    # ---- State type (observation space variant) ----
    # StateWrapper7: standard distributed state (default)
    # StateWrapper8: alert-aware state with additional alert features
    state_type = trial.suggest_categorical(
        'state_type', ['StateWrapper7', 'StateWrapper8'])
    overrides['environment.state_type'] = state_type

    # ---- Constrained reward sensitivity parameters ----
    # These control how fast the tanh penalties saturate past their thresholds.
    # Higher sensitivity = slower saturation (softer penalty), lower = faster (harder penalty)
    overrides['reward.alert_sensitivity'] = trial.suggest_float(
        'alert_sensitivity', 0.05, 0.5, log=True)
    overrides['reward.kl_sensitivity'] = trial.suggest_float(
        'kl_sensitivity', 0.05, 0.2, log=True)

    # All other parameters use default config values:
    # - Network architecture: from config/default.yaml (training.ppo.policy_kwargs, training.sac.policy_net_arch)
    # - Algorithm-specific params: from config/default.yaml (batch_size, buffer_size, ent_coef, use_sde, n_steps)
    # - Power-law exponents: from config/default.yaml (reward.power_*_exponent)
    # - Reward methods: from config/default.yaml (reward.alert_method, reward.distribution_method)

    return overrides


class OptunaReportCallback(BaseCallback):
    """SB3 callback that collects episode metrics for Optuna multi-objective trials.

    Collects per-episode values from ``info`` dicts at episode boundaries.
    On training end, stores tail-20% averages as trial user attrs.

    Also logs global metrics (fps, step count) to a shared CSV for cross-trial analysis.

    Note: ``trial.report()`` / ``trial.should_prune()`` are not supported for
    multi-objective studies, so pruning is not used here.
    """

    def __init__(self, trial: optuna.Trial, verbose: int = 0, global_metrics_file: str = None):
        super().__init__(verbose)
        self.trial = trial
        self._episode_energy: list[float] = []
        self._episode_alert_gap: list[float] = []
        self._episode_dist_value: list[float] = []
        self._episode_count = 0
        self.global_metrics_file = global_metrics_file
        self._step_start_time = datetime.datetime.now()
        self._step_start_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        if not infos:
            return True

        # Collect from all envs (supports both single and SubprocVecEnv)
        for i, info in enumerate(infos):
            if i < len(dones) and dones[i] and info:
                energy = info.get('energy_reward', 0.0)
                alert_gap = (
                    info.get('combined_metrics', {}).get('alert', 0.0)
                    - info.get('combined_baseline_metrics', {}).get('alert', 0.0)
                )
                dist_value = info.get('ac_distribution_value', 0.0)

                self._episode_energy.append(float(energy))
                self._episode_alert_gap.append(float(alert_gap))
                self._episode_dist_value.append(float(dist_value))
                self._episode_count += 1

        return True

    def _store_final_attrs(self):
        """Store tail-20% average values as trial user attributes."""
        for name, history in [
            ('final_energy_reward', self._episode_energy),
            ('final_alert_gap', self._episode_alert_gap),
            ('final_distribution_value', self._episode_dist_value),
        ]:
            if history:
                tail_n = max(1, len(history) // 5)
                self.trial.set_user_attr(name, float(np.mean(history[-tail_n:])))
            else:
                self.trial.set_user_attr(name, 0.0)

    def _on_training_end(self) -> None:
        self._store_final_attrs()


def objective(
    trial: optuna.Trial,
    env_config,
    base_overrides: dict,
    experiment_dir: str,
    trial_episodes: int,
) -> Tuple[float, float, float]:
    """Optuna objective: run a fixed-budget experiment and return 3 objectives.

    Args:
        trial: Optuna trial.
        env_config: SplunkConfig for environment creation.
        base_overrides: CLI overrides to merge with sampled HPs.
        experiment_dir: Host experiment directory path.
        trial_episodes: Number of episodes per trial (fixed budget).

    Returns:
        Tuple of (energy_reward, alert_gap, distribution_value).
        Directions: maximize energy, minimize alert_gap, minimize distribution_value.
    """
    from experiment_manager_new import ExperimentManager

    # Sample hyperparameters
    hp_overrides = define_search_space(trial)

    # Merge: base CLI overrides → sampled HPs (HP overrides win)
    merged = {**base_overrides, **hp_overrides}

    # Each trial is a fresh training run with fixed budget
    merged['experiment.mode'] = 'train'
    merged['training.num_episodes'] = trial_episodes

    # Disable eval callback for faster trials
    merged['callbacks.eval.enabled'] = False

    # Inject the Optuna reporting callback
    report_cb = OptunaReportCallback(trial)
    merged['_extra_callbacks'] = [report_cb]

    # Tag the sub-experiment
    merged['experiment_name'] = f'optuna_trial_{trial.number}'

    manager = ExperimentManager(base_dir=experiment_dir)
    try:
        manager.run_experiment(env_config, merged)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned(f"Trial failed: {e}")
    finally:
        # Flush whatever was collected — _on_training_end may not fire on
        # interruption (SIGTERM, exception mid-training), so we store here too.
        # _store_final_attrs is idempotent: safe to call twice.
        report_cb._store_final_attrs()
        # Break the report_cb → model reference so the SB3 model (and its
        # TensorBoardOutputFormat / SummaryWriter background thread) can be
        # GC'd before the next trial.  Without this, report_cb keeps the model
        # alive via a straight reference path that gc.collect() cannot break,
        # causing SummaryWriter instances to accumulate and silently fail for
        # trial 1+ (no FPS / ep_rew_mean / loss in TensorBoard).
        report_cb.model = None
        # Release manager (closes open handles) and force memory reclamation
        # between Optuna trials to prevent OOM accumulation.
        del manager
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()

    energy = trial.user_attrs.get('final_energy_reward', 0.0)
    alert_gap = trial.user_attrs.get('final_alert_gap', 0.0)
    dist_value = trial.user_attrs.get('final_distribution_value', 0.0)

    logger.info(
        f"Trial {trial.number} finished: energy={energy:.4f}, "
        f"alert_gap={alert_gap:.4f}, dist_value={dist_value:.4f}"
    )
    return energy, alert_gap, dist_value


def _select_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Select best trial from Pareto front using normalized-sum heuristic.

    Objectives: maximize energy (idx 0), minimize alert_gap (idx 1),
    minimize distribution_value (idx 2).

    Normalizes each objective to [0, 1] and inverts minimize-direction
    objectives so that higher score always means better.
    """
    pareto = study.best_trials
    if len(pareto) == 1:
        return pareto[0]

    values = np.array([t.values for t in pareto])
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    ranges = maxs - mins
    # Avoid division by zero for constant objectives
    ranges[ranges == 0] = 1.0
    normalized = (values - mins) / ranges  # 0=min, 1=max for each column

    # Directions: [maximize, minimize, minimize]
    # For minimize objectives, lower raw value is better → invert normalized score
    directions = np.array([1.0, -1.0, -1.0])  # +1 = keep, -1 = invert
    quality = np.where(directions > 0, normalized, 1.0 - normalized)
    scores = quality.sum(axis=1)
    best_idx = int(np.argmax(scores))
    return pareto[best_idx]


def _rebuild_overrides_from_trial(best: optuna.trial.FrozenTrial,
                                  base_overrides: dict) -> dict:
    """Reconstruct config overrides from a completed trial's params.

    Only sets the keys that were actually sampled in define_search_space:
    - model_type
    - learning_rate
    - state_type
    - alert_sensitivity
    - kl_sensitivity

    All other parameters (network arch, algo-specific, power exponents, reward methods)
    use default config values and are not overridden.
    """
    p = best.params
    overrides = {**base_overrides}

    # Sampled parameters
    overrides['experiment.model_type'] = p['model_type']
    overrides['training.learning_rate'] = p['learning_rate']
    overrides['environment.state_type'] = p['state_type']
    overrides['reward.alert_sensitivity'] = p['alert_sensitivity']
    overrides['reward.kl_sensitivity'] = p['kl_sensitivity']

    # All other parameters come from config/default.yaml (not sampled)

    return overrides


def run_optuna_search(
    env_config,
    base_overrides: dict,
    experiment_dir: str,
    n_trials: int = 50,
    study_name: Optional[str] = None,
    trial_episodes_fraction: float = 0.2,
    retrain_best: bool = True,
) -> optuna.Study:
    """Run multi-objective HPO search and optionally retrain the best config.

    Args:
        env_config: SplunkConfig for environment creation.
        base_overrides: CLI overrides to merge with sampled HPs.
        experiment_dir: Host experiment directory path.
        n_trials: Number of Optuna trials.
        study_name: Study name (also used for SQLite DB filename).
        trial_episodes_fraction: Unused (kept for CLI compatibility).
            Trials always run TRIAL_EPISODES episodes.
        retrain_best: Whether to retrain the best trial at full budget.

    Returns:
        The completed Optuna study.
    """
    from experiment_manager_new import ExperimentManager

    if study_name is None:
        study_name = f"optuna_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    runs_dir = f"{experiment_dir}/runs"
    os.makedirs(runs_dir, exist_ok=True)
    storage = f"sqlite:///{runs_dir}/{study_name}.db"

    logger.info(f"Starting Optuna search: {n_trials} trials, study={study_name}")
    logger.info(f"Storage: {storage}")
    logger.info(f"Trial budget: {TRIAL_EPISODES} episodes")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial, env_config, base_overrides,
            experiment_dir, TRIAL_EPISODES,
        ),
        n_trials=n_trials,
    )

    # Log Pareto front
    pareto = study.best_trials
    logger.info(f"Search complete. Pareto front size: {len(pareto)}")
    for t in pareto:
        logger.info(
            f"  Trial {t.number}: energy={t.values[0]:.4f}, "
            f"alert_gap={t.values[1]:.4f}, dist_value={t.values[2]:.4f} "
            f"[{t.params.get('model_type', '?')}, {t.params.get('alert_method', '?')}]"
        )

    # Optionally retrain best trial at full budget
    if retrain_best and pareto:
        best = _select_best_trial(study)
        logger.info(
            f"Retraining best trial {best.number} at full episode budget. "
            f"Params: {best.params}"
        )

        retrain_overrides = _rebuild_overrides_from_trial(best, base_overrides)
        retrain_overrides['experiment.mode'] = 'train'
        retrain_overrides['experiment_name'] = f'optuna_best_trial_{best.number}'
        # Full budget from CLI --num-episodes (don't cap to TRIAL_EPISODES)

        manager = ExperimentManager(base_dir=experiment_dir)
        manager.run_experiment(env_config, retrain_overrides)

    return study
