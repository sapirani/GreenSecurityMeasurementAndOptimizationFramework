"""
Optuna-based multi-objective hyperparameter optimization for DRL experiments.

Uses NSGA-II to search for Pareto-optimal configurations across energy,
alert, and distribution reward objectives. Each trial runs a fixed-budget
experiment (600K steps) via ExperimentManager, and the best configuration
can be automatically retrained at full budget.

Usage (via run_experiment.py):
    python -m SplunkResearch.src.run_experiment \
        --mode optuna_search --n-trials 50 --ip 1
"""

import datetime
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback

from config import config

logger = logging.getLogger(__name__)

# Fixed trial budget: 600K timesteps = 50,000 episodes * 12 steps/episode
TRIAL_EPISODES = 50_000


def define_search_space(trial: optuna.Trial) -> dict:
    """Sample hyperparameters and return a config override dict.

    Args:
        trial: Optuna trial object used for sampling.

    Returns:
        Dictionary of config overrides keyed by dot-notation config paths.
    """
    overrides: Dict[str, Any] = {}

    # ---- Model type ----
    model_type = trial.suggest_categorical(
        'model_type', ['sac', 'td3', 'ppo', 'a2c'])
    overrides['experiment.model_type'] = model_type

    # ---- Shared hyperparameters ----
    overrides['training.learning_rate'] = trial.suggest_float(
        'learning_rate', 1e-5, 1e-2, log=True)
    overrides['training.gamma'] = trial.suggest_float(
        'gamma', 0.9, 0.999, log=True)

    # Network architecture
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layer_size = trial.suggest_categorical('layer_size', [64, 128, 256, 512])
    arch = [layer_size] * n_layers

    # ---- Algorithm-specific hyperparameters ----
    if model_type in ('sac', 'td3'):
        overrides['training.sac.batch_size'] = trial.suggest_categorical(
            'batch_size', [256, 512, 1024, 2048])
        overrides['training.sac.buffer_size'] = trial.suggest_categorical(
            'buffer_size', [50_000, 100_000, 200_000])
        overrides['training.sac.policy_net_arch'] = {'pi': arch, 'qf': arch}

        if model_type == 'sac':
            ent_coef = trial.suggest_categorical(
                'ent_coef', ['auto', '0.01', '0.05', '0.1', '0.2'])
            overrides['training.sac.ent_coef'] = (
                ent_coef if ent_coef == 'auto' else float(ent_coef))
            use_sde = trial.suggest_categorical('use_sde', [True, False])
            overrides['training.sac.use_sde'] = use_sde

    elif model_type in ('ppo', 'a2c'):
        overrides['training.n_steps'] = trial.suggest_categorical(
            'n_steps', [512, 1024, 2048, 4096])
        overrides['training.ent_coef'] = trial.suggest_float(
            'on_policy_ent_coef', 0.001, 0.2, log=True)
        overrides['training.ppo.use_sde'] = trial.suggest_categorical(
            'on_policy_use_sde', [True, False])
        # PPO/A2C use policy_kwargs differently — set net_arch as a flat list
        overrides['training.ppo.policy_kwargs'] = {'net_arch': arch}

    # ---- Reward method ----
    alert_method = trial.suggest_categorical(
        'alert_method',
        ['ExpectedDiffPower', 'PredictedTanh', 'RelativeDiffPower',
         'RelativeDiffScaledTanh', 'ExpectedDiffScaledTanh'])
    overrides['reward.alert_method'] = alert_method

    distribution_method = trial.suggest_categorical(
        'distribution_method',
        ['DistributionRewardWrapper', 'DistributionRewardWrapper1',
         'DistributionRewardWrapper2'])
    overrides['reward.distribution_method'] = distribution_method

    # ---- Reward weights ----
    overrides['reward.alpha'] = trial.suggest_float('alpha', 0.1, 2.0)
    overrides['reward.beta'] = trial.suggest_float('beta', 0.1, 2.0)
    overrides['reward.gamma'] = trial.suggest_float('gamma_reward', 0.1, 2.0)

    # Power-law exponents (relevant for Power normalizers)
    overrides['reward.power_alert_exponent'] = trial.suggest_float(
        'power_alert_exponent', 1.0, 4.0)
    overrides['reward.power_energy_exponent'] = trial.suggest_float(
        'power_energy_exponent', 1.0, 4.0)
    overrides['reward.power_distribution_exponent'] = trial.suggest_float(
        'power_distribution_exponent', 1.0, 4.0)

    # ---- Action parameters ----
    overrides['action.softmax_temperature'] = trial.suggest_float(
        'softmax_temperature', 5.0, 50.0)
    overrides['action.base_log_count'] = trial.suggest_int(
        'base_log_count', 5000, 50000, step=5000)

    return overrides


class OptunaReportCallback(BaseCallback):
    """SB3 callback that collects episode rewards for Optuna multi-objective trials.

    Collects per-episode reward components from ``info`` dicts at episode
    boundaries. On training end, stores tail-20% averages as trial user attrs.

    Note: ``trial.report()`` / ``trial.should_prune()`` are not supported for
    multi-objective studies, so pruning is not used here.
    """

    def __init__(self, trial: optuna.Trial, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self._episode_energy: list[float] = []
        self._episode_alert: list[float] = []
        self._episode_dist: list[float] = []
        self._episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        if not infos:
            return True

        # Collect from all envs (supports both single and SubprocVecEnv)
        for i, info in enumerate(infos):
            if i < len(dones) and dones[i] and info:
                energy = info.get('norm_energy_reward', 0.0)
                alert = info.get('norm_alert_reward', 0.0)
                dist = info.get('ac_distribution_reward', 0.0)

                self._episode_energy.append(float(energy))
                self._episode_alert.append(float(alert))
                self._episode_dist.append(float(dist))
                self._episode_count += 1

        return True

    def _store_final_attrs(self):
        """Store tail-20% average rewards as trial user attributes."""
        for name, history in [
            ('final_energy_reward', self._episode_energy),
            ('final_alert_reward', self._episode_alert),
            ('final_distribution_reward', self._episode_dist),
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
        Tuple of (energy_reward, alert_reward, distribution_reward).
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

    try:
        manager = ExperimentManager(base_dir=experiment_dir)
        manager.run_experiment(env_config, merged)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned(f"Trial failed: {e}")

    energy = trial.user_attrs.get('final_energy_reward', 0.0)
    alert = trial.user_attrs.get('final_alert_reward', 0.0)
    dist = trial.user_attrs.get('final_distribution_reward', 0.0)

    logger.info(
        f"Trial {trial.number} finished: energy={energy:.4f}, "
        f"alert={alert:.4f}, dist={dist:.4f}"
    )
    return energy, alert, dist


def _select_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Select best trial from Pareto front using normalized-sum heuristic.

    Normalizes each objective to [0, 1] across the Pareto front and picks
    the trial with highest sum (balanced trade-off).
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
    normalized = (values - mins) / ranges
    scores = normalized.sum(axis=1)
    best_idx = int(np.argmax(scores))
    return pareto[best_idx]


def _rebuild_overrides_from_trial(best: optuna.trial.FrozenTrial,
                                  base_overrides: dict) -> dict:
    """Reconstruct config overrides from a completed trial's params."""
    p = best.params
    overrides = {**base_overrides}

    # Model type
    model_type = p['model_type']
    overrides['experiment.model_type'] = model_type

    # Shared
    overrides['training.learning_rate'] = p['learning_rate']
    overrides['training.gamma'] = p['gamma']

    n_layers = p['n_layers']
    layer_size = p['layer_size']
    arch = [layer_size] * n_layers

    # Algorithm-specific
    if model_type in ('sac', 'td3'):
        overrides['training.sac.batch_size'] = p['batch_size']
        overrides['training.sac.buffer_size'] = p['buffer_size']
        overrides['training.sac.policy_net_arch'] = {'pi': arch, 'qf': arch}
        if model_type == 'sac':
            ent_coef = p.get('ent_coef', 'auto')
            overrides['training.sac.ent_coef'] = (
                ent_coef if ent_coef == 'auto' else float(ent_coef))
            overrides['training.sac.use_sde'] = p.get('use_sde', True)
    elif model_type in ('ppo', 'a2c'):
        overrides['training.n_steps'] = p.get('n_steps', 2048)
        overrides['training.ent_coef'] = p.get('on_policy_ent_coef', 0.05)
        overrides['training.ppo.use_sde'] = p.get('on_policy_use_sde', True)
        overrides['training.ppo.policy_kwargs'] = {'net_arch': arch}

    # Reward method
    overrides['reward.alert_method'] = p['alert_method']
    overrides['reward.distribution_method'] = p['distribution_method']

    # Reward weights & exponents
    overrides['reward.alpha'] = p['alpha']
    overrides['reward.beta'] = p['beta']
    overrides['reward.gamma'] = p['gamma_reward']
    overrides['reward.power_alert_exponent'] = p['power_alert_exponent']
    overrides['reward.power_energy_exponent'] = p['power_energy_exponent']
    overrides['reward.power_distribution_exponent'] = p['power_distribution_exponent']

    # Action
    overrides['action.softmax_temperature'] = p['softmax_temperature']
    overrides['action.base_log_count'] = p['base_log_count']

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
            Trials always run TRIAL_EPISODES (600K steps).
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
    logger.info(f"Trial budget: {TRIAL_EPISODES} episodes ({TRIAL_EPISODES * 12} steps)")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "maximize", "maximize"],
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
            f"alert={t.values[1]:.4f}, dist={t.values[2]:.4f} "
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
