#!/usr/bin/env python3
"""
CLI entry point for running DRL experiments.

This module provides command-line interface for the Green Security
Optimization Framework. It parses arguments, creates configurations,
and delegates to ExperimentManager for execution.

Usage:
    python run_experiment.py --mode train --model-type sac --num-episodes 100
"""

import argparse
from argparse import BooleanOptionalAction
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from custom_splunk.envs.custom_splunk_env import SplunkConfig
from config import config
from experiment_manager_new import ExperimentManager

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command-line arguments.

    Note: Defaults are NOT set here - they come from config files.
    Arguments provided via CLI override config values.
    """
    parser = argparse.ArgumentParser(
        description='Run DRL experiments for Green Security Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and experiment configuration
    parser.add_argument('--model-name', type=str,
                        help='Name of the model to load (without .zip extension)')
    parser.add_argument('--mode', type=str,
                        choices=['train', 'eval_post_training', 'retrain', 'optuna_search'],
                        help='Experiment mode (default: from config)')
    parser.add_argument('--model-type', type=str,
                        choices=['ppo', 'a2c', 'dqn', 'sac', 'td3', 'recurrent_ppo'],
                        help='RL algorithm to use (default: from config)')
    parser.add_argument('--policy-type', type=str,
                        help='Policy network type (default: from config)')

    # Reward parameters
    parser.add_argument('--alpha-energy', type=float,
                        help='Energy reward weight (default: from config)')
    parser.add_argument('--beta-alert', type=float,
                        help='Alert reward weight (default: from config)')
    parser.add_argument('--gamma-dist', type=float,
                        help='Distribution reward weight (default: from config)')
    parser.add_argument('--alert-epsilon', type=float,
                        help='Alert reward epsilon (default: from config)')
    parser.add_argument('--normalizer-factor', type=float,
                        help='Reward normalizer factor (default: from config)')
    parser.add_argument('--alert-reward-method', type=str,
                        help='Alert reward calculation method (default: from config)')
    parser.add_argument('--distribution-reward-method', type=str,
                        help='Distribution reward calculation method (default: from config)')
    parser.add_argument('--reward-mode', type=str,
                        choices=['legacy', 'constrained'],
                        help='Reward aggregation mode (default: from config)')

    # Training parameters
    parser.add_argument('--learning-rate', type=float,
                        help='Learning rate (default: from config)')
    parser.add_argument('--num-episodes', type=int,
                        help='Number of episodes to run (default: from config)')

    # Environment configuration
    parser.add_argument('--hosts-num', type=int,
                        help='Percentage of hosts to use 0-100 (default: from config)')
    parser.add_argument('--additional-percentage', type=float,
                        help='Additional percentage for log generation (default: from config)')
    parser.add_argument('--action-type', type=str,
                        choices=['Action8', 'Action12', 'SoftmaxDistribution', 'SmoothTrigger',
                                 'SmoothTriggerVolume', 'SoftmaxDistributionVolume'],
                        help='Action space type (default: from config)')
    parser.add_argument('--ip', type=int,
                        help='Splunk host IP identifier 1,2,3... (default: 1)')

    parser.add_argument('--num-envs', type=int,
                        help='Number of parallel environments for training (default: from config). '
                             'Values >1 use SubprocVecEnv; requires mock mode or multiple Splunk hosts.')

    # Flags
    parser.add_argument('--random-agent', action='store_true',
                        help='Use random agent instead of trained model')
    parser.add_argument('--test-experiment', action='store_true',
                        help='Run in test mode (disables injection)')
    parser.add_argument('--eval-during-training', action=BooleanOptionalAction,
                        help='Enable/disable eval callback during training '
                             '(use --no-eval-during-training to disable)')

    # Logging
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: from config)')

    # Optuna HPO arguments
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--optuna-study-name', type=str, default=None,
                        help='Optuna study name (for resuming). Auto-generated if not set.')
    parser.add_argument('--trial-episodes-fraction', type=float, default=0.2,
                        help='Fraction of full episodes per trial (default: 0.2)')
    parser.add_argument('--no-retrain-best', action='store_true',
                        help='Skip automatic retraining of the best trial')

    return parser.parse_args()


def create_env_config_from_args(args):
    """Create SplunkConfig from parsed arguments, using config defaults when args not provided."""
    return SplunkConfig(
        rule_frequency=config.get('splunk.rule_frequency', 120),
        search_window=config.get('splunk.search_window', 2880),
        logs_per_minute=config.get('splunk.logs_per_minute', 150),
        additional_percentage=args.additional_percentage if args.additional_percentage is not None
                             else config.get('environment.additional_percentage', 1.0),
        action_duration=config.get('splunk.action_duration', 14400),
        num_of_measurements=config.get('splunk.num_measurements', 1),
        baseline_num_of_measurements=config.get('splunk.baseline_num_measurements', 1),
        env_id=config.get('splunk.train_env_id', 'splunk_train-v32'),
        end_time=config.get('splunk.end_time', '08/01/2025:00:00:00'),
        ip=args.ip if args.ip is not None else 1
    )


def create_overrides_from_args(args, model_path=None):
    """
    Create overrides dictionary from CLI arguments.

    Only includes arguments that were explicitly provided (not None).
    These override values from config files.

    Args:
        args: Parsed command-line arguments
        model_path: Optional model path (computed from model_name and host)

    Returns:
        dict: Overrides dictionary with only explicitly provided values
    """
    overrides = {}

    # Experiment configuration
    if args.mode is not None:
        overrides['experiment.mode'] = args.mode
    if args.model_type is not None:
        overrides['experiment.model_type'] = args.model_type
    if args.policy_type is not None:
        overrides['experiment.policy_type'] = args.policy_type
    if model_path is not None:
        overrides['model_path'] = model_path
    if args.test_experiment:
        overrides['experiment_name'] = 'test_experiment'

    # Reward parameters
    if args.alpha_energy is not None:
        overrides['reward.alpha'] = args.alpha_energy
    if args.beta_alert is not None:
        overrides['reward.beta'] = args.beta_alert
    if args.gamma_dist is not None:
        overrides['reward.gamma'] = args.gamma_dist
    if args.alert_epsilon is not None:
        overrides['reward.epsilon'] = args.alert_epsilon
    if args.normalizer_factor is not None:
        overrides['reward.normalizer_factor'] = args.normalizer_factor
    if args.alert_reward_method is not None:
        overrides['reward.alert_method'] = args.alert_reward_method
    if args.distribution_reward_method is not None:
        overrides['reward.distribution_method'] = args.distribution_reward_method
    if args.reward_mode is not None:
        overrides['reward.reward_mode'] = args.reward_mode

    # Training parameters
    if args.learning_rate is not None:
        overrides['training.learning_rate'] = args.learning_rate
    if args.num_episodes is not None:
        overrides['training.num_episodes'] = args.num_episodes

    # Environment configuration
    if args.num_envs is not None:
        overrides['training.n_envs'] = args.num_envs
    if args.hosts_num is not None:
        overrides['environment.hosts_percentage'] = args.hosts_num
    if args.additional_percentage is not None:
        overrides['environment.additional_percentage'] = args.additional_percentage
    if args.action_type is not None:
        overrides['environment.action_type'] = args.action_type

    # Flags
    if args.random_agent:
        overrides['use_random_agent'] = True
    if args.eval_during_training is not None:
        overrides['callbacks.eval.enabled'] = args.eval_during_training

    # Logging
    if args.log_level is not None:
        overrides['logging.level'] = args.log_level

    return overrides


def print_experiment_summary(args, overrides):
    """Print configuration summary before starting experiment."""
    mode = overrides.get('experiment.mode', config.get('experiment.mode', 'train'))
    model_type = overrides.get('experiment.model_type', config.get('experiment.model_type', 'sac'))
    alpha = overrides.get('reward.alpha', config.get('reward.alpha', 0.5))
    beta = overrides.get('reward.beta', config.get('reward.beta', 0.5))
    gamma = overrides.get('reward.gamma', config.get('reward.gamma', 0.2))
    hosts = overrides.get('environment.hosts_percentage', config.get('environment.hosts_percentage', 100))
    action_type = overrides.get('environment.action_type', config.get('environment.action_type', 'Action8'))

    logger.info("="*80)
    logger.info("Starting Green Security Optimization Experiment")
    logger.info("="*80)
    logger.info(f"Mode: {mode}")
    logger.info(f"Model: {args.model_name or 'New model (training)'}")
    logger.info(f"Algorithm: {model_type.upper()}")
    logger.info(f"Reward weights - Alpha (energy): {alpha}, "
                f"Beta (alert): {beta}, Gamma (dist): {gamma}")
    logger.info(f"Hosts: {hosts}%, Action type: {action_type}")
    logger.info("="*80)


def main():
    """Main CLI entry point for running experiments."""
    # Parse command-line arguments
    args = parse_arguments()

    # Get IP value (default to 1 if not provided)
    ip = args.ip if args.ip is not None else 1

    # Get Splunk host from environment
    host = os.getenv(f"SPLUNK_HOST_{ip}")
    if not host:
        raise ValueError(f"SPLUNK_HOST_{ip} not found in environment variables")

    # Determine model path if loading existing model (for retrain/eval modes)
    # For "train" mode without a model_name, we create a new model (no path needed)
    model_path = None
    mode = args.mode if args.mode is not None else config.get('experiment.mode', 'train')

    if args.model_name and mode != 'train':
        # Only set model_path for retrain/eval modes
        base_dir = config.get('paths.splunk_research_dir')
        exp_dir = f"{base_dir}/host_{host}_experiments"

        # Try new per-experiment directory structure first, then legacy flat layout
        new_path = f"{exp_dir}/runs/{args.model_name}/models/final"
        legacy_path = f"{exp_dir}/models/{args.model_name}"

        if os.path.exists(f"{new_path}.zip"):
            model_path = new_path
        else:
            model_path = legacy_path
        logger.info(f"Resolved model path: {model_path}")
    elif args.model_name and mode == 'train':
        logger.warning(f"model_name '{args.model_name}' provided but mode is 'train' - will create new model instead")

    # Create overrides dict from CLI arguments
    overrides = create_overrides_from_args(args, model_path)

    # Print configuration summary
    print_experiment_summary(args, overrides)

    # Create environment configuration
    env_config = create_env_config_from_args(args)

    # Create experiment manager with host-specific directory
    base_dir = config.get('paths.splunk_research_dir')
    experiment_dir = f"{base_dir}/host_{host}_experiments"
    logger.info(f"Experiment directory: {experiment_dir}")

    # Handle Optuna search mode separately
    if mode == 'optuna_search':
        from optuna_search import run_optuna_search
        study = run_optuna_search(
            env_config=env_config,
            base_overrides=overrides,
            experiment_dir=experiment_dir,
            n_trials=args.n_trials,
            study_name=args.optuna_study_name,
            trial_episodes_fraction=args.trial_episodes_fraction,
            retrain_best=not args.no_retrain_best,
        )
        logger.info(f"Optuna search complete. Pareto front: {len(study.best_trials)} trials")
        return

    manager = ExperimentManager(base_dir=experiment_dir)

    # Run experiment
    try:
        results = manager.run_experiment(env_config, overrides)
        logger.info("="*80)
        logger.info("Experiment completed successfully!")
        logger.info(f"Results: {results}")
        logger.info("="*80)
        return results
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
