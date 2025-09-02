import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict

from experiment_manager_new import ExperimentManager, ExperimentConfig
from custom_splunk.envs.custom_splunk_env import SplunkConfig

logger = logging.getLogger(__name__)

def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/experiment.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_configs_from_args(args: Dict[str, Any]) -> ExperimentConfig:
    """Create experiment configuration from command line arguments"""
    # Create SplunkConfig
    splunk_config = SplunkConfig(
        env_id=args['env_name'],
        fake_start_datetime=args['fake_start_datetime'].replace('_', ' ') if args['fake_start_datetime'] else None,
        rule_frequency=args['rule_frequency'],
        search_window=args['search_window'],
        action_duration=args['span_size'],
        logs_per_minute=args['logs_per_minute'],
        additional_percentage=args['additional_percentage'],
        num_of_measurements=args['num_of_measurements'],
        state_strategy=args['state_strategy_version'],
        action_strategy=args['action_strategy_version']
    )

    # Create ExperimentConfig
    experiment_config = ExperimentConfig(
        env_config=splunk_config,
        model_type=args['model'] if args['model'] else "ppo",
        policy_type=args['policy'] if args['policy'] else "mlp",
        learning_rate=args['learning_rate'] if args['learning_rate'] else 3e-4,
        n_steps=args['n_steps'] if args['n_steps'] else 2048,
        gamma=args['df'] if args['df'] else 0.99,
        ent_coef=args['ent_coef'] if args['ent_coef'] else 0.0,
        num_episodes=args['num_of_episodes'],
        
        # Reward parameters
        gamma_dist=args['gamma'] if args['gamma'] else 0.2,
        alpha_energy=args['alpha'] if args['alpha'] else 0.5,
        beta_alert=args['beta'] if args['beta'] else 0.3,
        
        experiment_name=args['experiment_name'],
        mode=args['mode']
    )

    return experiment_config

def setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(description="Run experiment with specified mode and parameters.")
    
    # Required arguments
    parser.add_argument('mode', 
                       choices=['train', 'retrain', 'test', 'manual_policy', 'random_policy'],
                       help="Mode of operation")
    parser.add_argument('--env_name', required=True, help="Environment name")
    parser.add_argument('--num_of_episodes', required=True, type=int, 
                       help="Number of episodes")

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', default=None, help="Model type (ppo, a2c, etc.)")
    model_group.add_argument('--policy', help="Policy type")
    model_group.add_argument('--learning_rate', type=float, help="Learning rate")
    model_group.add_argument('--ent_coef', type=float, help="Entropy coefficient")
    model_group.add_argument('--df', type=float, help="Discount factor")
    model_group.add_argument('--n_steps', type=int, help="Steps between updates")

    # Environment configuration
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument('--additional_percentage', type=float, 
                          help="Additional percentage")
    env_group.add_argument('--fake_start_datetime', help="Fake start datetime")
    env_group.add_argument('--rule_frequency', type=int, default=1, 
                          help="Rule frequency")
    env_group.add_argument('--span_size', type=int, help="Span size")
    env_group.add_argument('--logs_per_minute', type=int, default=300, 
                          help="Logs per minute")
    env_group.add_argument('--search_window', type=int, default=1, 
                          help="Search window")

    # Strategy configuration
    strategy_group = parser.add_argument_group('Strategy Configuration')
    strategy_group.add_argument('--reward_calculator_version', 
                              help="Reward calculator version")
    strategy_group.add_argument('--state_strategy_version', 
                              help="State strategy version")
    strategy_group.add_argument('--action_strategy_version', 
                              help="Action strategy version")

    # Reward parameters
    reward_group = parser.add_argument_group('Reward Parameters')
    reward_group.add_argument('--alpha', type=float, help="Alpha value")
    reward_group.add_argument('--beta', type=float, help="Beta value")
    reward_group.add_argument('--gamma', type=float, help="Gamma value")

    # Other parameters
    parser.add_argument('--num_of_measurements', type=int, default=1, 
                       help="Number of measurements")
    parser.add_argument('--experiment_name', help="Experiment name")

    return parser

def run_experiment(mode: str, **kwargs):
    """Run experiment with specified mode and parameters"""
    try:
        # Initialize experiment manager
        manager = ExperimentManager(base_dir="experiments")
        
        # Create configuration from arguments
        config = create_configs_from_args(kwargs)
        
        # Log configuration
        logger.info(f"Starting {mode} experiment with configuration:")
        logger.info(f"Environment config: {asdict(config.env_config)}")
        logger.info(f"Experiment config: {asdict(config)}")
        
        # Run experiment based on mode
        if mode == 'train':
            result = manager.train_model(config)
            logger.info(f"Training completed. Model saved at: {result.model_path}")
            
        elif mode == 'retrain':
            result = manager.retrain_model(config)
            logger.info(f"Retraining completed. Model saved at: {result.model_path}")
            
        elif mode == 'test':
            result = manager.test_model(config)
            logger.info(f"Testing completed. Mean reward: {result.mean_reward}")
            
        elif mode in ['manual_policy', 'random_policy']:
            result = manager.evaluate_policy(config, policy_type=mode)
            logger.info(f"Policy evaluation completed. Mean reward: {result.mean_reward}")
            
        logger.info("Experiment completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Convert args to dict and run experiment
    kwargs = vars(args)
    mode = kwargs.pop('mode')
    
    logger.info(f"Starting {mode} experiment")
    result = run_experiment(mode, **kwargs)