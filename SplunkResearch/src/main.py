import json
import os
import sys
import traceback
from experiment_manager_new import ExperimentManager
import logging
import argparse

def run_experiment(mode, **kwargs):
    try:
        # Define the base directory for experiments
        base_dir = "experiments_____"
        
        # Initialize the experiment manager
        experiment_manager = ExperimentManager(base_dir=base_dir, log_level=logging.INFO)
        if 'fake_start_datetime' in kwargs and kwargs['fake_start_datetime'] is not None:
            kwargs['fake_start_datetime'] = kwargs['fake_start_datetime'].replace('_', ' ')
        # Call the appropriate method based on the mode
        if mode == 'train':
            experiment_manager.train_model(**kwargs)
        elif mode == 'retrain':
            experiment_manager.retrain_model(**kwargs)
        elif mode == 'test':
            experiment_manager.test_model(**kwargs)
        elif mode == 'manual_policy':
            experiment_manager.manual_policy_eval(**kwargs)    
        elif mode == 'random_policy':
            experiment_manager.random_policy_eval(**kwargs)    
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    except Exception as e:
        # Log and print any exceptions
        logger = logging.getLogger(__name__)
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run experiment with specified mode and parameters.")

    # Add required positional arguments
    parser.add_argument('mode', type=str, choices=['train', 'retrain', 'test', 'manual_policy','random_policy'], help="Mode of operation: 'train', 'retrain', or 'test'.")
    parser.add_argument('--env_name', type=str, help="Environment name.")
    parser.add_argument('--num_of_episodes', type=int, help="Number of episodes.")

    # Add optional arguments
    parser.add_argument('--model', default=None, type=str, help="Model name.")
    parser.add_argument('--alpha', type=float, help="Alpha value for training.")
    parser.add_argument('--beta', type=float, help="Beta value for training.")
    parser.add_argument('--gamma', type=float, help="Gamma value for training.")
    parser.add_argument('--learning_rate', default=None, type=float, help="Learning rate.")
    parser.add_argument('--policy', type=str, help="Policy type, e.g., 'MlpPolicy'.")
    parser.add_argument('--reward_calculator_version', type=str, help="Reward calculator version, e.g., '1'.")
    parser.add_argument('--state_strategy_version', type=str, help="State strategy version, e.g., '1'.")
    parser.add_argument('--action_strategy_version', type=str, help="Action strategy version, e.g., '1'.")
    parser.add_argument('--ent_coef', default=None, type=float, help="Entropy coefficient.")
    # parser.add_argument('--agent_type', type=str, help="Type of agent for baseline testing.")
    parser.add_argument('--df', default=None, type=float, help="Discount factor.")
    parser.add_argument('--additional_percentage', type=float, help="Additional percentage.")
    parser.add_argument('--fake_start_datetime', default=None,type=str, help="Fake start datetime for simulations.")
    parser.add_argument('--rule_frequency', type=int, default=1, help="Rule frequency, default is 1.")
    parser.add_argument('--span_size', type=int, help="Span size.")
    parser.add_argument('--logs_per_minute', type=int, default=300, help="Logs per minute, default is 300.")
    parser.add_argument('--num_of_measurements', type=int, default=1, help="Number of measurements, default is 1.")
    parser.add_argument('--search_window', type=int, default=1, help="Search window, default is 1.")
    parser.add_argument('--experiment_name', type=str, help="Experiment name.")
    parser.add_argument('--n_steps', type=int, help="Number of steps between each update.")

    # Parse the arguments
    args = parser.parse_args()

    # Convert the argparse Namespace to a dictionary and extract the mode
    mode = args.mode
    kwargs = vars(args)
    kwargs.pop('mode')  # Remove mode from kwargs as it's not part of the experiment parameters

    # Start the experiment
    print(f'##########################################################################')
    print(f'start {mode}')
    print(f'##########################################################################\n##########################################################################')
    
    run_experiment(mode, **kwargs)
