import json
import os
import sys
import traceback
from experiment_new import Experiment
from experiment_manager_new import ExperimentManager
import logging
import argparse

def run_experiment(mode, **kwargs):
    try:
        # Define the base directory for experiments
        base_dir = "experiments___"
        
        # Initialize the experiment manager
        experiment_manager = ExperimentManager(base_dir=base_dir, log_level=logging.DEBUG)
        
        # Call the appropriate method based on the mode
        if mode == 'train':
            experiment_manager.train_model(**kwargs)
        elif mode == 'retrain':
            experiment_manager.retrain_model(**kwargs)
        elif mode == 'test':
            experiment_manager.test_model(**kwargs)
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
    parser.add_argument('mode', type=str, choices=['train', 'retrain', 'test'], help="Mode of operation: 'train', 'retrain', or 'test'.")
    parser.add_argument('--env_name', type=str, help="Environment name.")
    parser.add_argument('--model', type=str, help="Model name.")
    parser.add_argument('--num_of_episodes', type=int, help="Number of episodes.")

    # Add optional arguments
    parser.add_argument('--alpha', type=float, help="Alpha value for training.")
    parser.add_argument('--beta', type=float, help="Beta value for training.")
    parser.add_argument('--gamma', type=float, help="Gamma value for training.")
    parser.add_argument('--learning_rate', type=float, help="Learning rate.")
    parser.add_argument('--policy', type=str, help="Policy type, e.g., 'MlpPolicy'.")
    parser.add_argument('--reward_calculator_version', type=str, help="Reward calculator version, e.g., 'reward_calc_1'.")
    parser.add_argument('--ent_coef', type=float, help="Entropy coefficient.")
    # parser.add_argument('--agent_type', type=str, help="Type of agent for baseline testing.")
    parser.add_argument('--df', type=float, help="Discount factor.")
    parser.add_argument('--additional_percentage', type=float, help="Additional percentage.")
    parser.add_argument('--fake_start_datetime', type=str, help="Fake start datetime for simulations.")
    parser.add_argument('--rule_frequency', type=int, default=1, help="Rule frequency, default is 1.")
    parser.add_argument('--span_size', type=int, help="Span size.")
    parser.add_argument('--logs_per_minute', type=int, default=300, help="Logs per minute, default is 300.")
    parser.add_argument('--num_of_measurements', type=int, default=1, help="Number of measurements, default is 1.")
    parser.add_argument('--search_window', type=int, default=1, help="Search window, default is 1.")

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
