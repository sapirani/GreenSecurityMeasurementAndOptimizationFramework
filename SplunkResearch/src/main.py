import datetime
import json
import sys
from datetime import datetime
import traceback
from experiment import Experiment
from experiment_manager import ExperimentManager
import logging

def evaluation_preperation(experiment_manager, mode='no_agent'):
    if sys.argv[2] == 'last':
        experiment_dir = experiment_manager.get_last_experiment_dir()
    else:
        experiment_dir = sys.argv[2]
    experiment = experiment_manager.load_experiment(experiment_dir)
    log_file = f'{experiment_dir}/log_{mode}.txt'
    logger = experiment_manager.setup_logging(log_file)
    experiment.logger = logger
    num_of_episodes = int(sys.argv[3])
    return log_file,experiment,num_of_episodes

if __name__ == "__main__":
    print('##########################################################################start##########################################################################')
    print('##########################################################################\n##########################################################################')
    mode = sys.argv[1]
    experiment_manager = ExperimentManager(log_level=logging.DEBUG) 
    try:
        if mode == 'train':
            experiment_manager.delete_experiments_without_train()
            experiment_dir = experiment_manager.create_experiment_dir()
            log_file = f'{experiment_dir}/log_train.txt'
            logger = experiment_manager.setup_logging(log_file)
            experiment = Experiment(experiment_dir, logger)
            with open(f'./src/config.json', 'r') as fp:
                parameters = json.load(fp)
            experiment.train_model(parameters)
            experiment_manager.save_experiment(experiment, experiment_dir)
            
        if mode == 'retrain':
            if sys.argv[2] == 'last':
                experiment_dir = experiment_manager.get_last_experiment_dir()
            else:
                experiment_dir = sys.argv[2]
            log_file = f'{experiment_dir}/log_retrain.txt'
            logger = experiment_manager.setup_logging(log_file)
            experiment = experiment_manager.load_experiment(experiment_dir)
            experiment.logger = logger
            with open(f'./src/config.json', 'r') as fp:
                parameters = json.load(fp)
            experiment.retrain_model(parameters)
            experiment_manager.save_experiment(experiment, experiment_dir)
            
        elif mode == 'test':
            log_file, experiment, num_of_episodes = evaluation_preperation(experiment_manager, mode)
            experiment.test_model(num_of_episodes)
            
        elif mode == 'baseline':
            log_file, experiment, num_of_episodes = evaluation_preperation(experiment_manager, mode)
            experiment.test_baseline_agent(num_of_episodes, agent_type=sys.argv[4])
            
        elif mode == 'no_agent':
            log_file, experiment, num_of_episodes = evaluation_preperation(experiment_manager, mode='no_agent')
            experiment.test_no_agent(num_of_episodes)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    # finally:
    #     experiment_manager.send_email(log_file)

