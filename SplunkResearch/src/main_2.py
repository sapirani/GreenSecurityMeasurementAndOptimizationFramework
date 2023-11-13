import datetime
import json
import sys
from datetime import datetime
from eperiment import Experiment
from experiment_manager import ExperimentManager
if __name__ == "__main__":
    print('##########################################################################start##########################################################################')
    print('##########################################################################\n##########################################################################')
    mode = sys.argv[1]
    experiment_manager = ExperimentManager() 
    
    if mode == 'train':
        experiment_dir = experiment_manager.create_experiment_dir()
        logger = experiment_manager.setup_logging(f"{experiment_dir}/log.txt")
        experiment = Experiment(experiment_dir, logger)
        with open(f'./src/config.json', 'r') as fp:
            parameters = json.load(fp)
        experiment.train_model(parameters)
        experiment_manager.save_experiment(experiment, experiment_dir)
        
    elif mode == 'test':
        print(sys.argv)
        if sys.argv[2] == 'last':
            experiment_dir = experiment_manager.get_last_experiment_dir()
        else:
            experiment_dir = sys.argv[2]
        experiment = experiment_manager.load_experiment(experiment_dir)
        logger = experiment_manager.setup_logging(f"{experiment_dir}/log_test.txt")
        experiment.logger = logger
        num_of_episodes = int(sys.argv[3])
        experiment.test_model(num_of_episodes)
        
    elif mode == 'baseline':
        if sys.argv[2] == 'last':
            experiment_dir = experiment_manager.get_last_experiment_dir()
        else:
            experiment_dir = sys.argv[2]
        experiment = experiment_manager.load_experiment(experiment_dir)
        logger = experiment_manager.setup_logging(f"{experiment_dir}/log_baseline.txt")
        experiment.logger = logger
        num_of_episodes = int(sys.argv[3])
        experiment.test_baseline_agent(num_of_episodes, agent_type=sys.argv[4])
    

