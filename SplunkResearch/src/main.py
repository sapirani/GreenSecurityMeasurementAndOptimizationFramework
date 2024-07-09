import json
import os
import sys
import traceback
from experiment_new import Experiment
from experiment_manager import ExperimentManager
import logging

# def evaluation_preperation(experiment_manager, experiment_dir):
#     # if sys.argv[2] == 'last':
#     #     experiment_dir = experiment_manager.get_last_experiment_dir()
#     # else:
#     #     experiment_dir = sys.argv[2]
#     if not os.path.exists(experiment_dir):
#         os.makedirs(experiment_dir)
#     log_file = f'{experiment_dir}/log.txt'
#     experiment_manager.setup_logging(log_file)
#     num_of_episodes = int(sys.argv[3])
#     return log_file, num_of_episodes, experiment_dir

if __name__ == "__main__":
    print('##########################################################################start##########################################################################')
    print('##########################################################################\n##########################################################################')
    base_dir = "experiments__"
    experiment_manager = ExperimentManager(base_dir=base_dir, log_level=logging.DEBUG) 
    mode = sys.argv[1]
    env_name = sys.argv[2]
    experiment_dir = os.path.join(base_dir, env_name)
    if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
    experiment = Experiment(experiment_dir)
    
    model = sys.argv[3]
    num_of_episodes = int(sys.argv[4])

    with open(f'./src/config.json', 'r') as fp:
        parameters = json.load(fp)
    try:
        alpha = float(sys.argv[5])
        beta = float(sys.argv[6])
        gamma = float(sys.argv[7])
        learning_rate = float(sys.argv[8])
        if mode == 'train':
            experiment.train_model(env_name, model, num_of_episodes, alpha=alpha, beta=beta, gamma=gamma, learning_rate=learning_rate)
        elif mode == 'retrain':
            experiment.retrain_model(env_name, model, num_of_episodes, alpha=alpha, beta=beta, gamma=gamma, learning_rate=learning_rate)
        elif mode == 'test':
            experiment.test_model(env_name, model, num_of_episodes, alpha=alpha, beta=beta, gamma=gamma, learning_rate=learning_rate)
        elif mode == 'baseline':
            experiment.test_baseline_agent(env_name, model, num_of_episodes, alpha=alpha, beta=beta, gamma=gamma, learning_rate=learning_rate, agent_type=sys.argv[9])
        elif mode == 'no_agent':
            experiment.test_no_agent(env_name, model, num_of_episodes, alpha=alpha, beta=beta, gamma=gamma, learning_rate=learning_rate)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
    # try:
    #     if mode == 'train':
    #         # experiment_manager.delete_experiments_without_train()
    #         with open(f'./src/config.json', 'r') as fp:
    #             parameters = json.load(fp)
    #         alpha, beta, gamma = parameters['reward_parameters'].values()
    #         experiment_dir = experiment_manager.create_experiment_dir(env_name)
    #         path = f'{experiment_dir}/{model}_{alpha}_{beta}_{gamma}/{mode}'
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         log_file = f'{path}/log_train.txt'
    #         experiment_manager.setup_logging(log_file)
    #         experiment = Experiment(path)

    #         experiment.train_model(parameters, env_name, model, num_of_episodes)
    #         # experiment_manager.save_experiment(experiment, experiment_dir)
            
    #     if mode == 'retrain':
    #         # if sys.argv[2] == 'last':
    #         #     experiment_dir = experiment_manager.get_last_experiment_dir()
    #         # else:
    #         #     experiment_dir = sys.argv[2]
    #         log_file = f'{experiment_dir}/log_retrain.txt'
    #         logger = experiment_manager.setup_logging(log_file)
    #         with open(f'./src/config.json', 'r') as fp:
    #             parameters = json.load(fp)
    #         experiment.retrain_model(parameters)
            
    #     elif mode == 'test':
    #         experiment_dir = os.path.join(base_dir, env_name, model)
    #         log_file, num_of_episodes, experiment_dir = evaluation_preperation(experiment_manager, experiment_dir)
    #         experiment = Experiment(experiment_dir)
    #         experiment.test_model(num_of_episodes)
            
    #     elif mode == 'baseline':
    #         log_file, num_of_episodes, experiment_dir = evaluation_preperation(experiment_manager, mode)
    #         experiment = Experiment(experiment_dir)
    #         experiment.test_baseline_agent(num_of_episodes, agent_type=sys.argv[4])
            
    #     elif mode == 'no_agent':
    #         log_file, num_of_episodes, experiment_dir = evaluation_preperation(experiment_manager, mode)
    #         experiment = Experiment(experiment_dir)
    #         experiment.test_no_agent(num_of_episodes)
            
    # except Exception as e:
    #     logger = logging.getLogger(__name__)
    #     logger.error(f"An error occurred: {e}")
    #     traceback.print_exc()
    # # finally:
    # #     experiment_manager.send_email(log_file)

