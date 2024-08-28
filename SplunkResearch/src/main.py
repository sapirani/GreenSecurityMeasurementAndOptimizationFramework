import json
import os
import sys
import traceback
from experiment_new import Experiment
from experiment_manager import ExperimentManager
import logging

def prepare_experiment(base_dir, env_name, model, num_of_episodes, **kwargs):
    experiment_dir = os.path.join(base_dir, env_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    experiment = Experiment(experiment_dir)
    
    with open('./src/config.json', 'r') as fp:
        parameters = json.load(fp)
    
    experiment_kwargs = {
        'env_name': env_name,
        'model': model,
        'num_of_episodes': num_of_episodes,
        'alpha': kwargs.get('alpha', parameters['reward_parameters'].get('alpha', 0.5)),
        'beta': kwargs.get('beta', parameters['reward_parameters'].get('beta', 0.5)),
        'gamma': kwargs.get('gamma', parameters['reward_parameters'].get('gamma', 0.5)),
        'learning_rate': kwargs.get('learning_rate', 0.001),
        'policy': kwargs.get('policy', 'MlpPolicy'),
    }

    return experiment, experiment_kwargs

def run_experiment(mode, **kwargs):
    try:
        experiment, experiment_kwargs = prepare_experiment(**kwargs)        
        if mode == 'train':
            experiment_kwargs['reward_calculator_version'] = kwargs['reward_calculator_version']
            experiment_kwargs['ent_coef'] = kwargs['ent_coef']
            experiment_kwargs['df'] = kwargs['df']
            experiment.train_model(**experiment_kwargs)
        elif mode == 'retrain':
            experiment_kwargs['experiment_num'] = kwargs['experiment_num']
            experiment.retrain_model(**experiment_kwargs)
        elif mode == 'test':
            experiment_kwargs['experiment_num'] = kwargs['experiment_num']
            experiment.test_model(**experiment_kwargs)
        elif mode == 'baseline':
            experiment_kwargs['agent_type'] = kwargs['agent_type']
            experiment.test_baseline_agent(**experiment_kwargs)
        elif mode == 'no_agent':
            experiment_kwargs['experiment_num'] = kwargs['experiment_num']
            experiment.test_no_agent(**experiment_kwargs)
    
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    base_dir = "experiments__"
    experiment_manager = ExperimentManager(base_dir=base_dir, log_level=logging.DEBUG) 
    
    mode = sys.argv[1]
    print(f'##########################################################################start {mode}##########################################################################')
    print('##########################################################################\n##########################################################################')
    
    kwargs = {
        'base_dir': base_dir,
        'env_name': sys.argv[2],
        'model': sys.argv[3],
        'num_of_episodes': int(sys.argv[4]),
        'alpha': float(sys.argv[5]) if len(sys.argv) > 5 else None,
        'beta': float(sys.argv[6]) if len(sys.argv) > 6 else None,
        'gamma': float(sys.argv[7]) if len(sys.argv) > 7 else None,
        'learning_rate': float(sys.argv[8]) if len(sys.argv) > 8 else None,
        'policy': sys.argv[9] if len(sys.argv) > 9 else None,
        'reward_calculator_version': sys.argv[10] if len(sys.argv) > 10 else None, # 'reward_calc_1' or 'reward_calc_2
        'experiment_num': sys.argv[10] if len(sys.argv) > 10 else None,
        'ent_coef': float(sys.argv[11]) if len(sys.argv) > 11 else None,
        'agent_type': sys.argv[11] if len(sys.argv) > 11 else None,
        'df': float(sys.argv[12]) if len(sys.argv) > 12 else None,
    }
    
    run_experiment(mode, **kwargs)
