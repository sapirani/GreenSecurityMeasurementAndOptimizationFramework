from dataclasses import dataclass, replace
import inspect
from operator import is_
import ssl
from typing import Dict, Any, Optional, List
import sb3_contrib
import stable_baselines3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import custom_splunk #dont remove!!!
from custom_splunk.envs.custom_splunk_env import SplunkConfig
import gymnasium as gym
from gymnasium import register, spaces, make
# from gymnasium.vector import VecNormalize, DummyVecEnv
import pandas as pd
import numpy as np
import os
import logging
import datetime
from pathlib import Path
import json
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import A2C, PPO, DQN, DDPG, TD3, SAC
from stable_baselines3.ppo.policies import MlpPolicy
from sb3_contrib import RecurrentPPO

from wrappers.reward import *
from wrappers.state import *
from wrappers.action import *
from callbacks import *
from time_manager import TimeWrapper
import smtplib
from email.message import EmailMessage
from stable_baselines3.common.logger import configure

logger = logging.getLogger(__name__)
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Dropout regularization
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Environment config
    env_config: SplunkConfig
    
    # Training config
    model_type: str = "ppo"  # ppo, a2c, dqn, etc.
    policy_type: str = "mlp"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    # batch_size: int = 64
    # n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.0
    num_episodes: int = 1000
    is_mock: bool = False
    # Reward components
    use_distribution_reward: bool = True
    use_energy_reward: bool = True
    use_alert_reward: bool = True
    use_quota_violation: bool = True
    use_random_agent: bool = False
    action_type: str = "Action8"  # Action, SingleAction, Action8
    model_path: Optional[str] = None  # Path to load model from, if not training
    
    # Reward parameters
    gamma_dist: float = 0.2
    alpha_energy: float = 0.5
    beta_alert: float = 0.3
    
    # Experiment metadata
    experiment_name: Optional[str] = None
    mode: str = "train"  # train, eval, retrain

class ExperimentManager:
    """Manages training and evaluation experiments"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self._setup_directories()
        self.experiments_db = self._load_experiments_db()
        self.eval_env = None

    def _setup_directories(self):
        """Create necessary directories"""
        dirs = {
            'train': self.base_dir / 'train',
            'eval': self.base_dir / 'eval',
            'models': self.base_dir / 'models',
            'logs': self.base_dir / 'logs',
            'tensorboard': self.base_dir / 'tensorboard',
            'baseline': self.base_dir / 'baseline'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.dirs = dirs

    def _load_experiments_db(self) -> pd.DataFrame:
        """Load or create experiments database"""
        db_path = self.base_dir / 'experiments.csv'
        if db_path.exists():
            return pd.read_csv(db_path)
        return pd.DataFrame(columns=[
            'experiment_id', 'name', 'mode', 'start_time', 'end_time',
            'config', 'status', 'metrics'
        ])

    def _save_experiments_db(self):
        """Save experiments database"""
        self.experiments_db.to_csv(
            self.base_dir / 'experiments.csv',
            index=False
        )

    def create_environment(self, config: ExperimentConfig) -> gym.Env:
        """Create and configure environment with reward wrappers"""

        top_logtypes = pd.read_csv("/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/resources/top_logtypes.csv")
        # include only system and security logs
        top_logtypes = top_logtypes[top_logtypes['source'].str.lower().isin(['wineventlog:security', 'wineventlog:system'])]
        top_logtypes = top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:20]
        top_logtypes = [(x[0].lower(), str(x[1])) for x in top_logtypes]
        if "test_experiment" in  config.experiment_name:
            config.env_config.is_test = True
        env = make(
            id=config.env_config.env_id,
            config=config.env_config,
            top_logtypes=top_logtypes
        )
        # env = SingleAction2(env)
        # env = Action(env)
        # env = SngleAction(env, config.use_random_agent)
        if config.action_type == "Action8":        
            env = Action8(env, config.use_random_agent)
        elif config.action_type == "Action11":
            env = Action11(env, config.use_random_agent)
        

        # if config.use_random_agent:
        #     env = RandomAction(env)

        # Add reward wrappers
        if config.use_distribution_reward:
            env = DistributionRewardWrapper(
                env,
                gamma=config.gamma_dist,
                epsilon=1e-8,
                distribution_freq=1
            )
            
        env = BaseRuleExecutionWrapperWithPrediction(env, baseline_dir=self.dirs['baseline'], is_mock=config.is_mock, enable_prediction = True, alert_threshold = 2.5, skip_on_low_alert = True, use_energy = config.use_energy_reward, use_alert = config.use_alert_reward, is_train = (config.mode.__contains__('train')) , is_eval = (config.mode == "eval_post_training"))  
        # alert_threshold = -6, -2, -10
        
        if config.use_energy_reward:
                
            env = EnergyRewardWrapper(
                env,
                alpha=config.alpha_energy,
               is_mock=config.is_mock,
            )
            # env = AlertRewardWrapper(
            #     env,
            #     beta=config.beta_alert,
            #     epsilon=1e-3,
            #     is_mock=config.is_mock
            # )

            
        
        # if config.use_alert_reward and not config.use_energy_reward:
        #     env = BaseRuleExecutionWrapperWithPrediction(env, baseline_dir=self.dirs['baseline'], is_mock=config.is_mock)
        #     env = AlertRewardWrapper(
        #         env,
        #         beta=config.beta_alert,
        #         epsilon=1e-3,
        #         is_mock=config.is_mock
        #     )

            # env = AlertRewardWrapper1(
            #     env,
            #     beta=config.beta_alert,
            #     epsilon=1e-3
            # )
        # env = ClipRewardWrapper(env,
        #                         low=0,
        #                         high=1)
        env = TimeWrapper(env)
        env = StateWrapper3(env)

        return env


    def create_model(self, config: ExperimentConfig, env: gym.Env):
        """Create or load model based on config"""
        if config.mode == "train":
            return self._create_new_model(config, env)
        else:
            return self._load_existing_model(config, env)
        
    def _get_model_class(self, model_type: str):
        """Get model class based on type"""
        if model_type == "ppo":
            return PPO
        elif model_type == "a2c":
            return A2C
        elif model_type == "dqn":
            return DQN
        elif model_type == "sac":
            return SAC
        elif model_type == "recurrent_ppo":
            return RecurrentPPO
        elif model_type == "td3":
            return TD3
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_policy_class(self, policy_type: str):
        """Get policy class based on type"""
        if policy_type == "mlp":
            return MlpPolicy
        elif policy_type == "td3_mlp":
            return TD3.MlpPolicy
        elif policy_type == "MlpLstmPolicy":
            return sb3_contrib.ppo_recurrent.MlpLstmPolicy
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    def _create_new_model(self, config: ExperimentConfig, env: gym.Env):
        """Create new model instance"""
        model_cls = self._get_model_class(config.model_type)
        # decay learning rate
        model_kwargs = {
            'env': env,
            'policy': config.policy_type,
            'learning_rate': config.learning_rate,
            'gamma': config.gamma,
            'tensorboard_log': f"{str(self.dirs['tensorboard'])}/{config.experiment_name}",
            'stats_window_size': 5,
            'verbose': 0
        }
        
        if config.model_type in ['recurrent_ppo', 'ppo', 'a2c']:
            model_kwargs.update({
                'n_steps': config.n_steps,
                # 'batch_size': config.batch_size,
                # 'n_epochs': config.n_epochs,
                'ent_coef': config.ent_coef,
                'sde_sample_freq': 24,
                'use_sde': True,
                "policy_kwargs": {
                    "net_arch": [512, 128, 128, 64],
                    "features_extractor_class": CustomExtractor,
                },
            })
            
        elif config.model_type in ['sac', 'td3', 'ddpg']:
            model_kwargs.update({                
                #     "policy_kwargs": {
                #     "net_arch": [512,128,128,64],


                # },
                # "tau": 0.01,
                # "train_freq":(3),
                # 'action_noise':stable_baselines3.common.noise.NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.2 * np.ones(env.action_space.shape[0]))
                })
      
            
        return model_cls(**model_kwargs)
    
    def _generate_experiment_id(self):
        """Generate unique experiment ID"""
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    def _setup_experiment_logging(self, experiment_name: str):
        """Setup logging for experiment"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
            handlers=[
                logging.FileHandler(self.dirs['logs'] / f"{experiment_name}.log"),
                # logging.StreamHandler()
            ]
        )
    
    def run_experiment(self, config: ExperimentConfig):
        """Run experiment based on configuration"""
        # Generate experiment ID and name
        experiment_id = self._generate_experiment_id()
        if config.experiment_name is None:
            experiment_name = f"{config.mode}_{experiment_id}"
        else:
            experiment_name = f"{config.experiment_name}_{experiment_id}"
        config.experiment_name = experiment_name
        
        # Setup logging
        self._setup_experiment_logging(experiment_name)
        logger.info(f"Starting experiment: {experiment_name}")
        
        # Record experiment start
        self._record_experiment_start(experiment_id, experiment_name, config)
        
        try:
            # Create environment and model
            env = self.create_environment(config)
            model = self.create_model(config, env)
            # create eval env
            if "eval" not in config.mode:
                eval_config = replace(config, mode="eval")
            else:
                eval_config = config
            eval_config.env_config.env_id = "splunk_eval-v32"
            eval_config.env_config.rule_frequency = 60 #2880
            eval_config.env_config.end_time = "04/26/2025:23:59:59"
            # eval_config = replace(eval_config, is_mock=False)
            self.eval_env = self.create_environment(eval_config)
            self.eval_env.unwrapped.splunk_tools.load_real_logs_distribution_bucket(datetime.datetime.strptime(env.unwrapped.time_manager.first_start_datetime, '%m/%d/%Y:%H:%M:%S'), datetime.datetime.strptime(self.eval_env.unwrapped.time_manager.end_time, '%m/%d/%Y:%H:%M:%S'))

            # if "test_experiment" not  in config.experiment_name:
            #     # clean and warm up the env
            #     clean_env(env.unwrapped.splunk_tools, (env.unwrapped.time_manager.first_start_datetime, datetime.datetime.now().strftime("%m/%d/%Y:%H:%M:%S")))
            #     env.unwrapped.warmup()
            # else:
            #     env.unwrapped.disable_injection()
            #     self.eval_env.unwrapped.disable_injection()
            # Setup callbacks
            config.experiment_name = experiment_name
            callbacks = self._setup_callbacks(config)
            
            # Run experiment
            if config.mode == "train":
                results = self._run_training(model, env, config, callbacks)
            elif config.mode == "eval_post_training":  # eval after training
                results = self._run_evaluation(model, self.eval_env, eval_config)
            else:  # retrain
                results = self._run_retraining(model, env, config, callbacks)
                
            # Record success
            self._record_experiment_end(experiment_id, "completed", results)
            
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            self._record_experiment_end(experiment_id, "failed", {"error": str(e)})
            # Send email notification
            self.send_email(error_message=str(e))
            
            raise

    def _run_training(self, model, env, config, callbacks):
        """Run training experiment"""
        # total_timesteps = env.get_attr('total_steps')[0] * config.num_episodes
        total_timesteps = env.unwrapped.total_steps * config.num_episodes
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=config.experiment_name
        )
        
        # Save model
        model_path = self.dirs['models'] / f"{config.experiment_name}.zip"
        model.save(str(model_path))
        
        return {
            "model_path": str(model_path),
            "total_timesteps": total_timesteps
        }

    def _run_evaluation(self, model, env, config):
        """evaluate the model for a specific number of episodes. Create summary writers for the evaluation """
        model.set_env(env)
        eval_episodes = config.num_episodes
        log_dir = f"{self.dirs['tensorboard']._str}/{config.experiment_name}"
        eval_logger = configure(log_dir, ["tensorboard"])
        model.set_logger(eval_logger)
        rules = self.eval_env.unwrapped.splunk_tools.active_saved_searches.keys()
        event_types = [f"{x[0].lower()}_{x[1]}" for x in self.eval_env.unwrapped.top_logtypes]
        writers = self.create_summary_writers(log_dir, rules, event_types)
        eval_callback = CustomEvalCallback3(
            eval_env=self.eval_env,
            log_dir=f"{self.dirs['tensorboard']._str}/{config.experiment_name}", rules=rules, event_types=event_types,
            n_eval_episodes=1,
            eval_freq=1,
            best_model_save_path=self.dirs['models'],
            log_path=self.dirs['logs'],
            # eval_log_dir=str(self.dirs['tensorboard']/f"eval_{config.experiment_name}"),
            deterministic=False,
            render=False,
            verbose=1,
            writers=writers,
        )
        eval_callback.model = model
        for _ in range(eval_episodes):
            eval_callback.on_step()

    def create_summary_writers(self, log_dir, rules, event_types):
        writers = {
        rule: SummaryWriter(log_dir=log_dir + f"/{rule}") for rule in rules
        }
        writers.update({
            event_type: SummaryWriter(log_dir=log_dir + f"/{event_type.replace(':','_')}") for event_type in event_types
        })
        writers.update({
            f"{event_type}_{is_trigger}": SummaryWriter(log_dir=log_dir + f"/{event_type.replace(':','_')}_{is_trigger}")   for event_type in event_types  for is_trigger in  [0,1]
        })

        return writers



    def _load_existing_model(self, config: ExperimentConfig, env: gym.Env):
        """Load model from path"""
        model_cls = self._get_model_class(config.model_type)
        model = model_cls.load(config.model_path, env=env)
        logger.info(f"Loaded model from {config.model_path}")
        
        return model
        
        
    def _run_retraining(self, model, env, config, callbacks):
        """Run retraining experiment"""
        total_timesteps = env.unwrapped.total_steps * config.num_episodes
              
        # model = self.load_model(config.model_path, env)

         
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=config.experiment_name
        )
        model_path = self.dirs['models'] / f"{config.experiment_name}.zip"
        model.save(str(model_path))
        
        return {
            "model_path": str(model_path),
            "total_timesteps": total_timesteps
        }
    
    def _record_experiment_start(self, experiment_id: str, name: str, 
                               config: ExperimentConfig):
        """Record experiment start in database"""
        serialized_config = config.__dict__.copy()
        serialized_config['learning_rate'] = serialized_config['learning_rate'](1.0) if callable(serialized_config['learning_rate']) else serialized_config['learning_rate']
        serialized_config['env_config'] = serialized_config['env_config'].__dict__
        new_row = {
            'experiment_id': experiment_id,
            'name': name,
            'mode': config.mode,
            'start_time': datetime.datetime.now().isoformat(),
            'config': json.dumps(serialized_config),
            'status': 'running',
            'metrics': None
        }
        
        self.experiments_db = pd.concat([
            self.experiments_db,
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        self._save_experiments_db()
        

    def _record_experiment_end(self, experiment_id: str, status: str, 
                             metrics: Dict[str, Any]):
        """Record experiment completion in database"""
        idx = self.experiments_db['experiment_id'] == experiment_id
        self.experiments_db.loc[idx, 'status'] = status
        self.experiments_db.loc[idx, 'end_time'] = datetime.datetime.now().isoformat()
        self.experiments_db.loc[idx, 'metrics'] = json.dumps(metrics)
        
        self._save_experiments_db()

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results for a specific experiment"""
        experiment = self.experiments_db[
            self.experiments_db['experiment_id'] == experiment_id
        ].iloc[0]
        
        return {
            'name': experiment['name'],
            'mode': experiment['mode'],
            'start_time': experiment['start_time'],
            'end_time': experiment['end_time'],
            'status': experiment['status'],
            'metrics': json.loads(experiment['metrics']) if experiment['metrics'] else None,
            'config': json.loads(experiment['config'])
        }
    
    def _setup_callbacks(self, config: ExperimentConfig):
        """Setup training/evaluation callbacks"""
        log_dir = f"{self.dirs['tensorboard']._str}/{config.experiment_name}"
    
        rules = self.eval_env.unwrapped.splunk_tools.active_saved_searches.keys()
        event_types = [f"{x[0].lower()}_{x[1]}" for x in self.eval_env.unwrapped.top_logtypes]
        writers = self.create_summary_writers(log_dir, rules, event_types)
        return [
            CustomTensorboardCallback(log_dir=f"{self.dirs['tensorboard']._str}/{config.experiment_name}", rules=rules, event_types=event_types, writers=writers),
            # HParamsCallback(
            #     experiment_kwargs=config,
            #     phase=config.get('phase', 'train')
            # ),
            CheckpointCallback(save_freq=10000, save_path=self.dirs['models'], name_prefix=config.experiment_name),
            
            CustomEvalCallback3(
                eval_env=self.eval_env,
                log_dir=f"{self.dirs['tensorboard']._str}/{config.experiment_name}", rules=rules, event_types=event_types,
                n_eval_episodes=3,
                eval_freq=9600,
                best_model_save_path=self.dirs['models'],
                log_path=self.dirs['logs'],
                # eval_log_dir=str(self.dirs['tensorboard']/f"eval_{config.experiment_name}"),
                deterministic=False,
                render=False,
                verbose=1,
                writers=writers,
            ), 
            # SplunkLincenceCheckCallback()
            
        ]
        
    def send_email(self, error_message="Experiment has failed", log_file=None):
        my_email = os.getenv('EMAIL')
        email_password = os.getenv('EMAIL_PASSWORD')
        msg = EmailMessage()
        msg['Subject'] = 'Experiment has failed' 
        msg['From'] = my_email
        msg['To'] = my_email
        msg.set_content
        msg.set_content(f"The experiment has failed. \n Error message: {error_message}")
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(my_email, email_password)
            smtp.send_message(msg)

# def lr_schedule(initial_value: float, rate: float):
#     """
#     Learning rate schedule:
#         Exponential decay by factors of 10

#     :param initial_value: Initial learning rate.
#     :param rate: Exponential rate of decay. High values mean fast early drop in LR
#     :return: schedule that computes
#       current learning rate depending on remaining progress
#     """
#     def func(progress_remaining: float) -> float:
#         """
#         Progress will decrease from 1 (beginning) to 0.

#         :param progress_remaining:
#         :return: current learning rate
#         """
#         if progress_remaining <= 0:
#             return 1e-9
        
#         return initial_value * 10 ** (rate * np.log(progress_remaining))

#     return func


 
if __name__ == "__main__":
    # Create experiment config
    retrain_fake_start_datetime = "08/01/2024:00:00:00"
    # model_path = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/train_20250611165220_102000_steps.zip"
    # model_path = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/train_20250620175311_35000_steps"
    # model_path = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/test_experiment_20250623144601_43000_steps.zip"
    # model_path = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/train_20250711001649_264000_steps.zip"
    num_episodes = 2000000
    action_type = "Action8"
    for steps in range(45000, 160000, 500000):
        # model_path = f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/train_20250626010440_{steps}_steps.zip"
        # model_path = f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/train_20250726233927_243000_steps.zip"
        # model_path = f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/test_experiment_20250806144736_355000_steps.zip"
        model_path = f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/train_20250815121629_990000_steps.zip"
        print(f"Model path: {model_path}")
        for learning_rate in [0.0001]:
            for n_steps in [256]:
                for ent_coef in [0.1]:
                    for is_random in [False]:
                        lr = 1e-2
                        env_config = SplunkConfig(
                            # fake_start_datetime=retrain_fake_start_datetime,
                            rule_frequency=60, #600,
                            search_window=2880,
                            # savedsearches=["rule1", "rule2"],
                            logs_per_minute=150,
                            additional_percentage=1,
                            action_duration=7200, 
                            num_of_measurements=1,
                            baseline_num_of_measurements=2,
                            env_id="splunk_train-v32",
                            end_time="12/31/2024:23:59:59"       
                        )
                        # sched_LR = lr_schedule(initial_value = 0.01, rate = 5)
                        experiment_config = ExperimentConfig(
                            env_config=env_config,
                            model_type="recurrent_ppo",# "ppo", # "a2c", "dqn", "sac", "td3", "recurrent_ppo"
                            policy_type="MlpLstmPolicy",# "td3_mlp", # "mlp"
                            learning_rate=learning_rate,#sched_LR,
                            num_episodes=num_episodes,
                            n_steps=n_steps,
                            ent_coef=ent_coef,
                            gamma=1,
                            gamma_dist=1,#0.33,
                            alpha_energy=1,
                            beta_alert=1,
                            action_type=action_type,
                            # experiment_name="test_experiment",
                            use_alert_reward=True,
                            use_energy_reward=True,
                            use_random_agent=is_random,
                            is_mock=True,
                            model_path=model_path if model_path else None,
                            
                        )
                        
                        #retrain model
                        experiment_config.mode = "train"#"eval_post_training"  # eval after training
                        manager = ExperimentManager(base_dir="experiments")
                        results = manager.run_experiment(experiment_config)

                        
                    
                