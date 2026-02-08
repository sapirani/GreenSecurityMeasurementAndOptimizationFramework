"""
ExperimentManager: Core experiment orchestration for DRL-based Green Security Optimization.

This module provides the ExperimentManager class which orchestrates the complete lifecycle
of reinforcement learning experiments including:
- Environment and model creation
- Training, evaluation, and retraining workflows
- Callback management and logging
- Experiment metadata tracking
- Error handling and notifications

For CLI usage, see run_experiment.py
"""
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
# from energy_profile_final import handle_process_output
logger = logging.getLogger(__name__)
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from dotenv import load_dotenv
from env_utils import get_system_monitor_path, get_security_monitor_path, clean_env, empty_monitored_files
from config import config

load_dotenv(config.get('paths.env_file'))


class CustomExtractor(BaseFeaturesExtractor):
    """Custom feature extractor with configurable architecture."""

    def __init__(self, observation_space, features_dim=None):
        """Initialize feature extractor with dynamic architecture from config.

        Args:
            observation_space: Gymnasium observation space
            features_dim: Output dimension (default: from config)
        """
        # Get configuration values
        if features_dim is None:
            features_dim = config.get('model.features_dim', 128)

        super().__init__(observation_space, features_dim)

        # Get architecture from config
        hidden_layers = config.get('model.hidden_layers', [64, 128])
        dropout_rate = config.get('model.dropout_rate', 0.2)

        # Build network dynamically
        layers = []
        input_dim = observation_space.shape[0]

        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_layers) - 1:  # Dropout between hidden layers
                layers.append(nn.Dropout(p=dropout_rate))
            input_dim = hidden_dim

        # Final layer to features_dim
        layers.append(nn.Linear(input_dim, features_dim))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
            try:
                df = pd.read_csv(db_path)
                # Check if the dataframe is empty or has no columns
                if df.empty or len(df.columns) == 0:
                    logger.warning(f"Empty or malformed experiments.csv found, creating new database")
                    return self._create_empty_experiments_db()
                return df
            except pd.errors.EmptyDataError:
                logger.warning(f"Empty experiments.csv found, creating new database")
                return self._create_empty_experiments_db()
        return self._create_empty_experiments_db()

    def _create_empty_experiments_db(self) -> pd.DataFrame:
        """Create an empty experiments database with proper schema"""
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

    def create_environment(self, env_config: SplunkConfig, overrides: dict = None) -> gym.Env:
        """Create and configure environment with reward wrappers

        Args:
            env_config: SplunkConfig instance
            overrides: Dict of config overrides from CLI args
        """
        if overrides is None:
            overrides = {}

        # Helper function to get value from overrides or config
        def get_config(key, default=None):
            return overrides.get(key, config.get(key, default))

        # Load and filter top log types from CSV
        top_logtypes = pd.read_csv(config.get('paths.top_logtypes'))
        # Include only configured log types (default: system and security logs)
        log_types = get_config('environment.log_types', ['wineventlog:security', 'wineventlog:system'])
        max_logtypes = get_config('environment.max_logtypes', 20)
        top_logtypes = top_logtypes[top_logtypes['source'].str.lower().isin(log_types)]
        top_logtypes = top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:max_logtypes]
        top_logtypes = [(x[0].lower(), str(x[1])) for x in top_logtypes]

        experiment_name = get_config('experiment_name')
        if experiment_name and "test_experiment" in experiment_name:
            env_config.is_test = True

        env = make(
            id=env_config.env_id,
            config=env_config,
            top_logtypes=top_logtypes,
            baseline_dir=self.dirs['baseline']
        )

        # Configure action space
        action_type = get_config('environment.action_type', 'Action8')
        use_random_agent = get_config('use_random_agent', False)

        if action_type == "Action8":
            env = Action8(env, use_random_agent)
        elif action_type == "Action12":
            env = Action12(env, use_random_agent)

        # Add reward wrappers
        use_distribution_reward = get_config('reward.use_distribution_reward', True)
        if use_distribution_reward:
            distribution_method = get_config('reward.distribution_method', 'DistributionRewardWrapper')
            env = ENUM_DISTRIBUTION_REWARD_METHODS[distribution_method](
                env,
                gamma=get_config('reward.gamma', 0.2),
                epsilon=get_config('reward.epsilon', 1e-8),
                distribution_freq=get_config('reward.distribution_freq', 1),
                distribution_threshold=get_config('reward.distribution_threshold', 0.22)
            )

        # Base rule execution wrapper
        mode = get_config('experiment.mode', 'train')
        env = BaseRuleExecutionWrapperWithPrediction(
            env,
            is_mock=get_config('environment.is_mock', True),
            enable_prediction=True,
            alert_threshold=get_config('reward.alert_threshold', -10),
            skip_on_low_alert=get_config('reward.skip_on_low_alert', True),
            use_energy=get_config('reward.use_energy_reward', True),
            use_alert=get_config('reward.use_alert_reward', True),
            is_train='train' in mode,
            is_eval=(mode == "eval_post_training"),
            beta=get_config('reward.beta', 0.3),
            gamma=get_config('reward.gamma', 0.2)
        )

        # Energy and alert reward wrappers
        if get_config('reward.use_energy_reward', True):
            env = EnergyRewardWrapper(
                env,
                alpha=get_config('reward.alpha', 0.5),
                is_mock=get_config('environment.is_mock', True),
            )
            alert_method = get_config('reward.alert_method', 'AlertRewardWrapper')
            env = ENUM_ALERT_REWARD_METHODS[alert_method](
                env,
                beta=get_config('reward.beta', 0.3),
                epsilon=get_config('reward.alert_epsilon', 0.1),
                normalizer_factor=get_config('reward.normalizer_factor', 30.0)
            )

        # Time and state wrappers
        env = TimeWrapper(env)
        hosts_num = get_config('environment.hosts_percentage', 100)
        logger.info(f"Using {hosts_num}% of hosts")
        env = StateWrapper7(env, hosts_percentage=hosts_num)

        return env


    def create_model(self, env: gym.Env, overrides: dict = None):
        """Create or load model based on config"""
        if overrides is None:
            overrides = {}

        mode = overrides.get('experiment.mode', config.get('experiment.mode', 'train'))
        model_path = overrides.get('model_path')

        if mode == "train" and not model_path:
            return self._create_new_model(env, overrides)
        else:
            return self._load_existing_model(env, overrides)
        
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

    def _get_device(self, device_config: str = 'auto') -> str:
        """Determine PyTorch device based on config and availability.

        Args:
            device_config: Device configuration string ('auto', 'cuda', or 'cpu')

        Returns:
            Device string suitable for PyTorch ('cuda' or 'cpu')
        """
        if device_config == 'auto':
            device = 'cuda' if th.cuda.is_available() else 'cpu'
            logger.info(f"Auto-detected device: {device}")
            return device
        elif device_config == 'cuda' and not th.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return 'cpu'
        else:
            return device_config

    def _create_new_model(self, env: gym.Env, overrides: dict = None):
        """Create new model instance"""
        if overrides is None:
            overrides = {}

        def get_config(key, default=None):
            return overrides.get(key, config.get(key, default))

        model_type = get_config('experiment.model_type', 'sac')
        policy_type = get_config('experiment.policy_type', 'MlpPolicy')
        experiment_name = get_config('experiment_name', 'experiment')

        model_cls = self._get_model_class(model_type)

        # Get device with automatic CUDA fallback
        device = self._get_device(get_config('training.device', 'auto'))

        model_kwargs = {
            'env': env,
            'policy': policy_type,
            'learning_rate': get_config('training.learning_rate', 3e-4),
            'gamma': get_config('training.gamma', 0.95),
            'tensorboard_log': f"{str(self.dirs['tensorboard'])}/{experiment_name}",
            'stats_window_size': get_config('training.stats_window_size', 5),
            'verbose': get_config('training.verbose', 0),
            'device': device
        }

        if model_type in ['recurrent_ppo', 'ppo', 'a2c']:
            model_kwargs.update({
                'n_steps': get_config('training.n_steps', 2048),
                'ent_coef': get_config('training.ent_coef', 0.05),
                'sde_sample_freq': get_config('training.ppo.sde_sample_freq', 12),
                'use_sde': get_config('training.ppo.use_sde', True),
            })

        elif model_type in ['sac', 'td3', 'ddpg']:
            # Get SAC-specific config values
            train_freq_value = get_config('training.sac.train_freq', 4)
            train_freq_unit = get_config('training.sac.train_freq_unit', 'episode')
            pi_arch = get_config('training.sac.policy_net_arch.pi', [256, 256])
            qf_arch = get_config('training.sac.policy_net_arch.qf', [256, 256])

            model_kwargs.update({
                'learning_starts': get_config('training.sac.learning_starts', 60),  # (12 steps * 5 episodes)
                'gradient_steps': get_config('training.sac.gradient_steps', -1),
                'train_freq': (train_freq_value, train_freq_unit),
                'buffer_size': get_config('training.sac.buffer_size', 100_000),
                'batch_size': get_config('training.sac.batch_size', 2048),
                'ent_coef': get_config('training.sac.ent_coef', 'auto'),
                'use_sde': get_config('training.sac.use_sde', True),
                "policy_kwargs": {
                    "net_arch": dict(pi=pi_arch, qf=qf_arch),
                    "log_std_init": get_config('training.sac.log_std_init', -3),
                },
            })
      
            
        return model_cls(**model_kwargs)
    
    def _generate_experiment_id(self):
        """Generate unique experiment ID"""
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    def _setup_experiment_logging(self, experiment_name: str):
        """Setup logging for experiment"""
        log_level_str = config.get('logging.level', 'INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        log_format = config.get('logging.format',
                               "%(asctime)s [%(levelname)s] %(name)s %(message)s")
        log_to_console = config.get('logging.log_to_console', False)

        handlers = [
            logging.FileHandler(self.dirs['logs'] / f"{experiment_name}.log")
        ]
        if log_to_console:
            handlers.append(logging.StreamHandler())

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers
        )
    
    def run_experiment(self, env_config: SplunkConfig, overrides: dict = None):
        """Run experiment based on configuration

        Args:
            env_config: SplunkConfig instance
            overrides: Dict of config overrides from CLI args
        """
        if overrides is None:
            overrides = {}

        def get_config(key, default=None):
            return overrides.get(key, config.get(key, default))

        # Generate experiment ID and name
        logger.info(f"Experiment Config: {overrides}")
        experiment_id = self._generate_experiment_id()
        experiment_name = get_config('experiment_name')
        mode = get_config('experiment.mode', 'train')

        if experiment_name is None:
            experiment_name = f"{mode}_{experiment_id}"
        else:
            experiment_name = f"{experiment_name}_{experiment_id}"

        overrides['experiment_name'] = experiment_name
        
        # Setup logging
        self._setup_experiment_logging(experiment_name)
        logger.info(f"Starting experiment: {experiment_name}")

        # Record experiment start
        self._record_experiment_start(experiment_id, experiment_name, overrides)

        try:
            # Create environment and model
            env = self.create_environment(env_config, overrides)
            model = self.create_model(env, overrides)

            # Create eval environment
            eval_env_config = SplunkConfig(
                rule_frequency=config.get('splunk.eval_rule_frequency', 2880),
                search_window=config.get('splunk.search_window', 2880),
                logs_per_minute=config.get('splunk.logs_per_minute', 150),
                additional_percentage=get_config('environment.additional_percentage', 1.0),
                action_duration=config.get('splunk.action_duration', 14400),
                num_of_measurements=config.get('splunk.num_measurements', 1),
                baseline_num_of_measurements=config.get('splunk.baseline_num_measurements', 1),
                env_id=config.get('splunk.eval_env_id', 'splunk_eval-v32'),
                end_time=config.get('splunk.eval_end_time', '09/01/2025:00:00:00'),
                ip=env_config.ip
            )
            eval_overrides = overrides.copy()
            eval_overrides['experiment.mode'] = 'eval'
            eval_overrides['environment.is_mock'] = False

            self.eval_env = self.create_environment(eval_env_config, eval_overrides)
            self.eval_env.unwrapped.splunk_tools.load_real_logs_distribution_bucket(
                datetime.datetime.strptime(env.unwrapped.time_manager.first_start_datetime, '%m/%d/%Y:%H:%M:%S'),
                datetime.datetime.strptime(self.eval_env.unwrapped.time_manager.end_time, '%m/%d/%Y:%H:%M:%S')
            )

            host = os.getenv(f'SPLUNK_HOST_{env_config.ip}', 'localhost')
            empty_monitored_files(get_system_monitor_path(host))
            empty_monitored_files(get_security_monitor_path(host))

            if "test_experiment" not in experiment_name:
                # clean and warm up the env
                logger.info("Cleaning and warming up the environment")
                clean_env(env.unwrapped.splunk_tools,
                         (env.unwrapped.time_manager.first_start_datetime,
                          datetime.datetime.now().strftime("%m/%d/%Y:%H:%M:%S")),
                         host=host)
                env.unwrapped.warmup()
            else:
                action_env = env
                action_eval_env = self.eval_env
                while not isinstance(action_env, Action8):
                    action_env = action_env.env
                while not isinstance(action_eval_env, Action8):
                    action_eval_env = action_eval_env.env

                action_env.disable_injection()
                action_eval_env.disable_injection()

            # Setup callbacks
            callbacks = self._setup_callbacks(overrides)

            # Run experiment
            if mode == "train":
                results = self._run_training(model, env, overrides, callbacks)
            elif mode == "eval_post_training":  # eval after training
                # Create full eval environment
                full_eval_overrides = eval_overrides.copy()
                full_eval_overrides['environment.hosts_percentage'] = config.get('evaluation.full_eval.hosts_percentage', 100)
                full_eval_overrides['reward.use_energy_reward'] = config.get('evaluation.full_eval.use_energy_reward', False)
                full_eval_overrides['reward.use_alert_reward'] = config.get('evaluation.full_eval.use_alert_reward', False)
                full_eval_overrides['environment.is_mock'] = config.get('evaluation.full_eval.is_mock', True)

                full_eval_env = self.create_environment(eval_env_config, full_eval_overrides)
                results = self._run_evaluation(model, self.eval_env, eval_overrides, full_eval_env)
            else:  # retrain
                results = self._run_retraining(model, env, overrides, callbacks)
                
            # Record success
            self._record_experiment_end(experiment_id, "completed", results)
            
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            self._record_experiment_end(experiment_id, "failed", {"error": str(e)})
            # Send email notification
            experiment_name = overrides.get('experiment_name', 'unknown')
            self.send_email(error_message=str(e), experiment_name=experiment_name)

            raise

    def _run_training(self, model, env, overrides: dict, callbacks):
        """Run training experiment"""
        if overrides is None:
            overrides = {}

        def get_config(key, default=None):
            return overrides.get(key, config.get(key, default))

        num_episodes = get_config('training.num_episodes', 100)
        experiment_name = get_config('experiment_name', 'experiment')

        total_timesteps = env.unwrapped.total_steps * num_episodes

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=experiment_name
        )

        # Save model
        model_path = self.dirs['models'] / f"{experiment_name}.zip"
        model.save(str(model_path))

        return {
            "model_path": str(model_path),
            "total_timesteps": total_timesteps
        }

   
    def _run_evaluation(self, model, env, overrides: dict, full_eval_env=None):
        """evaluate the model for a specific number of episodes. Create summary writers for the evaluation """
        if overrides is None:
            overrides = {}

        def get_config(key, default=None):
            return overrides.get(key, config.get(key, default))

        model.set_env(env)
        eval_episodes = get_config('training.num_episodes', 100)
        experiment_name = get_config('experiment_name', 'experiment')

        log_dir = f"{str(self.dirs['tensorboard'])}/{experiment_name}"
        eval_logger = configure(log_dir, ["tensorboard"])
        model.set_logger(eval_logger)
        rules = self.eval_env.unwrapped.splunk_tools.active_saved_searches.keys()
        event_types = [f"{x[0].lower()}_{x[1]}" for x in self.eval_env.unwrapped.top_logtypes]
        writers = self.create_summary_writers(log_dir, rules, event_types)
        eval_callback = CustomEvalCallback3(
            eval_env=self.eval_env,
            log_dir=f"{str(self.dirs['tensorboard'])}/{experiment_name}",
            rules=rules,
            event_types=event_types,
            n_eval_episodes=get_config('evaluation.n_eval_episodes', 1),
            eval_freq=get_config('evaluation.eval_freq', 1),
            best_model_save_path=self.dirs['models'],
            log_path=self.dirs['logs'],
            # eval_log_dir=str(self.dirs['tensorboard']/f"eval_{experiment_name}"),
            deterministic=get_config('evaluation.deterministic', True),
            render=get_config('evaluation.render', False),
            verbose=get_config('evaluation.verbose', 1),
            writers=writers,
            full_eval_env=full_eval_env,
            additional_percentage=get_config('environment.additional_percentage', 1.0),
            hosts_num=get_config('environment.hosts_percentage', 100),
            is_random_agent=get_config('use_random_agent', False),
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



    def _load_existing_model(self, env: gym.Env, overrides: dict = None):
        """Load model from path"""
        if overrides is None:
            overrides = {}

        def get_config(key, default=None):
            return overrides.get(key, config.get(key, default))

        model_type = get_config('experiment.model_type', 'sac')
        model_path = get_config('model_path')

        if not model_path:
            raise ValueError("model_path must be provided to load existing model")

        model_cls = self._get_model_class(model_type)
        logger.info(f"Loading model from {model_path}")
        model = model_cls.load(model_path, env=env)
        logger.info(f"Successfully loaded model from {model_path}")

        return model
        
        
    def _run_retraining(self, model, env, overrides: dict, callbacks):
        """Run retraining experiment"""
        if overrides is None:
            overrides = {}

        def get_config(key, default=None):
            return overrides.get(key, config.get(key, default))

        num_episodes = get_config('training.num_episodes', 100)
        experiment_name = get_config('experiment_name', 'experiment')

        total_timesteps = env.unwrapped.total_steps * num_episodes

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=experiment_name
        )
        model_path = self.dirs['models'] / f"{experiment_name}.zip"
        model.save(str(model_path))

        return {
            "model_path": str(model_path),
            "total_timesteps": total_timesteps
        }
    
    def _record_experiment_start(self, experiment_id: str, name: str,
                               overrides: dict):
        """Record experiment start in database"""
        def get_config(key, default=None):
            return overrides.get(key, config.get(key, default))

        # Create serializable config dict
        serialized_config = overrides.copy()
        mode = get_config('experiment.mode', 'train')

        new_row = {
            'experiment_id': experiment_id,
            'name': name,
            'mode': mode,
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
    
    def _setup_callbacks(self, overrides: dict):
        """Setup training/evaluation callbacks"""
        if overrides is None:
            overrides = {}

        def get_config(key, default=None):
            return overrides.get(key, config.get(key, default))

        experiment_name = get_config('experiment_name', 'experiment')
        log_dir = f"{str(self.dirs['tensorboard'])}/{experiment_name}"

        rules = self.eval_env.unwrapped.splunk_tools.active_saved_searches.keys()
        event_types = [f"{x[0].lower()}_{x[1]}" for x in self.eval_env.unwrapped.top_logtypes]
        writers = self.create_summary_writers(log_dir, rules, event_types)
        return [
            CustomTensorboardCallback(
                log_dir=f"{str(self.dirs['tensorboard'])}/{experiment_name}",
                rules=rules,
                event_types=event_types,
                writers=writers
            ),
            # HParamsCallback(
            #     experiment_kwargs=overrides,
            #     phase=get_config('phase', 'train')
            # ),
            CheckpointCallback(
                save_freq=get_config('callbacks.checkpoint.save_freq', 10000),
                save_path=self.dirs['models'],
                name_prefix=get_config('callbacks.checkpoint.name_prefix', experiment_name)
            ),

            CustomEvalCallback3(
                eval_env=self.eval_env,
                log_dir=f"{str(self.dirs['tensorboard'])}/{experiment_name}",
                rules=rules,
                event_types=event_types,
                n_eval_episodes=get_config('evaluation.n_eval_episodes', 1),
                eval_freq=get_config('callbacks.eval.eval_freq', 600000),
                best_model_save_path=self.dirs['models'],
                log_path=self.dirs['logs'],
                # eval_log_dir=str(self.dirs['tensorboard']/f"eval_{experiment_name}"),
                deterministic=get_config('callbacks.eval.deterministic', False),
                render=get_config('callbacks.eval.render', False),
                verbose=get_config('callbacks.eval.verbose', 1),
                writers=writers,
            ),
            # SplunkLincenceCheckCallback()

        ]
        
    def send_email(self, error_message: str = "Experiment has failed",
                   log_file: Optional[str] = None,
                   experiment_name: Optional[str] = None):
        """Send email notification about experiment failure.

        Args:
            error_message: Error description
            log_file: Optional path to log file to attach
            experiment_name: Optional experiment name for subject
        """
        try:
            my_email = config.get('email.address')
            email_password = config.get('email.password')

            if not my_email or not email_password:
                logger.warning("Email credentials not configured, skipping notification")
                return

            # Get email configuration from secrets.yaml
            smtp_server = config.get('email.server', 'smtp.gmail.com')
            smtp_port = config.get('email.port', 465)
            use_ssl = config.get('email.use_ssl', True)
            verify_ssl = config.get('email.verify_ssl', True)
            subject_template = config.get('email.subject', 'DRL Experiment Failed')

            # Customize subject if experiment name provided
            subject = f"{subject_template}: {experiment_name}" if experiment_name else subject_template

            # Create message
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = my_email
            msg['To'] = my_email
            msg.set_content(f"The experiment has failed.\n\nError message:\n{error_message}")

            # Create SSL context with configurable verification
            context = ssl.create_default_context()
            if not verify_ssl:
                logger.warning("SSL verification disabled for email - this is insecure!")
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE

            # Send email
            if use_ssl:
                with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as smtp:
                    smtp.login(my_email, email_password)
                    smtp.send_message(msg)
            else:
                with smtplib.SMTP(smtp_server, smtp_port) as smtp:
                    smtp.starttls(context=context)
                    smtp.login(my_email, email_password)
                    smtp.send_message(msg)

            logger.info(f"Email notification sent to {my_email}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            # Don't re-raise - email failure shouldn't crash the experiment cleanup

