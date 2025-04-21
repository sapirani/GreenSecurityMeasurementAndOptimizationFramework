from dataclasses import dataclass, replace
import inspect
from typing import Dict, Any, Optional, List
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
logger = logging.getLogger(__name__)

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
    
    # Reward components
    use_distribution_reward: bool = True
    use_energy_reward: bool = True
    use_alert_reward: bool = True
    use_quota_violation: bool = True
    use_random_agent: bool = False
    
    
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
        top_logtypes = top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
        top_logtypes = [(x[0].lower(), str(x[1])) for x in top_logtypes]
        
        env = make(
            id=config.env_config.env_id,
            config=config.env_config,
            top_logtypes=top_logtypes
        )
        # env = SingleAction2(env)
        # env = Action(env)
        env = Action7(env)
        
        if config.use_random_agent:
            env = RandomAction(env)





                
        # Add reward wrappers
        if config.use_distribution_reward:
            env = DistributionRewardWrapper(
                env,
                gamma=config.gamma_dist,
                epsilon=1e-8,
                distribution_freq=1
            )
        if config.use_energy_reward:
            env = BaseRuleExecutionWrapper(env, baseline_dir=self.dirs['baseline'])
            
            env = EnergyRewardWrapper(
                env,
                alpha=config.alpha_energy
            )
            
        
        if config.use_alert_reward:
            # env = AlertRewardWrapper(
            #     env,
            #     beta=config.beta_alert,
            #     epsilon=1e-3
            # )
            env = AlertRewardWrapper1(
                env,
                beta=config.beta_alert,
                epsilon=1e-3
            )
        # env = ClipRewardWrapper(env,
        #                         low=0,
        #                         high=1)
        env = TimeWrapper(env)
        
        env = StateWrapper2(env)

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
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_policy_class(self, policy_type: str):
        """Get policy class based on type"""
        if policy_type == "mlp":
            return MlpPolicy

        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    def _create_new_model(self, config: ExperimentConfig, env: gym.Env):
        """Create new model instance"""
        model_cls = self._get_model_class(config.model_type)
        
        model_kwargs = {
            'env': env,
            'policy': config.policy_type,
            'learning_rate': config.learning_rate,
            'gamma': config.gamma,
            'tensorboard_log': str(self.dirs['tensorboard']),
            'stats_window_size': 5,
            'verbose': 1
        }
        
        if config.model_type in ['recurrent_ppo', 'ppo', 'a2c']:
            model_kwargs.update({
                'n_steps': config.n_steps,
                # 'batch_size': config.batch_size,
                # 'n_epochs': config.n_epochs,
                'ent_coef': config.ent_coef,
                'sde_sample_freq': 12,
                'use_sde': True
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
        experiment_name = config.experiment_name or f"{config.mode}_{experiment_id}"
        
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
            eval_config = replace(config, mode="eval")
            eval_config.env_config.env_id = "splunk_eval-v32"
            self.eval_env = self.create_environment(eval_config)

            # if config.experiment_name != "test_experiment" :
            #     # clean and warm up the env
            #     # clean_env(env.splunk_tools, (env.time_manager.first_start_datetime, datetime.datetime.now().strftime("%m/%d/%Y:%H:%M:%S")))
            #     env.warmup()
            # Setup callbacks
            config.experiment_name = experiment_name
            callbacks = self._setup_callbacks(config)
            
            # Run experiment
            if config.mode == "train":
                results = self._run_training(model, env, config, callbacks)
            elif config.mode == "eval":
                results = self._run_evaluation(model, env, config, callbacks)
            else:  # retrain
                results = self._run_retraining(model, env, config, callbacks)
                
            # Record success
            self._record_experiment_end(experiment_id, "completed", results)
            
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            self._record_experiment_end(experiment_id, "failed", {"error": str(e)})
            raise

    def _run_training(self, model, env, config, callbacks):
        """Run training experiment"""
        # total_timesteps = env.get_attr('total_steps')[0] * config.num_episodes
        total_timesteps = env.total_steps * config.num_episodes
        
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

    def _run_evaluation(self, model, env, config, callbacks):
        """Run evaluation experiment"""
        episode_rewards = []
        
        for episode in range(config.num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
            episode_rewards.append(episode_reward)
            
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "episode_rewards": episode_rewards
        }

    def _load_existing_model(self, config: ExperimentConfig, env: gym.Env):
        """Load model from path"""
        model_cls = self._get_model_class(config.model_type)
        model = model_cls.load(config.model_path, env=env)
        
        
        return model
        
        
    def _run_retraining(self, model, env, config, callbacks):
        """Run retraining experiment"""
        total_timesteps = env.total_steps * config.num_episodes
              
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
    
        
        return [
            CustomTensorboardCallback(),
            # HParamsCallback(
            #     experiment_kwargs=config,
            #     phase=config.get('phase', 'train')
            # ),
            CheckpointCallback(save_freq=1000, save_path=self.dirs['models'], name_prefix=config.experiment_name),
            
            CustomEvalCallback3(
                eval_env=self.eval_env,

                n_eval_episodes=3,
                eval_freq=460,
                best_model_save_path=self.dirs['models'],
                log_path=self.dirs['logs'],
                # eval_log_dir=str(self.dirs['tensorboard']/f"eval_{config.experiment_name}"),
                deterministic=True,
                render=False,
                verbose=1
            )
            
        ]

    
# Example usage:
if __name__ == "__main__":
    # Create experiment config
    # retrain_fake_start_datetime = "05/01/2024:00:00:00"
    
    env_config = SplunkConfig(
        # fake_start_datetime=retrain_fake_start_datetime,
        rule_frequency=60,
        search_window=2880,
        # savedsearches=["rule1", "rule2"],
        logs_per_minute=150,
        additional_percentage=.5,
        action_duration=7200,
        num_of_measurements=3,
        env_id="splunk_train-v32"        
    )
    experiment_config = ExperimentConfig(
        env_config=env_config,
        model_type="recurrent_ppo",
        policy_type="MlpLstmPolicy",
        learning_rate=1e-3,
        num_episodes=6000,
        n_steps=48,
        ent_coef=0.01,
        gamma=1,
        # experiment_name="test_experiment",
        use_alert_reward=False,
        use_energy_reward=True,
        use_random_agent=False,
    )
    
    #retrain model
    experiment_config.mode = "train"
    
    model_path = "/home/shouei/GreenSecurity-FirstExperiment/experiments/models/train_20250320100253_42000_steps.zip"
    experiment_config.model_path = model_path
    
    
    # Create manager and run experiment
    manager = ExperimentManager(base_dir="experiments")
    results = manager.run_experiment(experiment_config)
