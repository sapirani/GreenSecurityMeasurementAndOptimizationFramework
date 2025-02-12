from dataclasses import dataclass
import inspect
from typing import Dict, Any, Optional, List
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
from strategies.action_strategy import ActionStrategy
from strategies.state_strategy import StateStrategy
from wrappers.reward import *
from callbacks import *
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
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.0
    num_episodes: int = 1000
    
    # Reward components
    use_distribution_reward: bool = True
    use_energy_reward: bool = True
    use_alert_reward: bool = True
    use_quota_violation: bool = True
    
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
                # Load available strategies
        self.state_strategies = {
            name: cls for name, cls in inspect.getmembers(
                sys.modules['strategies.state_strategy'],
                lambda x: inspect.isclass(x) and issubclass(x, StateStrategy) and x != StateStrategy
            )
        }
        
        self.action_strategies = {
            name: cls for name, cls in inspect.getmembers(
                sys.modules['strategies.action_strategy'],
                lambda x: inspect.isclass(x) and issubclass(x, ActionStrategy) and x != ActionStrategy
            )
        }
        
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
        # Create base environment using gym.make with env_id
        savedsearches = ["Windows Event For Service Disabled",
                    "Detect New Local Admin account",
                    "ESCU Network Share Discovery Via Dir Command Rule",
                    "Known Services Killed by Ransomware",
                    "Non Chrome Process Accessing Chrome Default Dir",
                    "Kerberoasting spn request with RC4 encryption",
                    "Clop Ransomware Known Service Name"]
        fake_start_datetime = "02/01/2024:00:00:00"
        env_id = "splunk_train-v32"
        register(id=env_id,
                entry_point='custom_splunk.envs:SplunkEnv', 
                kwargs={
                        'savedsearches':savedsearches,
                        'fake_start_datetime':fake_start_datetime,

                })
        env = make(
            id=config.env_config.env_id,
            config=config.env_config,
            state_strategy=self.state_strategies[config.env_config.state_strategy],
            action_strategy=self.action_strategies[config.env_config.action_strategy]
        )
        
        # Add reward wrappers
        if config.use_distribution_reward:
            env = DistributionRewardWrapper(
                env,
                gamma=config.gamma_dist,
                epsilon=1e-8
            )
        
        if config.use_energy_reward:
            env = BaseRuleExecutionWrapper(env, baseline_dir=self.dirs['baseline'])
            env = EnergyRewardWrapper(
                env,
                alpha=config.alpha_energy
            )
        
        if config.use_alert_reward:
            env = AlertRewardWrapper(
                env,
                beta=config.beta_alert,
                epsilon=1e-3
            )
        
        if config.use_quota_violation:
            env = QuotaViolationWrapper(env)
        
        # Vectorize and normalize
        # env = DummyVecEnv([lambda: env])
        # env = VecNormalize(
        #     env,
        #     norm_obs=True,
        #     norm_reward=True,
        #     clip_obs=10.,
        #     clip_reward=10.,
        #     gamma=config.gamma
        # )
        
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
            'policy': self._get_policy_class(config.policy_type),
            'learning_rate': config.learning_rate,
            'gamma': config.gamma,
            'tensorboard_log': str(self.dirs['tensorboard']),
            'verbose': 1
        }
        
        if config.model_type in ['ppo', 'a2c']:
            model_kwargs.update({
                'n_steps': config.n_steps,
                'batch_size': config.batch_size,
                'n_epochs': config.n_epochs,
                'ent_coef': config.ent_coef
            })
            
        return model_cls(**model_kwargs)
    
    def _generate_experiment_id(self):
        """Generate unique experiment ID"""
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    def _setup_experiment_logging(self, experiment_name: str):
        """Setup logging for experiment"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.dirs['logs'] / f"{experiment_name}.log"),
                logging.StreamHandler()
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
        return create_callbacks(self, config.__dict__)
    
# Example usage:
if __name__ == "__main__":
    # Create experiment config
    env_config = SplunkConfig(
        # fake_start_datetime="02/12/2025:00:00:00",
        rule_frequency=60,
        search_window=120,
        # savedsearches=["rule1", "rule2"],
        logs_per_minute=300,
        additional_percentage=0.1,
        action_duration=60,
        num_of_measurements=1,
        num_of_episodes=1000,
        state_strategy="StateStrategy12",
        action_strategy="ActionStrategy14",
        env_id="splunk_train-v32"        
    )
    
    experiment_config = ExperimentConfig(
        env_config=env_config,
        model_type="ppo",
        policy_type="mlp",
        learning_rate=3e-4,
        num_episodes=1000,
        experiment_name="test_experiment"
    )
    
    # Create manager and run experiment
    manager = ExperimentManager(base_dir="experiments")
    results = manager.run_experiment(experiment_config)