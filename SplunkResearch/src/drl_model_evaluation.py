# import sb3
import stable_baselines3 as sb3
from sb3_contrib import RecurrentPPO
sb3.__version__
from SplunkResearch.src.experiment_manager_new import *
manager = ExperimentManager(base_dir="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments")
# load model
# model_path = r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/train_20251010153827_70000_steps.zip"
model_path = r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/retrain_20251020135416_30000_steps.zip"
env_config = SplunkConfig(
    # fake_start_datetime=retrain_fake_start_datetime,
    rule_frequency=60, #600,
    search_window=2880,
    # savedsearches=["rule1", "rule2"],
    logs_per_minute=150,
    additional_percentage=1,
    action_duration=7200, 
    num_of_measurements=1,
    baseline_num_of_measurements=1,
    env_id="splunk_train-v32",
    end_time="12/31/2024:23:59:59"       
)
# sched_LR = lr_schedule(initial_value = 0.01, rate = 5)
experiment_config = ExperimentConfig(
    env_config=env_config,
    model_type="recurrent_ppo",# "ppo", # "a2c", "dqn", "sac", "td3", "recurrent_ppo"
    policy_type="MlpLstmPolicy",# "td3_mlp", # "mlp"
    learning_rate=0,#sched_LR,
    num_episodes=0,
    n_steps=64,
    ent_coef=0.01,
    gamma=1,
    gamma_dist=1,#0.33,
    alpha_energy=1,
    beta_alert=1,
    action_type="Action8",
    experiment_name="test_experiment",
    use_alert_reward=True,
    use_energy_reward=True,
    use_random_agent=0,
    is_mock=True,
    model_path=model_path if model_path else None,
    distribution_threshold=0.22,
    alert_threshold=-10,
)
experiment_config.mode = "eval_post_training"#"eval_post_training"  # eval after training

env = manager.create_environment(experiment_config)
env.unwrapped.splunk_tools.load_real_logs_distribution_bucket(datetime.datetime.strptime(env.unwrapped.time_manager.first_start_datetime, '%m/%d/%Y:%H:%M:%S'), datetime.datetime.strptime(env.unwrapped.time_manager.end_time, '%m/%d/%Y:%H:%M:%S'))

model = RecurrentPPO.load(model_path, env=env)
# evaluate model
for i in range(2):
    obs,info = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    done = False
    total_reward = 0
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
        # clip decimal action to 2 decimal places
        action = np.round(action, 2)
        print(f"Action taken: {action}")
        obs, reward, done, truncated, info = env.step(action)
        episode_starts = done
        total_reward += reward
    print(f"Episode {i+1}: Total Reward: {total_reward}")