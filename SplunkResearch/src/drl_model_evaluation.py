# import sb3
import stable_baselines3 as sb3
from sb3_contrib import RecurrentPPO
sb3.__version__
from SplunkResearch.src.experiment_manager_new import *
manager = ExperimentManager(base_dir="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments")
action_type = "Action8"

# load model
# model_path = r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/train_20251010153827_70000_steps.zip"
# model_path = r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/retrain_20251020135416_30000_steps.zip"
model_path = r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/train_20251215190219_30000_steps.zip"
# model_path = r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/models/train_20250626010440_151000_steps.zip"
env_config = SplunkConfig(
    # fake_start_datetime=retrain_fake_start_datetime,
    rule_frequency=120, #600,
    search_window=2880,
    # savedsearches=["rule1", "rule2"],
    logs_per_minute=150,
    additional_percentage=1,
    action_duration=14400,#7200, 
    num_of_measurements=1,
    baseline_num_of_measurements=1,
    env_id="splunk_train-v32",
    # end_time="12/10/2024:00:00:00"       
    end_time="08/01/2025:00:00:00"       
)
# sched_LR = lr_schedule(initial_value = 0.01, rate = 5)
experiment_config = ExperimentConfig(
    env_config=env_config,
    model_type="sac",# "ppo", # "a2c", "dqn", "sac", "td3", "recurrent_ppo"
    policy_type="MlpPolicy",# "td3_mlp", # "mlp", "MlpLstmPolicy" "MlpPolicy"
    learning_rate=1e-5,#sched_LR,
    num_episodes=20,
    n_steps=10,
    ent_coef=0,
    gamma=0.95,
    gamma_dist= 0,
    alpha_energy= 0.8,
    beta_alert= 0.2,
    action_type=action_type,
    experiment_name="test_experiment",
    use_alert_reward=False,
    use_energy_reward=False,
    use_random_agent=False,
    is_mock=True,
    model_path=model_path if model_path else None,
    distribution_threshold=0.22,
    alert_threshold=-10,
    is_sample= False,
    
)
experiment_config.mode = "eval_post_training"#"eval_post_training"  # eval after training

env = manager.create_environment(experiment_config)
env.unwrapped.splunk_tools.load_real_logs_distribution_bucket(datetime.datetime.strptime(env.unwrapped.time_manager.first_start_datetime, '%m/%d/%Y:%H:%M:%S'), datetime.datetime.strptime(env.unwrapped.time_manager.end_time, '%m/%d/%Y:%H:%M:%S'))

model = SAC.load(model_path, env=env)
# model = RecurrentPPO.load(model_path, env=env)
# evaluate model
for i in range(20):
    obs,info = env.reset()
    lstm_states = None
    episode_starts = np.zeros((1,), dtype=bool)
    done = False
    total_reward = 0
    actions_list = []
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
        # clip decimal action to 2 decimal places

        obs, reward, done, truncated, info = env.step(action)
        action = np.round(action, 2)
        actions_list.append(action)
        # generate random state using numpy with bound
        # obs = np.random.uniform(low=0, high=1, size=obs.shape)
        # print(obs)
        episode_starts = done
        total_reward += reward
    print(f"Episode {i+1}: Total Reward: {total_reward}")
    actions_df = pd.DataFrame(actions_list, columns=[f"Action_{j}" for j in range(actions_list[0].shape[0])])
    # show entropy of each action column in one line
    actions_entropy = actions_df.apply(lambda x: -np.sum(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True) + 1e-9)))
    print(actions_df)
    print(f"Episode {i+1}: Actions Entropy:\n{actions_entropy}")