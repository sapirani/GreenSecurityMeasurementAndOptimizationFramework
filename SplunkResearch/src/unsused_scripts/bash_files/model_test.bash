#!/bin/bash
# # cd in SplunkResearch and run this bash like this: screen -L bash model/model_run.bash after cd to SplunkReasearch dir
# Set environment variables
# export SPLUNK_HOME=/opt/splunk
# export PATH=$SPLUNK_HOME/bin:$PATH
# export PATH=/home/shouei/anaconda3/bin:$PATH
# export PATH=/home/shouei/anaconda3/envs/py38/bin:$PATH
echo -n " Please enter the password for the given user: "
read password
export PATH=$PATH:/home/shouei/local/dmidecode
PYTHON_SCRIPT=src/main.py
limits_path="/opt/splunk/etc/system/local/limits.conf"
saved_searches_path="/opt/splunk/etc/users/shouei/search/local/savedsearches.conf"
# conda init bash
# conda activate /home/shouei/anaconda3/envs/py38
# which python
# echo $password | sudo -S sed -i 's/^max_searches_per_process = .*/max_searches_per_process = 1/' $limits_path
test_episodes=50
# env_name=

# test_experiment="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/exp_20240207_180124"
test_experiment="last"
config_path="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/config.json"


alpha=0
beta=0.2
gamma=$(awk "BEGIN {print 1 - $alpha - $beta}")
echo "learning_rate: $learning_rate, alpha: $alpha, beta: $beta, gamma: $gamma"

declare -A kwargs
kwargs['model']="a2c" #"recurrentppo"
kwargs['policy']="mlp" #"lstm"
kwargs['additional_percentage']="0.1"
kwargs['span_size']=72
#kwargs['fake_start_datetime']="05/03/2024:13:00:00"
kwargs['search_window']="360"

kwargs['ent_coef']=0
kwargs['df']=1
kwargs['rule_frequency']=1
kwargs['logs_per_minute']=300
kwargs['num_of_measurements']=7
kwargs['alpha']=0
kwargs['beta']=.2
kwargs['gamma']=.8
kwargs['learning_rate']=0.001
kwargs['reward_calculator_version']=25
kwargs['state_strategy_version']=5
kwargs['action_strategy_version']=5

kwargs['env_name']="splunk_eval-v32"
kwargs['num_of_episodes']=$test_episodes

# kwargs['experiment_name']="train_20241126_230112"
# args=""
# for key in "${!kwargs[@]}"; do
#     args+="--$key ${kwargs[$key]} "
# done
# echo $args
# echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test $args

kwargs['reward_calculator_version']=24
kwargs['experiment_name']="train_20241125_115736"
args=""
for key in "${!kwargs[@]}"; do
    args+="--$key ${kwargs[$key]} "
done
echo $args

echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test $args



# kwargs['experiment_name']="train_20241112_120658"
# kwargs['env_name']="splunk_eval-v"$env_name
# kwargs['num_of_episodes']=$test_episodes

# args=""
# for key in "${!kwargs[@]}"; do
#     args+="--$key ${kwargs[$key]} "
# done
# echo $args

# echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test $args
# wait


echo $password | sudo -S -E env PATH="$PATH" chmod -R 777 ./experiments__/
