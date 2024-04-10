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
echo $password | sudo -S sed -i 's/^max_searches_per_process = .*/max_searches_per_process = 1/' $limits_path
train_episodes=300
test_episodes=30
env_name="splunk-v4"

# test_experiment="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/exp_20240207_180124"
test_experiment="last"
config_path="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/config.json"

# sed -i 's/"alpha": .*,/"alpha": 0.334,/' $config_path
# sed -i 's/"beta": .*,/"beta": 0.334,/' $config_path
# sed -i 's/"gamma": .*/"gamma": 0.334/' $config_path
# echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" train $train_episodes
# wait

# echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test $test_experiment $test_episodes
# wait
# edit config file - change alpha beta gamma
sed -i 's/"alpha": .*,/"alpha": 0.1667,/' $config_path
sed -i 's/"beta": .*,/"beta": 0.1667,/' $config_path
sed -i 's/"gamma": .*/"gamma": 0.667/' $config_path
model="ppo"
echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" train $env_name $model $train_episodes
wait
echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test $env_name $model $test_episodes
wait
# echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" baseline $env_name _ $test_episodes random
# wait
# echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" no_agent $env_name _ $test_episodes
# wait
# echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" baseline $env_name _ $test_episodes autopic
# wait
model="a2c"
echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" train $env_name $model $train_episodes
wait
echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test $env_name $model $test_episodes
wait

echo $password | sudo -S chmod -R 777 "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.108.1.el7.x86_64"
echo $password | sudo -S  -E env PATH="$PATH" chmod -R 777 ./experiments__/


# echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" retrain $test_experiment
# wait