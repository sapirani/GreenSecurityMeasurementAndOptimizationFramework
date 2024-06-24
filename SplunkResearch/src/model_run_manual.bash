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
train_episodes=360
test_episodes=60
# env_name=

# test_experiment="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/exp_20240207_180124"
test_experiment="last"
config_path="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/config.json"

# greed search on learning rate alpha beta and gamma
for env_name in "splunk-v12"
do
    for learning_rate in 0.001 #0.00001
    do
        for alpha in 0.15 #$(seq 0.1 0.5 1)
        do
            for beta in 0.6 #$(seq 0.1 0.5 $(echo "1 - $alpha" | bc))
            do
                    gamma=$(awk "BEGIN {print 1 - $alpha - $beta}")
                    echo "learning_rate: $learning_rate, alpha: $alpha, beta: $beta, gamma: $gamma"
                    # # edit config file - change alpha beta gamma
                    # sed -i "s/\"learning_rate\": .*,/\"learning_rate\": $learning_rate,/" $config_path
                    # sed -i "s/\"alpha\": .*,/\"alpha\": $alpha,/" $config_path
                    # sed -i "s/\"beta\": .*,/\"beta\": $beta,/" $config_path
                    # sed -i "s/\"gamma\": .*/\"gamma\": $gamma/" $config_path

                    # parameter_train_path="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments__/splunk-v9/parameters_train.json"
                    
                    # # Use sed to update the JSON values
                    # sed -i "s/\"learning_rate\": [^,]*/\"learning_rate\": $learning_rate/" "$parameter_train_path"
                    # sed -i "s/\"alpha\": [^,]*/\"alpha\": $alpha/" "$parameter_train_path"
                    # sed -i "s/\"beta\": [^,]*/\"beta\": $beta/" "$parameter_train_path"
                    # sed -i "s/\"gamma\": [^,}]*/\"gamma\": $gamma/" "$parameter_train_path"


                    model="a2c"
                    echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" train $env_name $model $train_episodes $alpha $beta $gamma $learning_rate
                    wait
                    echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test $env_name $model $test_episodes $alpha $beta $gamma $learning_rate
                    wait

                    # model="ppo"
                    # echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" train $env_name $model $train_episodes $alpha $beta $gamma $learning_rate
                    # wait
                    # echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test $env_name $model $test_episodes $alpha $beta $gamma $learning_rate
                    # wait

                    # echo $password | sudo -S chmod -R 777 "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.108.1.el7.x86_64"
                    echo $password | sudo -S -E env PATH="$PATH" chmod -R 777 ./experiments__/

            done
        done
    done
    echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" baseline $env_name _ $test_episodes random
    wait
    echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" no_agent $env_name _ $test_episodes
    wait
done


# echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" retrain $test_experiment
# wait