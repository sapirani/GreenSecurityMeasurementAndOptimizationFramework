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
test_episodes=10
# env_name=

# test_experiment="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/exp_20240207_180124"
test_experiment="last"
config_path="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/config.json"

# greed search on learning rate alpha beta and gamma
for env_name in "splunk-v15"
do
    for learning_rate in 0.001 #0.00001
    do
        for alpha in 0 #$(seq 0.1 0.5 1)
        do
            for beta in 0.2 #$(seq 0.1 0.5 $(echo "1 - $alpha" | bc))
            do
                # for reward_calc in 7
                # do
                #     gamma=$(awk "BEGIN {print 1 - $alpha - $beta}")
                #     echo "learning_rate: $learning_rate, alpha: $alpha, beta: $beta, gamma: $gamma"
                #     policy="lstm"
                #     model="recurrentppo"
                #     ent_coef=0.01
                #     df=0.95
                #     echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" train $env_name $model $train_episodes $alpha $beta $gamma $learning_rate $policy $reward_calc $ent_coef $df
                #     wait

                #     echo $password | sudo -S -E env PATH="$PATH" chmod -R 777 ./experiments__/
                # done
                for test_experiment in "last"
                do
                    gamma=$(awk "BEGIN {print 1 - $alpha - $beta}")
                    echo "learning_rate: $learning_rate, alpha: $alpha, beta: $beta, gamma: $gamma"
                    policy="lstm"
                    model="recurrentppo"
                    # echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" retrain $env_name $model $train_episodes $alpha $beta $gamma $learning_rate $policy $test_experiment
                    # wait
                    echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test $env_name $model $test_episodes $alpha $beta $gamma $learning_rate $policy $test_experiment
                    wait

                    # echo $password | sudo -S chmod -R 777 "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/VMware, Inc. Linux 3.10.0-1160.108.1.el7.x86_64"
                    echo $password | sudo -S -E env PATH="$PATH" chmod -R 777 ./experiments__/
                done
            done
        done
    done
    # echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" baseline $env_name $model $test_episodes $alpha $beta $gamma $learning_rate $policy random
    # wait
done


# echo $password | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" retrain $test_experiment
# wait