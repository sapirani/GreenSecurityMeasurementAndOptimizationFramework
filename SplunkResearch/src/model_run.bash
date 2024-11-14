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
train_episodes=50000
test_episodes=20
# env_name=

# test_experiment="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/exp_20240207_180124"
test_experiment="last"
config_path="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/config.json"

# greed search on learning rate alpha beta and gamma
for env_name in 5
do
    for learning_rate in 0.001
    do
        for model in a2c
        do
            for state_strategy_version in 5
            do
                for search_window in 120
                do
                    for additional_percentage in .2
                    do
                        for reward_calculator_version in 24
                        do
                            for action_strategy_version in 5
                            do
                                for df in 1
                                do
                                    for num_of_measurements in 3
                                    do
                                        alpha=0
                                        beta=0.2
                                        gamma=$(awk "BEGIN {print 1 - $alpha - $beta}")
                                        echo "learning_rate: $learning_rate, alpha: $alpha, beta: $beta, gamma: $gamma"

                                        declare -A kwargs

                                        # Add key-value pairs
                                        kwargs['model']=$model #"recurrentppo"
                                        kwargs['policy']="mlp" #"lstm"
                                        kwargs['additional_percentage']=$additional_percentage
                                        kwargs['span_size']=72
                                        #kwargs['fake_start_datetime']="05/03/2024:13:00:00"
                                        kwargs['search_window']=$search_window

                                        kwargs['ent_coef']=0
                                        kwargs['df']=$df
                                        kwargs['rule_frequency']=1
                                        kwargs['logs_per_minute']=300
                                        kwargs['num_of_measurements']=$num_of_measurements
                                        kwargs['alpha']=$alpha
                                        kwargs['beta']=$beta
                                        kwargs['gamma']=$gamma
                                        kwargs['learning_rate']=$learning_rate
                                        kwargs['reward_calculator_version']=$reward_calculator_version
                                        kwargs['state_strategy_version']=$state_strategy_version
                                        kwargs['action_strategy_version']=$action_strategy_version

                                        kwargs['env_name']="splunk_train-v"$env_name
                                        kwargs['num_of_episodes']=$train_episodes
                                        
                                        args=""
                                        for key in "${!kwargs[@]}"; do
                                            args+="--$key ${kwargs[$key]} "
                                        done
                                        echo $args

                                        # echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" retrain $args
                                        # wait

                                        echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" train $args
                                        
                                        # ######################

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
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done