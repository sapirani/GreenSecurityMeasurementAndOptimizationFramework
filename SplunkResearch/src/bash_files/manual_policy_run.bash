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
train_episodes=275
test_episodes=5
# env_name=

# test_experiment="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/exp_20240207_180124"
test_experiment="last"
config_path="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/config.json"

# greed search on learning rate alpha beta and gamma
for env_name in 32 
do
    for state_strategy_version in 5
        do
            for search_window in 1440
            do
                for reward_calculator_version in 27
                do
                    for additional_percentage in .5
                    do
                        for policy in '4732_0' '4663_0' '4663_1' '4732_1' '4769_0' '4769_1' '5140_0' '5140_1' '7036_0' '7036_1' '7040_0' '7040_1' '7045_0' '7045_1' '4624_0' 'equal_0' 'equal_1'
                        # for policy in '4663_0' '4663_1' '4732_0' '4732_1' '4769_0' '4769_1' '5140_0' '5140_1' '7036_0' '7036_1' '7040_0' '7040_1' '7045_0' '7045_1' '4624_0' 'equal_0' 'equal_1'
                        do
                            alpha=0
                            beta=0.2
                            gamma=$(awk "BEGIN {print 1 - $alpha - $beta}")
                            echo "learning_rate: $learning_rate, alpha: $alpha, beta: $beta, gamma: $gamma"

                            declare -A kwargs

                            # Add key-value pairs
                            kwargs['policy']=$policy
                            kwargs['additional_percentage']=$additional_percentage
                            kwargs['span_size']=86400 #450 
                            #kwargs['fake_start_datetime']="05/03/2024:13:00:00"
                            kwargs['search_window']=$search_window

                        
                            kwargs['rule_frequency']=1
                            kwargs['logs_per_minute']=150
                            kwargs['num_of_measurements']=1
                            kwargs['alpha']=$alpha
                            kwargs['beta']=$beta
                            kwargs['gamma']=$gamma
                            kwargs['reward_calculator_version']=$reward_calculator_version
                            kwargs['state_strategy_version']=$state_strategy_version
                            kwargs['action_strategy_version']=6

                            kwargs['env_name']="splunk_eval-v"$env_name
                            kwargs['num_of_episodes']=$test_episodes
                            
                            args=""
                            for key in "${!kwargs[@]}"; do
                                args+="--$key ${kwargs[$key]} "
                            done
                            echo $args

                            echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" manual_policy $args
                            wait
                            
    


                            echo $password | sudo -S -E env PATH="$PATH" chmod -R 777 ./experiments__/
                        done
                    done
                done
            done
        done
    done
done