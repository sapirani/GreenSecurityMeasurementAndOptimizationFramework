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
train_episodes=150
test_episodes=20
# env_name=

# test_experiment="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/exp_20240207_180124"
test_experiment="last"
config_path="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/config.json"

# greed search on learning rate alpha beta and gamma
# for env_name in "splunk-v16"
# do
#     for learning_rate in 0.001 0.0001 #0.00001
#     do
#         for alpha in 0 #$(seq 0.1 0.5 1)
#         do
#             for beta in 0.2 #$(seq 0.1 0.5 $(echo "1 - $alpha" | bc))
#             do
#                 for search_window in 5 10 30 60 120
#                 do
#                     for additional_percentage in 0.2 1
#                     do
#                         for num_of_measurements in 5 10 
#                         do
#                             for df in 0.99 0.95 0.9
#                             do
#                                 for ent_coef in 0 0.01
#                                 do
#                                 gamma=$(awk "BEGIN {print 1 - $alpha - $beta}")
#                                 echo "learning_rate: $learning_rate, alpha: $alpha, beta: $beta, gamma: $gamma"

#                                 declare -A train_kwargs

#                                 # Add key-value pairs
#                                 train_kwargs['model']="a2c"
#                                 train_kwargs['policy']="mlp"
#                                 train_kwargs['additional_percentage']=$additional_percentage
#                                 train_kwargs['span_size']=30
#                                 train_kwargs['fake_start_datetime']="05/03/2024:13:00:00"
#                                 train_kwargs['search_window']=$search_window

#                                 train_kwargs['ent_coef']=$ent_coef
#                                 train_kwargs['df']=$df
#                                 train_kwargs['rule_frequency']=1
#                                 train_kwargs['logs_per_minute']=300
#                                 train_kwargs['num_of_measurements']=$num_of_measurements
#                                 train_kwargs['env_name']=$env_name
#                                 train_kwargs['num_of_episodes']=$train_episodes
#                                 train_kwargs['alpha']=$alpha
#                                 train_kwargs['beta']=$beta
#                                 train_kwargs['gamma']=$gamma
#                                 train_kwargs['learning_rate']=$learning_rate
#                                 train_kwargs['reward_calc']=10

                                
#                                 args=""
#                                 for key in "${!train_kwargs[@]}"; do
#                                     args+="--$key ${train_kwargs[$key]} "
#                                 done
#                                 echo $args

#                                 echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" train $args
#                                 wait

#                                 echo $password | sudo -S -E env PATH="$PATH" chmod -R 777 ./experiments__/
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

for env_name in "splunk-v16"
do
    for learning_rate in 0.001 0.0001 #0.00001
    do
        for alpha in 0 #$(seq 0.1 0.5 1)
        do
            for beta in 0.2 #$(seq 0.1 0.5 $(echo "1 - $alpha" | bc))
            do
                for search_window in 5 10 30 60 120
                do
                    for additional_percentage in 0.2 1
                    do
                        for num_of_measurements in 5 10 
                        do
                            for df in 0.99 0.95 0.9
                            do
                                for ent_coef in 0 0.01
                                do
                                gamma=$(awk "BEGIN {print 1 - $alpha - $beta}")
                                echo "learning_rate: $learning_rate, alpha: $alpha, beta: $beta, gamma: $gamma"

                                declare -A test_kwargs


                                # Add key-value pairs
                                test_kwargs['model']="a2c"
                                test_kwargs['policy']="mlp"
                                test_kwargs['additional_percentage']=$additional_percentage
                                test_kwargs['span_size']=30
                                test_kwargs['fake_start_datetime']="06/03/2024:13:00:00"
                                test_kwargs['search_window']=$search_window

                                test_kwargs['ent_coef']=$ent_coef
                                test_kwargs['df']=$df
                                test_kwargs['rule_frequency']=1
                                test_kwargs['logs_per_minute']=300
                                test_kwargs['num_of_measurements']=$num_of_measurements
                                test_kwargs['env_name']=$env_name
                                test_kwargs['num_of_episodes']=$train_episodes
                                test_kwargs['alpha']=$alpha
                                test_kwargs['beta']=$beta
                                test_kwargs['gamma']=$gamma
                                test_kwargs['learning_rate']=$learning_rate
                                test_kwargs['reward_calc']=10


                                
                                args=""
                                for key in "${!test_kwargs[@]}"; do
                                    args+="--$key ${test_kwargs[$key]} "
                                done
                                echo $args

                                echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test $args
                                wait

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