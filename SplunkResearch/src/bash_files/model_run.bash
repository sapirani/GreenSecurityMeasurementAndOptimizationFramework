#!/bin/bash

# =====================================================
# 1. Environment Setup
# =====================================================
echo -n "Please enter the password for the given user: "
read -s password
echo

# Path configurations
export PATH=$PATH:/home/shouei/local/dmidecode
PYTHON_SCRIPT="src/main.py"
LIMITS_PATH="/opt/splunk/etc/system/local/limits.conf"
CONFIG_PATH="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/config.json"

# Update Splunk configuration
echo $password | sudo -S sed -i 's/^max_searches_per_process = .*/max_searches_per_process = 1/' $LIMITS_PATH

# =====================================================
# 2. Configuration Functions
# =====================================================

configure_experiment() {
    local model_type=$1
    local policy_type=$2
    local num_episodes=$3
    local learning_rate=$4
    local additional_pct=$5
    local env_version=$6
    local reward_calculator=$7
    local experiment_name=$8
    local action_strategy_version=$9
    local search_window=${10}
    local span_size=${11}
    local rule_frequency=${12}
    local state_strategy_version=${13}
    if [[ ${14} == "test" ]]; then
        local mode="eval"
    else 
        local mode="train"
    fi
    
    local alpha=${15}
    local beta=${16}
    local fake_start_datetime=${17}
    local n_steps=${18}
    local gamma=$(awk "BEGIN {print 1 - $alpha - $beta}")
    
    declare -A kwargs
    
    kwargs=(
        ["model"]=$model_type
        ["policy"]=$policy_type
        ["span_size"]=$span_size
        ["search_window"]=$search_window
        ["env_name"]="splunk_${mode}-v${env_version}"  # Dynamic environment name
        ["num_of_episodes"]=$num_episodes
        ["learning_rate"]=$learning_rate
        ["additional_percentage"]=$additional_pct
        ["ent_coef"]=0.01
        ["df"]=1
        ["alpha"]=$alpha
        ["beta"]=$beta
        ["gamma"]=$gamma
        ["reward_calculator_version"]=$reward_calculator
        ["state_strategy_version"]=$state_strategy_version
        ["action_strategy_version"]=$action_strategy_version
        ["rule_frequency"]=$rule_frequency
        ["logs_per_minute"]=150
        ["num_of_measurements"]=5
        ['experiment_name']=$experiment_name
        ['fake_start_datetime']=$fake_start_datetime
        ['n_steps']=$n_steps
    )
    local args=""
    for key in "${!kwargs[@]}"; do
        args+="--$key ${kwargs[$key]} "
    done
    echo "$args"
}


# =====================================================
# 3. Run Function
# =====================================================
run_experiment() {
    local name=$1
    local mode=$2
    local args=$3
    
    echo "Running experiment: $name"
    echo "Arguments: $args"
    echo "----------------------------------------"
    
    echo $password | sudo -S -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" $mode $args
    echo $password | sudo -S -E env PATH="$PATH" chmod -R 777 ./experiments/
    echo "----------------------------------------"
}

# =====================================================
# 4. Define and Run Experiments
# =====================================================
# Arguments order reminder for configure_experiment:
# configure_experiment "model" "policy" num_episodes learning_rate additional_pct env_version reward_calc

# Arguments order for configure_experiment:
#  1. model_type          - e.g., "ddpg", "a2c"
#  2. policy_type         - e.g., "ddpgmlp", "mlp"
#  3. num_episodes        - e.g., 2000, 500
#  4. learning_rate       - e.g., 0.001, 0.01
#  5. additional_pct      - e.g., 0.5, 0.1
#  6. env_version        - e.g., 32, 33
#  7. reward_calculator  - e.g., 28, 29
#  8. experiment_name     - e.g., "train_20241209_233937"
#  9. action_strategy_version - e.g., 7
# 10. search_window - e.g., 60 (minutes)
# 11. span_size - e.g., 180 (seconds)
# 12. rule_frequency - e.g., 60 (minutes)
# 13. state_strategy_version - e.g., 8
# 14. mode - e.g., "train", "test"
# 15. alpha - e.g., 0.5
# 16. beta - e.g., 0.5  
# 17. fake_start_datetime - e.g., "2024-06-15_13:00:00"
# 18. n_steps - e.g., 144


# mode="test"

# args=$(configure_experiment  "recurrentppo" "lstm" 60 0.0001 1 32 57 "train_20250131_000213" 10 1440 600 1440 4 "$mode", 0.7 0.15 "2024-03-01_13:00:00")
# run_experiment "Dynamic env test" "$mode" "$args"

mode="train"

args=$(configure_experiment  "sac" "sacmlp" 4000 0.0001 0.5 32 57 "train_20250131_000213" 14 1440 3600 120 12 "$mode", 0.4 0.3 '2024-06-15_13:00:00' 12)
run_experiment "Dynamic env test" "$mode" "$args"

# args=$(configure_experiment  "recurrentppo" "lstm" 60 0.0001 1 32 57 "train_20250129_220331" 10 1440 600 1440 4 "$mode", 0.8 0.15)
# run_experiment "Dynamic env test" "$mode" "$args"
# mode="random_policy"
# args=$(configure_experiment  "recurrentppo" "lstm" 60 0.0001 1 32 57 "train_20250129_220331" 10 1440 600 1440 4 "$mode", 0.8 0.15)
# run_experiment "Dynamic env test" "$mode" "$args"

# mode="manual_policy"
# args=$(configure_experiment  "recurrentppo" "no_agent" 60 0.0001 1 32 57 "train_20250129_220331" 10 1440 600 1440 4 "$mode", 0.8 0.15)
# run_experiment "Dynamic env test" "$mode" "$args"

# args=$(configure_experiment "ppo" "mlp" 36 0.0005 0.5 32 49 "train_20250113_200347" 10 60 120 60 4 "$mode")
# run_experiment "Dynamic env test" "$mode" "$args"
# args=$(configure_experiment "recurrentppo" "lstm" 1000 0.0005 0.5 32 44 "train_20241223_143017" 8 60 180 60)
# run_experiment "" "train" "$args"
# args=$(configure_experiment "recurrentppo" "lstm" 500 0.001 0.5 32 41 "train_20241220_104215")
# run_experiment "" "test" "$args"
# args=$(configure_experiment "a2c" "mlp" 10 0.005 0.5 32 36 "retrain_20241215_000707")
# run_experiment "test1" "test" "$args"
# args=$(configure_experiment "a2c" "mlp" 10 0.005 0.5 32 35 "train_20241212_132453")
# run_experiment "test2" "test" "$args"
# args=$(configure_experiment "a2c" "no_agent" 10 0.005 0 32 35 "train_20241209_233937")
# run_experiment "running no agent manual_policy" "manual_policy" "$args"
# Template for new experiments:
#
# echo "Starting [Experiment Name]..."
# args=$(configure_experiment "model" "policy" episodes lr add_pct env reward)
# run_experiment "[Experiment Name]" "$args"

# Quick reference for common values:
# Models: "ddpg", "a2c"
# Policies: "ddpgmlp", "mlp"
# Episodes: 2000, 5000
# Learning rates: 0.001, 0.01
# Additional percentages: 0.5, 0.1
# Environment versions: 32, 33
# Reward calculator versions: 28, 29