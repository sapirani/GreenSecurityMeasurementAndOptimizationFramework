#!/bin/bash

# Set environment variables
export SPLUNK_HOME=/opt/splunk
export PATH=$SPLUNK_HOME/bin:$PATH
export PATH=/home/shouei/anaconda3/bin:$PATH
export PATH=/home/shouei/anaconda3/envs/py38/bin:$PATH

PYTHON_SCRIPT=scanner.py
props_path="/opt/splunk/etc/users/shouei/search/local/props.conf"
saved_searches_path="/opt/splunk/etc/users/shouei/search/local/savedsearches.conf"

EXTRACTION_STATES=("enable" "disable")  # Extraction rule states to use
TIME_WINDOWS=("-24h" "-48h")  # Time windows to use
num_of_experiments=1
# cron_schedule="*/10 * * * *"
running_time=15

source /opt/splunk/bin/setSplunkEnv
eval "$(conda shell.bash hook)"
source activate /home/shouei/anaconda3/envs/py38

date

sudo -E env PATH="$PATH" python cron_update.py $saved_searches_path "*/10 * * * *"

for extraction_state in "${EXTRACTION_STATES[@]}"; do
    sudo -E env PATH="$PATH" python change_extraction_state.py $props_path $extraction_state
    for time_window in "${TIME_WINDOWS[@]}"; do
        sudo -E env PATH="$PATH" python win_time_modify.py $saved_searches_path $time_window "now"
        
        for i in $(seq 0 $((num_of_experiments-1))); do
            sudo -E env "PATH=$PATH" splunk restart
            wait

            echo "Extraction state: $extraction_state"
            echo "Time window: $time_window"
            echo "Experiment number: $i"

            # Modify the extraction state and time window
            sed -i "s/RUNNING_TIME = .*/RUNNING_TIME = $running_time/" "program_parameters.py"

            sudo -E env PATH="$PATH" python "$PYTHON_SCRIPT"
            wait
            latest_folder=$(ls -td -- ../GreenSecurity-FirstExperiment/VMware\,\ Inc.\ Linux\ 3.10.0-1160.88.1.el7.x86_64/Splunk\ Enterprise\ SIEM/Power\ Saver\ Plan/One\ Scan/* | head -n 1)
            
            # Wait for a few seconds before continuing the loop
            sleep 5
            chmod -R 777 VMware\,\ Inc.\ Linux\ 3.10.0-1160.88.1.el7.x86_64/
            sudo -E env "PATH=$PATH" splunk search "index=main where earliest=-$60m@m" -output csv -maxout 0 -auth shouei:sH231294 > "${latest_folder}/logs.csv"
            python "extract_jobs_data.py" "$latest_folder" &
            wait
            
        done
    done
done

wait



# #!/bin/bash
# # before running this script run this command in terminal: ~/miniconda3/bin/python "change_disabled_state.py" lite
# #run this bash like this: screen -L sudo bash splunk_run.bash
# export SPLUNK_HOME=/opt/splunk
# export PATH=$SPLUNK_HOME/bin:$PATH
# export PATH=/home/shouei/anaconda3/bin:$PATH
# export PATH=/home/shouei/anaconda3/envs/py38:$PATH
# CONFIG_FILE=/opt/splunk/etc/apps/SA-Eventgen/local/eventgen.conf
# PYTHON_SCRIPT=scanner.py
# TIME_MULTIPLE_VALUES=(0.0008) ###(0.5 1 1.5)  # Values to use for the timeMultiple parameter

# # sudo -E env "PATH=$PATH" splunk start
# # wait
# # python "change_disabled_state.py" lite &
# # Wait for the Python script to finish running
# # wait
# # sleep 5
# # sudo -E env "PATH=$PATH" splunk stop
# # sudo -E env "PATH=$PATH" splunk clean eventdata -index eventgen -f

# # today=$(date +%Y-%m-%d -d "tomorrow")
# # source green_sec_env/bin/activate
# source /opt/splunk/bin/setSplunkEnv

# eval "$(conda shell.bash hook)"
# source activate /home/shouei/anaconda3/envs/py38
# num_of_experiments=4
# # first_start_minute=1
# # last_start_minute=60
# # step=60
# # search_delay=20
# date
# for i in $(seq 0 $((num_of_experiments-1))); do
#     # if [ "$hour" -eq 24 ]; then
#     #     today=$(date +%Y-%m-%d -d "tomorrow")
#     #     hour=0
#     # fi
#     # for m in $(seq $first_start_minute $step $last_start_minute); do
#     # for time_multiple in "${TIME_MULTIPLE_VALUES[@]}"; do
#     # Modify the timeMultiple parameter in the config file
#     # sudo date -s "$today $hour:$m:00"
#     # sed -i "s/^timeMultiple = .*/timeMultiple = $time_multiple/" "$CONFIG_FILE"
    
#     running_time=6 # change me!!!
#     sed -i "s/RUNNING_TIME = .*/RUNNING_TIME = $running_time/" "program_parameters.py"

#     # Run the Python script in the background
#     # ~/micromamba/bin/python "$PYTHON_SCRIPT" &
#     python "$PYTHON_SCRIPT"
#     # ~/miniconda3/bin/python "$PYTHON_SCRIPT" &
#     # Wait for the Python script to finish running
#     wait
#     # get the name of the latest folder a given path
#     latest_folder=$(ls -td -- ../GreenSecurity-FirstExperiment/VMware\,\ Inc.\ Linux\ 3.10.0-1160.88.1.el7.x86_64/Splunk\ Enterprise\ SIEM/Power\ Saver\ Plan/One\ Scan/* | head -n 1)
#     # Wait for a few seconds before continuing the loop
#     sleep 5
#     chmod -R 777 VMware\,\ Inc.\ Linux\ 3.10.0-1160.88.1.el7.x86_64/
#     sudo -E env "PATH=$PATH" splunk search "index=main where earliest=-$60m@m" -output csv -maxout 0 -auth shouei:sH231294 > "${latest_folder}/logs.csv"
#     # sudo -E env "PATH=$PATH" splunk start
#     # wait
#     # sleep 30
#     # TODO add -u to python command or write sys.stdout.flush() in python script 
#     python "extract_jobs_data.py" "$latest_folder" &
#     # Wait for the Python script to finish running
#     # wait
#     # change searches
#     # if [ "$m" -eq $last_start_minute ] && [ "$i" -eq $(( $num_of_experiments / 2)) ]; then
#     #     python "change_disabled_state.py" heavy &
#     #     # Wait for the Python script to finish running
#     wait
#     # fi

#     sleep 5
#     # sudo -E env "PATH=$PATH" splunk stop
#     # sudo -E env "PATH=$PATH" splunk clean eventdata -index eventgen -f
#     # (( hour++ ))
#     # done
#     # done




            
# done
# wait