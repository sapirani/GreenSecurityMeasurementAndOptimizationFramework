#!/bin/bash
# before running this script run this command in terminal: ~/miniconda3/bin/python "change_disabled_state.py" lite
#run this bash like this: sudo -E env "PATH=$PATH" bash splunk_run.bash
export SPLUNK_HOME=/opt/splunk
export PATH=$SPLUNK_HOME/bin:$PATH
CONFIG_FILE=/opt/splunk/etc/apps/SA-Eventgen/local/eventgen.conf
PYTHON_SCRIPT=scanner.py
TIME_MULTIPLE_VALUES=(0.0008) ###(0.5 1 1.5)  # Values to use for the timeMultiple parameter

sudo -E env "PATH=$PATH" splunk start
wait
~/miniconda3/bin/python "change_disabled_state.py" lite &
# Wait for the Python script to finish running
wait
sleep 5
sudo -E env "PATH=$PATH" splunk stop
sudo -E env "PATH=$PATH" splunk clean eventdata -index eventgen -f

today=$(date +%Y-%m-%d -d "tomorrow")
hour=0
num_of_experiments=8
first_start_minute=1
last_start_minute=46
step=15
search_delay=20

for i in $(seq 1 $num_of_experiments); do
    if [ "$hour" -eq 24 ]; then
        today=$(date +%Y-%m-%d -d "tomorrow")
        hour=0
    fi
    for m in $(seq $first_start_minute $step $last_start_minute); do
        for time_multiple in "${TIME_MULTIPLE_VALUES[@]}"; do
            # Modify the timeMultiple parameter in the config file
            sudo date -s "$today $hour:$m:00"
            sed -i "s/^timeMultiple = .*/timeMultiple = $time_multiple/" "$CONFIG_FILE"

            running_time=$(( 60 + $search_delay - $m )) # change me!!!
            sed -i "s/RUNNING_TIME = .*/RUNNING_TIME = $running_time/" "program_parameters.py"

            # Run the Python script in the background
            # ~/micromamba/bin/python "$PYTHON_SCRIPT" &
            ~/miniconda3/bin/python "$PYTHON_SCRIPT" &
            # Wait for the Python script to finish running
            wait

            # Wait for a few seconds before continuing the loop
            sleep 5
            sudo chmod -R 777 Dell\ Inc.\ Linux\ 5.15.0-70-generic/

            sudo -E env "PATH=$PATH" splunk start
            wait
            # sleep 30
            ~/miniconda3/bin/python "extract_jobs_data.py" &
            # Wait for the Python script to finish running
            wait
            # change searches
            if [ "$m" -eq $last_start_minute ] && [ "$i" -eq $(( $num_of_experiments / 2)) ]; then
                ~/miniconda3/bin/python "change_disabled_state.py" heavy &
                # Wait for the Python script to finish running
                wait
            fi

            sleep 5
            sudo -E env "PATH=$PATH" splunk stop
            sudo -E env "PATH=$PATH" splunk clean eventdata -index eventgen -f
            (( hour++ ))
        done
    done
done