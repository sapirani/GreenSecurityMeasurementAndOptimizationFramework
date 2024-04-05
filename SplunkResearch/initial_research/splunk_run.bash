#!/bin/bash
# # cd in SplunkResearch and run this bash like this: screen -L sudo bash splunk_run.bash
# Set environment variables
export SPLUNK_HOME=/opt/splunk
export PATH=$SPLUNK_HOME/bin:$PATH
export PATH=/home/shouei/anaconda3/bin:$PATH
export PATH=/home/shouei/anaconda3/envs/py38/bin:$PATH

PYTHON_SCRIPT=../scanner.py
props_path="/opt/splunk/etc/users/shouei/search/local/props.conf"
saved_searches_path="/opt/splunk/etc/users/shouei/search/local/savedsearches.conf"
limits_path="/opt/splunk/etc/system/local/limits.conf"
EXTRACTION_STATES=("enable")  # Extraction rule states to use
TIME_WINDOWS=(12)  # Time windows to use (in hours)
running_time=1 #in minutes
search_frequency=40 # in minutes

# Provided date
latest_date_string="06/15/2023 08:00:00 AM"
latest_timestamp=$(date -d"$latest_date_string" +%s)
echo "Timestamp of latest date: $latest_timestamp"

source /opt/splunk/bin/setSplunkEnv
eval "$(conda shell.bash hook)"
source activate /home/shouei/anaconda3/envs/py38

sed -i "s/RUNNING_TIME = .*/RUNNING_TIME = $running_time/" "../program_parameters.py"

sed -i 's/^max_searches_per_process = .*/max_searches_per_process = 1/' $limits_path

sudo -E env PATH="$PATH" python splunk_tools/cron_update.py $saved_searches_path "*/$search_frequency * * * *"

for extraction_state in "${EXTRACTION_STATES[@]}"; do
    sudo -E env PATH="$PATH" python splunk_tools/change_extraction_state.py $props_path $extraction_state
    for time_window in "${TIME_WINDOWS[@]}"; do
    # Convert provided date to timestamp


        # Get date of hours before the provided date
        date_before=$(date -d"$latest_date_string - $time_window hours" +%s)
        echo "Date of $time_window hours before the provided date: $date_before"

        sudo -E env PATH="$PATH" python splunk_tools/time_range_modify.py $saved_searches_path $date_before $latest_timestamp
        
        # for i in $(seq 0 $((num_of_experiments-1))); do
        date

        sudo -E env "PATH=$PATH" splunk restart
        wait

        echo "Extraction state: $extraction_state"
        echo "Time window: $time_window"
        # echo "Experiment number: $i"

        sudo -E env PATH="$PATH" python "$PYTHON_SCRIPT"
        wait
        
        # Wait for a few seconds before continuing the loop
        sleep 5
        chmod -R 777 VMware\,\ Inc.\ Linux\ 3.10.0-1160.88.1.el7.x86_64/

        latest_folder=$(ls -td -- ../../GreenSecurity-FirstExperiment/VMware\,\ Inc.\ Linux\ 3.10.0-1160.88.1.el7.x86_64/Splunk\ Enterprise\ SIEM/Power\ Saver\ Plan/One\ Scan/* | head -n 1)
        # echo "Latest folder: $latest_folder"
        # sudo -E env "PATH=$PATH" splunk search "index=main where earliest=-$60m@m" -output csv -maxout 0 -auth shouei: > "${latest_folder}/logs.csv"
        python "splunk_tools/jobs_info_extractor.py" "$latest_folder" "$running_time" &
        wait
        
        # done
    done
done

wait
sed -i 's/^max_searches_per_process = .*/max_searches_per_process = 500/' $limits_path