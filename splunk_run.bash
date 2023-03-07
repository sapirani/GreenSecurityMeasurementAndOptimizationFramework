#!/bin/bash
# regex hadoop: ^[^,\n]*,\d+\s+(?P<log_level>\w+)\s+\[(?P<node_name>[^\]]+)\]\s+(?P<class_name>[^:]+)(:\s)(?P<msg>.*)$
#run this bash like this: sudo -E env "PATH=$PATH" bash run.bash 
export SPLUNK_HOME=/opt/splunk
export PATH=$SPLUNK_HOME/bin:$PATH
CONFIG_FILE=/opt/splunk/etc/apps/SA-Eventgen/local/eventgen.conf
PYTHON_SCRIPT=scanner.py
TIME_MULTIPLE_VALUES=(0.5 1 1.5)  # Values to use for the timeMultiple parameter

for time_multiple in "${TIME_MULTIPLE_VALUES[@]}"; do
    # Modify the timeMultiple parameter in the config file
    sed -i "s/^timeMultiple = .*/timeMultiple = $time_multiple/" "$CONFIG_FILE"

    # Run the Python script in the background
    # ~/micromamba/bin/python "$PYTHON_SCRIPT" &
    ~/miniconda3/bin/python "$PYTHON_SCRIPT" &


    # Wait for the Python script to finish running
    wait

    # Wait for a few seconds before continuing the loop
    sleep 5
    sudo chmod -R 777 Dell\ Inc.\ Linux\ 5.15.0-67-generic/
done
