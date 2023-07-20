#!/bin/bash
# # cd in SplunkResearch and run this bash like this: screen -L sudo bash splunk_run.bash
# Set environment variables
export SPLUNK_HOME=/opt/splunk
export PATH=$SPLUNK_HOME/bin:$PATH
export PATH=/home/shouei/anaconda3/bin:$PATH
export PATH=/home/shouei/anaconda3/envs/py38/bin:$PATH
PYTHON_SCRIPT=model/attack_model.py

eval "$(conda shell.bash hook)"
source activate /home/shouei/anaconda3/envs/py38
sudo -E env PATH="$PATH" python "$PYTHON_SCRIPT"
wait
