#!/bin/bash
# # cd in SplunkResearch and run this bash like this: screen -L bash model/model_run.bash after cd to SplunkReasearch dir
# Set environment variables
# export SPLUNK_HOME=/opt/splunk
# export PATH=$SPLUNK_HOME/bin:$PATH
# export PATH=/home/shouei/anaconda3/bin:$PATH
# export PATH=/home/shouei/anaconda3/envs/py38/bin:$PATH
export PATH=$PATH:/home/shouei/local/dmidecode
PYTHON_SCRIPT=src/main_2.py
limits_path="/opt/splunk/etc/system/local/limits.conf"
saved_searches_path="/opt/splunk/etc/users/shouei/search/local/savedsearches.conf"
# conda init bash
# conda activate /home/shouei/anaconda3/envs/py38
# which python
echo sH231294 | sudo -S sed -i 's/^max_searches_per_process = .*/max_searches_per_process = 1/' $limits_path


echo "running"
# if $2 == "test" run the test script else only train
if [ "$2" == "test" ]; then
    echo sH231294 | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" $1
    wait
else
    echo "$1" "$2"
    echo sH231294 | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" $1 $2 $3
    wait
fi
if [ "$2" == "test" ] || [ "$3" == "test" ]; then
    echo sH231294 | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" test last $3
    wait
fi
if ["$4" == "baseline"]; then
    echo sH231294 | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" baseline last $5 passive
    echo sH231294 | sudo -S  -E env PATH="$PATH" python3 "$PYTHON_SCRIPT" baseline last $5 random
    wait
fi
echo sH231294 | sudo -S chmod -R 777 VMware\,\ Inc.\ Linux\ 3.10.0-1160.88.1.el7.x86_64/
echo sH231294 | sudo chmod -R 777 ./experiments/