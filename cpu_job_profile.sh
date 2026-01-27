#!/bin/bash

#SBATCH --partition main
#SBATCH --time 1-00:00:00
#SBATCH --job-name my_job
#SBATCH --output job-%A_%a.out
#SBATCH --mail-user=shouei@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --gpus=0
#SBATCH --mem=24G
#SBATCH --cpus-per-task=6

#SBATCH --array=0-0%1

echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_ARRAY_TASK_ID:\t" $SLURM_ARRAY_TASK_ID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
nvidia-smi -L

module load anaconda
source activate py310_modelenv
IPs=("132.72.81.184" "132.72.80.159", "132.72.81.150")
###############################################################
### PARAMETER CONFIGURATION
###############################################################

# 1. IP
ARGS_1=(
1
)

# --- SELECT PARAMS FOR CURRENT TASK ---
CURR_ARG1=${ARGS_1[$SLURM_ARRAY_TASK_ID]}

IP_INDEX=$(($CURR_ARG1 - 1))
IP=${IPs[$IP_INDEX]}
echo "--- Triggering Splunk Reset on $IP ---"
ssh -n -o StrictHostKeyChecking=no splunk@$IP "sudo /opt/splunk_data/reset_index.sh"

echo "Running: $CURR_ARG1"

###############################################################
### EXECUTION
###############################################################

# Note: I replaced the hardcoded 'DistributionRewardWrapper' with $CURR_ARG4
/home/shouei/.conda/envs/py310_modelenv/bin/python3 -m SplunkResearch.src.energy_profile_final $CURR_ARG1