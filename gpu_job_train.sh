#!/bin/bash

#SBATCH --partition "RTX 3090"
#SBATCH --time 1-00:00:00
#SBATCH --job-name my_job
#SBATCH --output job-%A_%a.out
#SBATCH --mail-user=shouei@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8

#SBATCH --array=0-1%1

echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_ARRAY_TASK_ID:\t" $SLURM_ARRAY_TASK_ID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
nvidia-smi -L

module load anaconda
source activate py310_modelenv
IPs=("132.72.81.184" "132.72.80.159")
###############################################################
### PARAMETER CONFIGURATION
###############################################################

# 1. IP
ARGS_1=(
2
)
# 2. Action type
ARGS_2=(
"Action8"
)


ARGS_3=(
"1e-4"
)

ARGS_4=(
0.334
)
ARGS_5=(
0.333
)
ARGS_6=(
0.333
)

# --- SELECT PARAMS FOR CURRENT TASK ---
CURR_ARG1=${ARGS_1[$SLURM_ARRAY_TASK_ID]}
CURR_ARG2=${ARGS_2[$SLURM_ARRAY_TASK_ID]}
CURR_ARG3=${ARGS_3[$SLURM_ARRAY_TASK_ID]}
CURR_ARG4=${ARGS_4[$SLURM_ARRAY_TASK_ID]}
CURR_ARG5=${ARGS_5[$SLURM_ARRAY_TASK_ID]}
CURR_ARG6=${ARGS_6[$SLURM_ARRAY_TASK_ID]}
IP_INDEX=$(($CURR_ARG1 - 1))
IP=${IPs[$IP_INDEX]}
echo "--- Triggering Splunk Reset on $IP ---"
ssh -n -o StrictHostKeyChecking=no splunk@$IP "sudo /opt/splunk_data/reset_index.sh"

echo "Running: $CURR_ARG1 with params $CURR_ARG2 $CURR_ARG3 $CURR_ARG4 $CURR_ARG5 $CURR_ARG6"

###############################################################
### EXECUTION
###############################################################

# Note: I replaced the hardcoded 'DistributionRewardWrapper' with $CURR_ARG4
/home/shouei/.conda/envs/py310_modelenv/bin/python3 -m SplunkResearch.src.experiment_manager_new \
    "train_20260111202046_360000_steps" \
    $CURR_ARG4 $CURR_ARG5 $CURR_ARG6 \
    "100" \
    "1" \
    "AlertRewardWrapper2" \
    "DistributionRewardWrapper" \
    $CURR_ARG3 \
    0 \
    "train" \
    50000 \
    $CURR_ARG1 \
    1 \
    2 \
    $CURR_ARG2