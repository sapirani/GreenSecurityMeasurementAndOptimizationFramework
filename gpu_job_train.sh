#!/bin/bash

#SBATCH --partition main
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
# Arrays for parameter sweeps across SLURM array jobs
# Use SLURM_ARRAY_TASK_ID to index into these arrays

# IP identifier (1=132.72.81.184, 2=132.72.80.159)
ARGS_1=(
2
)

# Action space type (Action8 or Action12)
ARGS_2=(
"Action8"
)

# Learning rate
ARGS_3=(
"1e-4"
)

# Reward weights: alpha (energy), beta (alert), gamma (distribution)
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

# NEW: Using run_experiment.py with named arguments
# Config files provide defaults, CLI args override when needed
# Note: --model-name is not needed for train mode (creates new model)
/home/shouei/.conda/envs/py310_modelenv/bin/python3 -m SplunkResearch.src.run_experiment \
    --alpha-energy $CURR_ARG4 \
    --beta-alert $CURR_ARG5 \
    --gamma-dist $CURR_ARG6 \
    --hosts-num 100 \
    --alert-reward-method "AlertRewardWrapper2" \
    --distribution-reward-method "DistributionRewardWrapper" \
    --learning-rate $CURR_ARG3 \
    --mode "train" \
    --num-episodes 50000 \
    --ip $CURR_ARG1 \
    --action-type $CURR_ARG2