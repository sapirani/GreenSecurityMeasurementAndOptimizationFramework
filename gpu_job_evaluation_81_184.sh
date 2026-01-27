#!/bin/bash

#SBATCH --partition main
#SBATCH --time 1-00:00:00
#SBATCH --job-name splunk_train_184
#SBATCH --output job_184-%A_%a.out
#SBATCH --mail-user=shouei@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --gpus=0
#SBATCH --mem=60G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-24%1
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_ARRAY_TASK_ID:\t" $SLURM_ARRAY_TASK_ID
nvidia-smi -L

module load anaconda
source activate py310_modelenv



# --- STEP 2: PARAMETERS (Exp 13-24) ---

# Exp 13-14 (50, 0.75-1.0) | Exp 15-19 (75, x) | Exp 20-24 (100, x)
# ARGS_2 (Sizes/Dimensions)
ARGS_2=(
10 10 10
50 50 50
100 100 100
)

# ARGS_3 (Rates/Factors)
ARGS_3=(
0.1 0.5 1.0
0.1 0.5 1.0
0.1 0.5 1.0
)

CURR_ARG2=${ARGS_2[$SLURM_ARRAY_TASK_ID]}
CURR_ARG3=${ARGS_3[$SLURM_ARRAY_TASK_ID]}
# --- STEP 1: REMOTE RESET (Server 184) ---
echo "--- Triggering Splunk Reset on 132.72.81.184 ---"
ssh -n -o StrictHostKeyChecking=no splunk@132.72.81.184 "sudo /opt/splunk_data/reset_index.sh"
echo "Running Experiment with: Arg2=$CURR_ARG2 Arg3=$CURR_ARG3"

# --- STEP 3: EXECUTION ---
/home/shouei/.conda/envs/py310_modelenv/bin/python3 -m SplunkResearch.src.experiment_manager_new \
    "train_20260205095228_600000_steps" \
    0.334 0.333 0.333 \
    $CURR_ARG2 \
    $CURR_ARG3 \
    "AlertRewardWrapper2" \
    "DistributionRewardWrapper" \
    "1e-4" \
    0 \
    "eval_post_training" \
    40 \
    1 \
    1 \
    1