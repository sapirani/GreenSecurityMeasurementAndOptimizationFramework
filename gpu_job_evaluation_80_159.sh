#!/bin/bash

#SBATCH --partition main
#SBATCH --time 1-00:00:00
#SBATCH --job-name splunk_train_159
#SBATCH --output job_159-%A_%a.out
#SBATCH --mail-user=shouei@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --gpus=0
#SBATCH --mem=60G
#SBATCH --cpus-per-task=6

# RUNS EXP 0 - 24 (25 Jobs Total)
#SBATCH --array=0-24%1

echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_ARRAY_TASK_ID:\t" $SLURM_ARRAY_TASK_ID
nvidia-smi -L

module load anaconda
source activate py310_modelenv

# --- STEP 1: REMOTE RESET (Server 159) ---
echo "--- Triggering Splunk Reset on 132.72.80.159 ---"
ssh -n -o StrictHostKeyChecking=no splunk@132.72.80.159 "sudo /opt/splunk_data/reset_index.sh"

if [ $? -ne 0 ]; then
    echo "CRITICAL ERROR: Splunk reset failed. Aborting."
    exit 1
fi

# --- STEP 2: PARAMETERS (Exp 0-12) ---

# Exp 0-4 (10, x) | Exp 5-9 (25, x) | Exp 10-12 (50, 0.1-0.5)
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

echo "Running Experiment with: Arg2=$CURR_ARG2 Arg3=$CURR_ARG3"

# --- STEP 3: EXECUTION ---
/home/shouei/.conda/envs/py310_modelenv/bin/python3 -m SplunkResearch.src.run_experiment \
    --model-name "train_20260205101454_600000_steps" \
    --mode "eval_post_training" \
    --alpha-energy 0.334 \
    --beta-alert 0.333 \
    --gamma-dist 0.333 \
    --hosts-num $CURR_ARG2 \
    --additional-percentage $CURR_ARG3 \
    --alert-reward-method "AlertRewardWrapper2" \
    --distribution-reward-method "DistributionRewardWrapper" \
    --learning-rate "1e-4" \
    --num-episodes 40 \
    --ip 2