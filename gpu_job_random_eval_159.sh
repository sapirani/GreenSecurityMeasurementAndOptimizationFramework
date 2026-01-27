#!/bin/bash

#SBATCH --partition main
#SBATCH --time 1-00:00:00
#SBATCH --job-name splunk_eval_rnd_159
#SBATCH --output job_rand-159-%A_%a.out
#SBATCH --mail-user=shouei@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --gpus=0
#SBATCH --mem=60G
#SBATCH --cpus-per-task=6

#SBATCH --array=0-4%1

echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_ARRAY_TASK_ID:\t" $SLURM_ARRAY_TASK_ID

module load anaconda
source activate py310_modelenv

ARGS_1=(
0.1 0.25 0.5 0.75 1 0.1 0.25 0.5 0.75 1
)

CURR_ARG1=${ARGS_1[$SLURM_ARRAY_TASK_ID]}


echo "--- Triggering Splunk Reset on 132.72.80.159 ---"
ssh -n -o StrictHostKeyChecking=no splunk@132.72.80.159 "sudo /opt/splunk_data/reset_index.sh"

if [ $? -ne 0 ]; then
    echo "CRITICAL ERROR: Splunk reset failed. Aborting."
    exit 1
fi

echo "Running Experiment with: Arg1=$CURR_ARG1"

# --- STEP 3: EXECUTION ---
/home/shouei/.conda/envs/py310_modelenv/bin/python3 -m SplunkResearch.src.experiment_manager_new \
####    "train_20260131230226_600000_steps" \
    "train_20260201112425_600000_steps" \
    0.2 0.4 0.4 \
    100 \
    $CURR_ARG1 \
    "AlertRewardWrapper2" \
    "DistributionRewardWrapper" \
    "1e-4" \
    1 \
    "eval_post_training" \
    50 \
    2