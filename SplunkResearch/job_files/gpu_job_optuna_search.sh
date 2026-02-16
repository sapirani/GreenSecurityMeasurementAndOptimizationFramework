#!/bin/bash

#SBATCH --partition "rtx3090"
#SBATCH --time 3-00:00:00
#SBATCH --job-name optuna_hpo
#SBATCH --output job-%J.out
#SBATCH --mail-user=shouei@post.bgu.ac.il
#SBATCH --mail-type=ALL
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16

echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
nvidia-smi -L

module load anaconda
source activate py310_modelenv

###############################################################
### PARAMETER CONFIGURATION
###############################################################

# Splunk host IP identifier (1=132.72.81.184, 2=132.72.80.159)
IP_ID=2

# Parallel environments per trial (SubprocVecEnv workers)
# Rule of thumb: NUM_ENVS <= cpus-per-task - 2
NUM_ENVS=14

# Number of Optuna trials
N_TRIALS=50

# Full training episode count (used for best-trial retrain only;
# each trial runs a fixed 50K episodes = 600K steps)
NUM_EPISODES=150000

# Study name — reuse the same name to resume an interrupted search
STUDY_NAME="optuna_hpo_v2_ip${IP_ID}"

# Set to "--no-retrain-best" to skip automatic retraining of the best trial
RETRAIN_FLAG=""

echo "Running Optuna HPO search (multi-model, multi-reward)"
echo "  IP_ID=$IP_ID, NUM_ENVS=$NUM_ENVS, N_TRIALS=$N_TRIALS"
echo "  Trial budget: 50K episodes (600K steps) each"
echo "  Retrain budget: $NUM_EPISODES episodes"
echo "  STUDY_NAME=$STUDY_NAME"

###############################################################
### EXECUTION
###############################################################

/home/shouei/.conda/envs/py310_modelenv/bin/python3 -m SplunkResearch.src.run_experiment \
    --mode optuna_search \
    --ip $IP_ID \
    --num-envs $NUM_ENVS \
    --num-episodes $NUM_EPISODES \
    --n-trials $N_TRIALS \
    --optuna-study-name $STUDY_NAME \
    --hosts-num 100 \
    --action-type "SmoothTrigger" \
    --reward-mode legacy \
    $RETRAIN_FLAG
