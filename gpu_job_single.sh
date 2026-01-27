#!/bin/bash



################################################################################################

### sbatch configuration parameters must start with #SBATCH and must precede any other commands.

### To ignore, just add another # - like so: ##SBATCH

################################################################################################



#SBATCH --partition main ### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes

#SBATCH --time 1-00:00:00 ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS

#SBATCH --job-name my_job ### name of the job

#SBATCH --output job-%J.out ### output log for running job - %J for job number

#SBATCH --mail-user=shouei@post.bgu.ac.il ### user's email for sending job status messages

#SBATCH --mail-type=ALL ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE



#SBATCH --gpus=0 ### number of GPUs, allocating more than 1 requires IT team's permission

#SBATCH --mem=60G ### ammount of RAM memory, allocating more than 60G requires IT team's permission

#SBATCH --cpus-per-task=8 ### number of CPU cores, allocating more than 10G requires IT team's permission

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_ARRAY_TASK_ID:\t" $SLURM_ARRAY_TASK_ID  ### Good to log which array index this is
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
nvidia-smi -L

### Start your code below ####
module load anaconda
source activate py310_modelenv

# Note: The arguments $1, $2, etc. passed to 'sbatch' will be the SAME for all jobs in the array.
# If you need them to change per run, use $SLURM_ARRAY_TASK_ID to select parameters inside the script.

/home/shouei/.conda/envs/py310_modelenv/bin/python3 -m SplunkResearch.src.experiment_manager_new train_20260119001706_600000_steps 0.3334 0.3334 0.3334 25 0.5 AlertRewardWrapper2 DistributionRewardWrapper "3e-4" 0 eval_post_training 20 
echo "--- Triggering Remote Splunk Reset ---"
ssh -o StrictHostKeyChecking=no splunk@132.72.80.159 "sudo /opt/splunk_data/reset_index.sh"
echo "--- Splunk Reset Complete ---"