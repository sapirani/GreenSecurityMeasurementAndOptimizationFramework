Find the experiment directory for SLURM job $ARGUMENTS and start a TensorBoard instance with SSH tunnel instructions.

Steps:
1. Run `scontrol show job <job_id>` to get the job's StdOut file path and submit time
2. Read the first 30 lines of the job output file to identify the Splunk host IP (look for lines like "Triggering Splunk Reset on X.X.X.X")
3. Find the experiment directory under `SplunkResearch/host_{ip}_experiments/runs/` that matches the job's start date (use `ls -td` with a glob pattern based on the submit date from scontrol, format: train_YYYYMMDD*)
4. Verify the tensorboard subdirectory exists inside that experiment dir
5. Check for already-running tensorboard processes (`ps aux | grep tensorboard`) to find which ports are in use
6. Run `hostname -f` to get the current machine's FQDN — store this as CURRENT_HOST
7. Start tensorboard using the conda env binary (`/home/shouei/.conda/envs/py310_modelenv/bin/tensorboard`) on the next available port (starting from 6006), with `--bind_all`
8. Verify it started successfully by checking the log output
9. Print the experiment directory path and the SSH tunnel command using CURRENT_HOST (from step 6):
   ```
   ssh -L <port>:<CURRENT_HOST>:<port> shouei@<CURRENT_HOST>
   ```
   Then open http://localhost:<port>
