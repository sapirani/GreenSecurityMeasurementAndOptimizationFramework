Compare diversity_policy metrics between a training experiment and its evaluation experiment for all relevant logtypes (_1 suffix).

Arguments: $ARGUMENTS should be two experiment directory paths or names: <train_experiment> <eval_experiment>
Example: /compare-diversity train_20260309155118 eval_post_training_20260310193533

Steps:
1. Resolve experiment directories under `SplunkResearch/host_*_experiments/runs/`. If full paths are given, use them directly. If just names, search for matching dirs.
2. List all tensorboard subdirectories ending with `_1` (excluding `train_1`) — these are the relevant logtypes with trigger=1.
3. For each `<logtype>_1` subdir, use Python with `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` to:
   - Load the train experiment's `<logtype>_1/` tensorboard dir
   - Load the eval experiment's `<logtype>_1/` tensorboard dir
   - Extract all scalar events matching `diversity_policy` tag
   - For train: use the **last 200 values** to compute mean and std
   - For eval: use **all values** to compute mean and std
   - Compute ratio = eval_mean / train_mean
4. Always activate conda first: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate py310_modelenv`
5. Print a formatted comparison table with columns: Logtype, Train mean, Train std, Eval mean, Eval std, Ratio
6. Highlight logtypes where the ratio is < 0.5 or > 2.0 (significant divergence)
7. Summarize findings: which logtypes shifted most, which disappeared, which appeared
