# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research framework with two main components:
1. **Process-level energy/resource measurement** (`scanner.py`, `utils/`, `application_logging/`) — profiles CPU, memory, disk, network, and energy for arbitrary programs (Windows/POSIX)
2. **DRL adversarial log injection on Splunk** (`SplunkResearch/`) — trains RL agents (Stable Baselines 3) to inject logs into Splunk, maximizing detection rule CPU cost while staying stealthy

## Running Experiments

All commands run from the **project root** (required for `utils` and `application_logging` imports).

```bash
# Train a new model
python -m SplunkResearch.src.run_experiment \
    --mode train --model-type sac --num-episodes 50000 --ip 1

# Evaluate a trained model
python -m SplunkResearch.src.run_experiment \
    --mode eval_post_training --model-name "train_20260205101454_600000_steps" \
    --num-episodes 40 --ip 1

# Retrain from existing model
python -m SplunkResearch.src.run_experiment \
    --mode retrain --model-name "train_20260205101454_600000_steps" --ip 1

# Submit to SLURM cluster
sbatch gpu_job_train.sh

# TensorBoard
tensorboard --logdir SplunkResearch/host_X_experiments/runs/
```

### CLI Arguments

| Argument | Type | Description |
|---|---|---|
| `--mode` | `{train, eval_post_training, retrain}` | Experiment mode |
| `--model-type` | `{ppo, a2c, dqn, sac, td3, recurrent_ppo}` | RL algorithm |
| `--model-name` | str | Model to load (without `.zip`); required for eval/retrain |
| `--policy-type` | str | Policy network type (e.g., `MlpPolicy`) |
| `--num-episodes` | int | Number of episodes |
| `--learning-rate` | float | Learning rate |
| `--ip` | int | Splunk host identifier (1, 2, 3...) — maps to `SPLUNK_HOST_{ip}` env var |
| `--action-type` | `{Action8, SoftmaxDistribution, SmoothTrigger, SmoothTriggerVolume, SoftmaxDistributionVolume}` | Action space type |
| `--hosts-num` | int | Percentage of hosts to use (0-100) |
| `--additional-percentage` | float | Additional percentage for log generation |
| `--alpha-energy` | float | Energy reward weight |
| `--beta-alert` | float | Alert reward weight |
| `--gamma-dist` | float | Distribution reward weight |
| `--alert-reward-method` | str | Alert reward calculation method |
| `--distribution-reward-method` | str | Distribution reward calculation method |
| `--random-agent` | flag | Use random agent instead of trained model |
| `--test-experiment` | flag | Test mode (disables injection) |
| `--eval-during-training` / `--no-eval-during-training` | flag | Enable/disable eval callback during training (default: enabled) |
| `--log-level` | `{DEBUG, INFO, WARNING, ERROR, CRITICAL}` | Logging level |

## Running Tests

```bash
python -m unittest tests/scanner_tests.py
```

## Installing the Custom Gym Environment

```bash
cd SplunkResearch/src/custom_splunk && pip install -e .
```

Dependencies: `requirements/requirements_all_components.txt`

## Configuration System

Three-layer hierarchy (each layer overrides the previous):
1. `SplunkResearch/config/default.yaml` — all defaults
2. `SplunkResearch/config/secrets.yaml` — credentials (gitignored; copy from `secrets.yaml.example`)
3. CLI arguments — runtime overrides

Access in code via singleton: `from config import config; config.get('reward.beta', 0.33)`

Runtime secrets (Splunk credentials, HEC tokens) are in `SplunkResearch/src/.env` (gitignored).

### Key Config Sections

- **`reward.*`**: `alpha` (energy weight), `beta` (alert weight), `gamma` (distribution weight), `alert_method`, `distribution_method`, `use_stationary_scaling`, `expected_alerts_per_rule`, `rule_names`
- **`action.*`**: `softmax_temperature`, `base_log_count`, `sigmoid_sharpness`, `diversity_factor`
- **`training.*`**: `learning_rate`, `num_episodes`, `device`, `sac.*` (SAC-specific: `learning_starts`, `buffer_size`, `batch_size`, `policy_net_arch`)
- **`environment.*`**: `action_type` (default `Action8`), `state_type` (default `StateWrapper7`), `hosts_percentage`, `is_mock`, `log_types`
- **`evaluation.*`**: `n_eval_episodes`, `eval_freq`, `deterministic`
- **`callbacks.eval.*`**: `enabled` (run eval during training, default `true`), `eval_freq`, `deterministic`

## Architecture

### DRL Framework (`SplunkResearch/src/`)

**Entry point**: `run_experiment.py` → `ExperimentManager` (in `experiment_manager_new.py`)

**Experiment modes**: `train` creates a new model and trains it; `eval_post_training` loads a saved model and evaluates it; `retrain` loads a saved model and continues training.

**Wrapper composition order** (inner to outer, as applied in `ExperimentManager.create_environment()`):

```
SplunkEnv (base env)
 └→ Action wrapper          (create_action_wrapper)
   └→ DistributionRewardWrapper  (if use_distribution_reward)
     └→ BaseRuleExecutionWrapperWithPrediction
       └→ EnergyRewardWrapper    (if use_energy_reward)
         └→ AlertRewardWrapper   (create_alert_reward_wrapper)
           └→ TimeWrapper
             └→ StateWrapper     (create_state_wrapper) ← outermost
```

**Strategy pattern with registries:**

| Module | ABC(s) | Registry | Factory |
|---|---|---|---|
| `action_interpreters.py` | `ActionInterpreter` | `ACTION_INTERPRETER_REGISTRY` | `create_action_wrapper()` |
| `state_interpreters.py` | `StateInterpreter` | `STATE_INTERPRETER_REGISTRY` | `create_state_wrapper()` |
| `reward_interpreters.py` | `AlertSignal`, `AlertNormalizer` | `ALERT_REWARD_REGISTRY` | `create_alert_reward_wrapper()` |

**Shared episode state**: `EpisodeSharedState` dataclass in `wrappers/shared_state.py`; wrappers access via `env.unwrapped.field_name` (property aliases on SplunkEnv).

**Supporting modules**:
- `splunk_tools.py` — Splunk REST API (saved search dispatch, profiling, index management)
- `log_generator.py` — Template-based fake Windows Event Log generation
- `time_manager.py` — Episode time window management
- `callbacks.py` — SB3 callbacks: `CustomTensorboardCallback`, `HParamsCallback`, `CustomEvalCallback3`, `SplunkLicenceCheckCallback`
- `config.py` — YAML config singleton with dot-notation access

### Experiment Lifecycle

- Experiment statuses: `running` → `completed` | `failed` | `interrupted` | `crashed`
- Signal handlers (SIGTERM/SIGINT) mark experiments as `interrupted`
- Stale experiments (>48h running) auto-marked as `crashed` on `ExperimentManager` init
- Models saved: `final.zip`, `best_model.zip`, `replay_buffer.pkl` (SAC/TD3/DDPG), checkpoints (auto-pruned, last 3 kept)

### Measurement Framework (project root)

- `scanner.py` — Periodic resource profiling with Elasticsearch logging
- `utils/` — Shared utilities (constants, environment impact calculations)
- `application_logging/` — Elasticsearch-based logging infrastructure

### Experiment Directory Layout

```
SplunkResearch/host_{ip}_experiments/
├── baseline/                    # Shared baseline measurements
├── experiments.csv              # Index with git_hash, status columns
└── runs/{experiment_name}/      # Self-contained per-experiment
    ├── config.json, experiment.log, models/, tensorboard/, results/
```

## Key Gotchas

- **Must run from project root**: `utils/` and `application_logging/` are root-level packages, not under `SplunkResearch/src/`
- **Namespace package shadowing**: `src/custom_splunk/` (outer) shadows the editable-installed inner `custom_splunk` package when `src` is on `sys.path`. Fixed by `src/custom_splunk/__init__.py` that redirects `__path__`
- **Model loading fallback**: `run_experiment.py` tries `runs/{name}/models/final.zip` first, then legacy `models/{name}`
- **SB3 TensorBoard naming**: `model.learn()` uses `tb_log_name="train"` (not experiment name) to avoid `_0`/`_1` subdirs
- **`--ip` mapping**: The `--ip N` flag reads `SPLUNK_HOST_N` from `.env` to resolve the actual Splunk host IP; experiment dir becomes `host_{resolved_ip}_experiments/`
- **Conda env**: `py310_modelenv` (Python 3.10), activate before running

## Tech Stack

- **RL**: Stable-Baselines3 + sb3-contrib (SAC default algorithm)
- **Environment**: Gymnasium (Farama)
- **DL**: PyTorch
- **SIEM**: Splunk via splunklib
- **Cluster**: SLURM (RTX 3090, 60GB RAM, 8 CPUs)
- **Monitoring**: psutil, Elasticsearch, TensorBoard
