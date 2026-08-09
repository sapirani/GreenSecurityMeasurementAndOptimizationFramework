# Onboarding Guide — Green Security Measurement & Optimization Framework

**Audience**: New collaborator joining the project  
**Maintained by**: Shimon Shouei

---

## What This Project Does

This framework trains Deep Reinforcement Learning (DRL) agents to inject fake Windows Event Logs into Splunk in order to maximize the CPU cost triggered by Splunk's detection rules, while remaining stealthy (not generating too many alerts, not deviating from the normal log distribution). The goal is to study the energy cost of security monitoring and how adversarial inputs can skew it.

There are two top-level components:

| Component | Location | Purpose |
|---|---|---|
| **DRL / Splunk research** | `SplunkResearch/` | RL training, evaluation, Splunk interaction |
| **Resource measurement** | `scanner.py`, `utils/`, `application_logging/` | CPU/memory/energy profiling of arbitrary processes |

This guide focuses on the DRL component.

---

## Environment Setup

### 1. Conda environment

```bash
conda activate py310_modelenv   # Python 3.10
```

### 2. Install the custom Gym environment

```bash
cd SplunkResearch/src/custom_splunk
pip install -e .
```

### 3. Secrets / credentials

Copy the example and fill in your Splunk credentials:

```bash
cp SplunkResearch/config/secrets.yaml.example SplunkResearch/config/secrets.yaml
# Edit secrets.yaml with real Splunk host, port, username, password, HEC token
```

Runtime environment variables (Splunk host IPs, HEC tokens) go in `SplunkResearch/src/.env`.  
The `--ip N` flag maps to the env var `SPLUNK_HOST_N` defined there.

### 4. Always run from the project root

```bash
cd /home/shouei/GreenSecurityMeasurementAndOptimizationFramework
# Then use the module path:
python -m SplunkResearch.src.run_experiment ...
```

`utils/` and `application_logging/` are root-level packages; running from a subdirectory breaks imports.

---

## Quick-Start: Train → Evaluate

### Train a new model

```bash
python -m SplunkResearch.src.run_experiment \
    --mode train \
    --model-type sac \
    --num-episodes 50000 \
    --ip 1
```

- `--ip 1` selects the Splunk host defined as `SPLUNK_HOST_1` in `.env`
- Default algorithm is SAC (Soft Actor-Critic); alternatives: `ppo`, `a2c`, `dqn`, `td3`
- The experiment directory is created at `SplunkResearch/host_{ip}_experiments/runs/{experiment_name}/`

### Evaluate a trained model

```bash
python -m SplunkResearch.src.run_experiment \
    --mode eval_post_training \
    --model-name "train_20260205101454_600000_steps" \
    --num-episodes 40 \
    --ip 1
```

`--model-name` is the folder name inside `runs/` (no `.zip`). The manager first looks for `runs/{name}/models/final.zip`, then falls back to a legacy flat path.

### Retrain from a checkpoint

```bash
python -m SplunkResearch.src.run_experiment \
    --mode retrain \
    --model-name "train_20260205101454_600000_steps" \
    --ip 1
```

### Submit to SLURM cluster

```bash
sbatch gpu_job_train.sh
```

---

## Key CLI Arguments

| Argument | Description |
|---|---|
| `--mode` | `train`, `eval_post_training`, `retrain` |
| `--model-type` | `sac` (default), `ppo`, `a2c`, `dqn`, `td3`, `recurrent_ppo` |
| `--model-name` | Experiment folder name to load (required for eval/retrain) |
| `--num-episodes` | Number of training/eval episodes |
| `--ip` | Splunk host index (maps to `SPLUNK_HOST_N` in `.env`) |
| `--alpha-energy` | Energy reward weight (overrides config) |
| `--beta-alert` | Alert reward weight (overrides config) |
| `--gamma-dist` | Distribution reward weight (overrides config) |
| `--action-type` | Action space type (e.g. `Action8`, `SoftmaxDistribution`) |
| `--hosts-num` | % of hosts to use (0–100) |
| `--random-agent` | Use random baseline instead of trained model |
| `--test-experiment` | Dry run — disables actual log injection |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

All flags override `config/default.yaml` values. Unset flags use defaults from the config file.

---

## Configuration System

Three-layer hierarchy (each layer overrides the previous):

```
config/default.yaml       ← all defaults (committed)
config/secrets.yaml       ← credentials (gitignored)
CLI arguments             ← runtime overrides
```

Access config in code:

```python
from config import config
lr = config.get('training.learning_rate', 3e-4)
```

Key config sections:
- `reward.*` — reward weights, methods, normalizers, constraint thresholds
- `action.*` — action space parameters (temperature, sharpness)
- `training.*` — learning rate, episodes, SAC-specific params
- `environment.*` — action/state type, mock mode, hosts percentage
- `callbacks.eval.*` — whether to run eval during training, frequency

---

## Experiment Output Structure

```
SplunkResearch/host_{ip}_experiments/
├── baseline/                      # Shared CPU baseline measurements
├── experiments.csv                # Index: name, status, git hash, timestamps
└── runs/{experiment_name}/
    ├── config.json                # Full merged config + git hash
    ├── experiment.log             # Rotating log (50 MB max)
    ├── models/
    │   ├── final.zip              # Final trained model
    │   ├── best_model.zip         # Best checkpoint (by eval reward)
    │   ├── replay_buffer.pkl      # SAC/TD3 replay buffer
    │   └── checkpoints/           # Auto-pruned, last 3 kept
    ├── tensorboard/               # TensorBoard event files
    └── results/                   # Per-episode CSVs, plots
```

Experiment status lifecycle: `running` → `completed` | `failed` | `interrupted` | `crashed`  
Experiments running > 48 h are auto-marked `crashed` on next startup.

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir SplunkResearch/host_X_experiments/runs/
```

Or use the built-in skill (if using Claude Code):

```
/tensorboard <SLURM job ID>
```

---

## Code Architecture

### Entry point flow

```
run_experiment.py  →  ExperimentManager  →  create_environment()  →  model.learn()
```

`run_experiment.py` parses CLI args and builds a config dict, then hands off to `ExperimentManager` (`experiment_manager_new.py`) which owns the full experiment lifecycle.

### Environment wrapper stack (inner → outer)

```
SplunkEnv  (base Gymnasium env — log injection, Splunk queries)
 └→ Action wrapper           (interprets agent actions → log counts)
   └→ DistributionRewardWrapper
     └→ BaseRuleExecutionWrapperWithPrediction  (runs Splunk saved searches or mock)
       └→ EnergyRewardWrapper
         └→ AlertRewardWrapper
           └→ TimeWrapper
             └→ StateWrapper                   ← outermost, what the agent sees
```

### Strategy registries (extensible)

| Concern | ABC | Registry |
|---|---|---|
| Action interpretation | `ActionInterpreter` | `ACTION_INTERPRETER_REGISTRY` |
| State construction | `StateInterpreter` | `STATE_INTERPRETER_REGISTRY` |
| Alert reward | `AlertSignal` + `AlertNormalizer` | `ALERT_REWARD_REGISTRY` |

To add a new action type or reward method, implement the ABC and register it — no changes to the wrappers themselves.

### Key source files

| File | Role |
|---|---|
| `src/run_experiment.py` | CLI entry point |
| `src/experiment_manager_new.py` | Experiment lifecycle, model training/loading |
| `src/callbacks.py` | SB3 callbacks (TensorBoard, eval, licence check) |
| `src/splunk_tools.py` | Splunk REST API (dispatch searches, profile CPU, manage index) |
| `src/log_generator.py` | Template-based fake Windows Event Log generation |
| `src/time_manager.py` | Episode time window management |
| `src/wrappers/` | All Gymnasium wrappers (action, state, reward, shared state) |
| `src/config.py` | YAML config singleton |
| `config/default.yaml` | All default parameter values |

---

## Mock Mode

When `environment.is_mock: true` (or `--test-experiment`), the expensive Splunk saved-search execution is replaced by a pre-trained CPU prediction model (`.joblib` files in `src/`). Log injection, state observation, and reward computation still run normally. Use mock mode for fast iteration without a live Splunk instance.

---

## Common Gotchas

1. **Run from project root** — `utils/` and `application_logging/` must be importable at top level.
2. **Conda env** — always activate `py310_modelenv` before running anything.
3. **Secrets file** — `config/secrets.yaml` is gitignored. You must create it from the `.example`.
4. **`.env` file** — `SplunkResearch/src/.env` holds `SPLUNK_HOST_N` entries. Required for `--ip` to work.
5. **SAC replay buffer** — loading a SAC model for retrain also loads `replay_buffer.pkl`; make sure it exists alongside the `.zip`.
6. **TensorBoard subdirs** — SB3 appends `_0`, `_1`, … to `tb_log_name`; the code forces `tb_log_name="train"` to keep paths predictable.

---

## Useful One-Liners

```bash
# Check experiment status
grep "status" SplunkResearch/host_1_experiments/experiments.csv

# Watch live training log
tail -f SplunkResearch/host_1_experiments/runs/<experiment_name>/experiment.log

# List saved models
ls SplunkResearch/host_1_experiments/runs/<experiment_name>/models/

# Run a quick sanity-check (no injection, mock CPU model)
python -m SplunkResearch.src.run_experiment \
    --mode train --model-type sac --num-episodes 10 \
    --ip 1 --test-experiment --log-level DEBUG
```

---

## Getting Help

- Project CLAUDE.md at the repo root has a full architecture reference.
- Raise questions in the team chat or open a GitHub issue.
