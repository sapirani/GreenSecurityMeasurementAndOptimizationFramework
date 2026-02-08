# SplunkEnergyAttack

A Deep Reinforcement Learning (DRL) research framework for studying the energy and resource impact of adversarial log injection on Splunk security detection systems.

## Overview

This project trains RL agents to strategically inject fake Windows Event Logs into Splunk, then measures the impact on:

- **CPU / energy consumption** of security detection rules
- **Alert accuracy** (false positive/negative rates)
- **Log distribution integrity** (deviation from baseline patterns)

The framework enables researchers to understand trade-offs between security monitoring coverage and resource consumption, evaluate detection rule robustness against log injection attacks, and optimize security operations for cost-effectiveness.

## Architecture

The system is built as a [Gymnasium](https://gymnasium.farama.org/) reinforcement learning environment with a wrapper-based composition pattern and centralized configuration management:

```
┌──────────────────────────────────────────────────┐
│              CLI (run_experiment.py)              │
│         Command-line interface with argparse      │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │      Config System (config.py)          │    │
│  │  default.yaml + secrets.yaml + CLI args │    │
│  └──────────────────┬──────────────────────┘    │
│                     │                            │
│  ┌──────────────────▼──────────────────────┐    │
│  │         ExperimentManager               │    │
│  │  (Training orchestration, model mgmt)   │    │
│  ├─────────────────────────────────────────┤    │
│  │                                         │    │
│  │  ┌─────────────┐  ┌──────────────────┐  │    │
│  │  │TimeManager  │  │  Log Generator   │  │    │
│  │  │(Windows,    │  │  (Template-based │  │    │
│  │  │ Episodes)   │  │  event creation) │  │    │
│  │  └──────┬──────┘  └──────────┬───────┘  │    │
│  │         │                    │          │    │
│  │  ┌──────▼────────────────────▼────────┐  │    │
│  │  │       SplunkEnv (Base)            │  │    │
│  │  │  ┌────────────────────────────┐   │  │    │
│  │  │  │  Action Wrapper            │   │  │    │
│  │  │  │  (Quota, distribution,     │   │  │    │
│  │  │  │   trigger, diversity)      │   │  │    │
│  │  │  ├────────────────────────────┤   │  │    │
│  │  │  │  Reward Wrapper            │   │  │    │
│  │  │  │  (Energy + Alert +         │   │  │    │
│  │  │  │   Distribution rewards)    │   │  │    │
│  │  │  ├────────────────────────────┤   │  │    │
│  │  │  │  State Wrapper             │   │  │    │
│  │  │  │  (Distributions, KL div,   │   │  │    │
│  │  │  │   step counter)            │   │  │    │
│  │  │  └────────────────────────────┘   │  │    │
│  │  └────────────────┬──────────────────┘  │    │
│  │                   │                     │    │
│  │  ┌────────────────▼──────────────────┐  │    │
│  │  │       Splunk Tools                │  │    │
│  │  │  (REST API, saved searches,       │  │    │
│  │  │   resource profiling)             │  │    │
│  │  └───────────────────────────────────┘  │    │
│  └─────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

### Key Improvements (2026 Refactoring)

- **Centralized Configuration**: YAML-based config system with CLI overrides
- **Clean CLI Interface**: Argparse-based command-line tool with named arguments
- **Modular Design**: Clear separation between environment, manager, and execution
- **Better Documentation**: Comprehensive docstrings and configuration guides
- **SLURM Integration**: Production-ready job scripts for cluster execution

## Project Structure

```
GreenSecurityMeasurementAndOptimizationFramework/
├── gpu_job_train.sh                   # SLURM training job script
├── gpu_job_evaluation_80_159.sh       # SLURM evaluation job (host 159)
├── gpu_job_evaluation_81_184.sh       # SLURM evaluation job (host 184)
└── SplunkResearch/
    ├── config/
    │   ├── README.md                  # Configuration documentation
    │   ├── default.yaml               # Default configuration values
    │   └── secrets.yaml.example       # Example secrets template
    ├── resources/
    │   ├── logtypes.py                # Log type definitions (38 types)
    │   ├── section_logtypes.py        # Rule-to-event-code mappings (27 rules)
    │   ├── log_generator_resources.py # Log templates and variations
    │   └── state_span.py              # State space definitions
    └── src/
        ├── .env                       # Environment variables (gitignored)
        ├── config.py                  # Configuration management system
        ├── run_experiment.py          # CLI entry point for experiments
        ├── custom_splunk/             # Gymnasium environment package
        │   ├── custom_splunk/
        │   │   └── envs/
        │   │       └── custom_splunk_env.py  # Base RL environment
        │   └── setup.py
        ├── wrappers/
        │   ├── action.py              # Action wrappers (Action8, Action12, etc.)
        │   ├── reward.py              # Multi-component reward computation
        │   └── state.py               # State normalization and observation
        ├── experiment_manager_new.py  # Experiment orchestration
        ├── splunk_tools.py            # Splunk API integration
        ├── log_generator.py           # Fake log generation and injection
        ├── time_manager.py            # Episode and time window management
        ├── callbacks.py               # Training callbacks and metrics logging
        ├── energy_profile_final.py    # Energy profiling experiments
        ├── env_utils.py               # Environment utilities
        ├── cpu_prediction_nb.ipynb    # CPU prediction analysis
        ├── result_analysis.ipynb      # Results visualization
        └── splunk_tools_notebook.ipynb # Splunk tools demo
```

## Installation

### Prerequisites

- Python 3.10+
- A running Splunk instance with:
  - Security detection rules (saved searches) configured
  - HTTP Event Collector (HEC) enabled
  - Windows Event Log data indexed
- SLURM cluster (for distributed training jobs) or local execution

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sapirani/GreenSecurityMeasurementAndOptimizationFramework.git
   cd GreenSecurityMeasurementAndOptimizationFramework
   ```

2. Install the custom Gymnasium environment:
   ```bash
   cd SplunkResearch/src/custom_splunk
   pip install -e .
   ```

3. Install dependencies:
   ```bash
   pip install stable-baselines3 sb3-contrib gymnasium torch splunklib pandas numpy joblib psutil python-dotenv requests elasticsearch pyyaml
   ```

4. Configure the framework:

   a. **Create secrets file** at `SplunkResearch/src/.env`:
   ```env
   # Splunk hosts (IP identifier mapping)
   SPLUNK_HOST_1=132.72.81.184
   SPLUNK_HOST_2=132.72.80.159
   SPLUNK_HOST_3=132.72.81.150

   # Splunk credentials
   SPLUNK_PORT=8089
   SPLUNK_USERNAME=<username>
   SPLUNK_PASSWORD=<password>
   INDEX_NAME=<target_index>

   # HTTP Event Collector tokens
   HEC_TOKEN1=<hec_token>
   HEC_TOKEN2=<hec_token>

   # Email notifications
   EMAIL=<notification_email>
   EMAIL_PASSWORD=<email_password>
   ```

   b. **Configure experiment settings** in `SplunkResearch/config/default.yaml`:
   - Adjust model hyperparameters (learning rate, batch size, etc.)
   - Set reward weights (alpha, beta, gamma)
   - Configure environment parameters (log rates, time windows)
   - Customize training/evaluation settings

   c. **Optional: Add personal secrets** in `SplunkResearch/config/secrets.yaml`:
   ```yaml
   # Override any default.yaml values here
   # This file is gitignored and won't be committed
   email:
     address: "your-email@example.com"
     password: "your-app-password"
   ```

## Usage

### Configuration System

The framework uses a hierarchical configuration system with three layers:

1. **`config/default.yaml`** - Base configuration with all default values
2. **`config/secrets.yaml`** (optional) - Override sensitive values (gitignored)
3. **CLI arguments** - Override any config value at runtime

Configuration values are accessed through the `config.py` module which merges all layers automatically.

### Running Experiments

#### Training Mode

**Basic Training (uses defaults from config files):**
```bash
python -m SplunkResearch.src.run_experiment \
    --mode train \
    --num-episodes 50000 \
    --ip 1
```

**Training with Custom Parameters:**
```bash
python -m SplunkResearch.src.run_experiment \
    --mode train \
    --model-type sac \
    --alpha-energy 0.334 \
    --beta-alert 0.333 \
    --gamma-dist 0.333 \
    --hosts-num 100 \
    --learning-rate 1e-4 \
    --num-episodes 50000 \
    --ip 1 \
    --action-type Action8
```

#### Evaluation Mode

**Evaluate Trained Model:**
```bash
python -m SplunkResearch.src.run_experiment \
    --model-name "train_20260205101454_600000_steps" \
    --mode eval_post_training \
    --alpha-energy 0.334 \
    --beta-alert 0.333 \
    --gamma-dist 0.333 \
    --hosts-num 10 \
    --additional-percentage 0.1 \
    --num-episodes 40 \
    --ip 1
```

#### SLURM Cluster Usage

For distributed training on SLURM clusters, use the provided job scripts:

**Training:**
```bash
sbatch gpu_job_train.sh
```

**Evaluation:**
```bash
sbatch gpu_job_evaluation_80_159.sh
sbatch gpu_job_evaluation_81_184.sh
```

These scripts handle:
- Splunk index reset before each experiment
- Parameter sweeps across array jobs
- Environment setup and cleanup
- Results aggregation

### Supported RL Algorithms

| Algorithm | `model_type` | Description |
|-----------|-------------|-------------|
| PPO | `"ppo"` | Proximal Policy Optimization |
| A2C | `"a2c"` | Advantage Actor-Critic |
| DQN | `"dqn"` | Deep Q-Network |
| SAC | `"sac"` | Soft Actor-Critic |
| DDPG | `"ddpg"` | Deep Deterministic Policy Gradient |
| TD3 | `"td3"` | Twin Delayed DDPG |
| Recurrent PPO | `"recurrent_ppo"` | LSTM-based PPO for temporal patterns |

### CLI Arguments

All configuration values from `default.yaml` can be overridden via command-line arguments:

**Experiment Configuration:**
| Argument | Description | Example |
|----------|-------------|---------|
| `--mode` | Experiment mode | `train`, `eval_post_training`, `retrain` |
| `--model-name` | Model to load (for eval/retrain) | `train_20260205101454_600000_steps` |
| `--model-type` | RL algorithm | `sac`, `ppo`, `a2c`, `dqn`, `td3`, `recurrent_ppo` |
| `--policy-type` | Policy network | `MlpPolicy`, `MlpLstmPolicy` |

**Reward Weights:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--alpha-energy` | `0.5` | Weight for energy/CPU reward |
| `--beta-alert` | `0.3` | Weight for alert accuracy reward |
| `--gamma-dist` | `0.2` | Weight for distribution fidelity reward |
| `--alert-epsilon` | `0.1` | Alert reward epsilon |
| `--normalizer-factor` | `30.0` | Reward normalizer factor |

**Training Parameters:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--learning-rate` | `3e-4` | Learning rate for optimizer |
| `--num-episodes` | `100` | Number of training episodes |

**Environment Configuration:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--hosts-num` | `100` | Percentage of hosts to use (0-100) |
| `--additional-percentage` | `1.0` | Fraction of extra logs to inject |
| `--action-type` | `Action8` | Action space type (`Action8`, `Action12`) |
| `--ip` | `1` | Splunk host IP identifier (1, 2, 3) |

**Reward Methods:**
| Argument | Options |
|----------|---------|
| `--alert-reward-method` | `AlertRewardWrapper`, `AlertRewardWrapper2` |
| `--distribution-reward-method` | `DistributionRewardWrapper`, `DistributionRewardWrapper2` |

**Flags:**
| Flag | Description |
|------|-------------|
| `--random-agent` | Use random agent instead of trained model |
| `--test-experiment` | Run in test mode (disables injection) |

### Configuration Files

**`config/default.yaml`** contains all default values organized by category:
- `paths.*` - File paths and directories
- `splunk.*` - Splunk connection and indexing settings
- `environment.*` - Environment parameters
- `reward.*` - Reward function weights and methods
- `training.*` - Model training hyperparameters
- `model.*` - Neural network architecture
- `evaluation.*` - Evaluation settings
- `callbacks.*` - Training callbacks configuration
- `logging.*` - Logging levels and formats
- `email.*` - Email notification settings

See [config/README.md](config/README.md) for detailed documentation.

### Energy Profiling

Run standalone energy profiling (without RL training) using `energy_profile_final.py`:

```python
python SplunkResearch/src/energy_profile_final.py
```

This iterates over rules, diversity levels, and log quantities to measure CPU and I/O metrics for each detection rule.

## Detection Rules

The framework targets Windows Event Log-based Splunk detection rules. A subset of the 27+ supported rules:

| Detection Rule | Event Source | Event Code |
|---|---|---|
| Detect New Local Admin Account | `wineventlog:security` | 4732 |
| Kerberoasting SPN Request with RC4 Encryption | `wineventlog:security` | 4769 |
| Windows Rapid Authentication on Multiple Hosts | `wineventlog:security` | 4624 |
| Network Share Discovery Via Dir Command | `wineventlog:security` | 5140 |
| Non-Chrome Process Accessing Chrome Default Dir | `wineventlog:security` | 4663 |
| AD Replication Request from Unsanctioned Location | `wineventlog:security` | 4662 |
| Known Services Killed by Ransomware | `wineventlog:system` | 7036 |
| Windows Event For Service Disabled | `wineventlog:system` | 7040 |
| Clop Ransomware Known Service Name | `wineventlog:system` | 7045 |

Full mapping available in [section_logtypes.py](SplunkResearch/resources/section_logtypes.py).

## How It Works

Each training step follows this cycle:

1. **Observe** — The agent receives the current state: real and fake log distributions, KL divergence, and step counter.
2. **Act** — The agent outputs an action vector specifying injection quota, per-rule log distribution, triggering levels, and diversity.
3. **Inject** — The log generator creates realistic Windows Event Logs from templates and writes them to Splunk monitor files.
4. **Execute** — Splunk saved searches (detection rules) run over the current time window.
5. **Measure** — CPU time, I/O operations, result counts, and alert counts are recorded.
6. **Reward** — A multi-component reward is computed from energy cost, alert deviations, and distribution fidelity.
7. **Cleanup** — Fake logs are deleted from Splunk at episode end.

## Analysis

Jupyter notebooks are provided for post-experiment analysis:

- [cpu_prediction_nb.ipynb](SplunkResearch/src/cpu_prediction_nb.ipynb) — CPU consumption prediction models
- [result_analysis.ipynb](SplunkResearch/src/result_analysis.ipynb) — Training results visualization
- [splunk_tools_notebook.ipynb](SplunkResearch/src/splunk_tools_notebook.ipynb) — Splunk API interaction demos

Training metrics are logged to **TensorBoard**:

```bash
tensorboard --logdir SplunkResearch/experiments/tensorboard
```

## Tech Stack

- **RL Framework:** [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), sb3-contrib
- **Environment:** [Gymnasium](https://gymnasium.farama.org/)
- **Deep Learning:** PyTorch
- **SIEM:** Splunk (via splunklib SDK)
- **Monitoring:** psutil, Elasticsearch
- **Visualization:** TensorBoard, Jupyter notebooks
