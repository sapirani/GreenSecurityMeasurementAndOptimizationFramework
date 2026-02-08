# SplunkEnergyAttack

A Deep Reinforcement Learning (DRL) research framework for studying the energy and resource impact of adversarial log injection on Splunk security detection systems.

## Overview

This project trains RL agents to strategically inject fake Windows Event Logs into Splunk, then measures the impact on:

- **CPU / energy consumption** of security detection rules
- **Alert accuracy** (false positive/negative rates)
- **Log distribution integrity** (deviation from baseline patterns)

The framework enables researchers to understand trade-offs between security monitoring coverage and resource consumption, evaluate detection rule robustness against log injection attacks, and optimize security operations for cost-effectiveness.

## Architecture

The system is built as a [Gymnasium](https://gymnasium.farama.org/) reinforcement learning environment with a wrapper-based composition pattern:

```
┌─────────────────────────────────────────────┐
│              ExperimentManager               │
│  (Training orchestration, model selection)   │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ TimeManager  │  │   Log Generator      │  │
│  │ (Windows,    │  │   (Template-based    │  │
│  │  Episodes)   │  │    event creation)   │  │
│  └──────┬───────┘  └──────────┬───────────┘  │
│         │                     │              │
│  ┌──────▼─────────────────────▼───────────┐  │
│  │           SplunkEnv (Base)             │  │
│  │  ┌─────────────────────────────────┐   │  │
│  │  │  Action Wrapper                 │   │  │
│  │  │  (Quota, distribution, trigger, │   │  │
│  │  │   diversity control)            │   │  │
│  │  ├─────────────────────────────────┤   │  │
│  │  │  Reward Wrapper                 │   │  │
│  │  │  (Energy + Alert + Distribution │   │  │
│  │  │   multi-objective reward)       │   │  │
│  │  ├─────────────────────────────────┤   │  │
│  │  │  State Wrapper                  │   │  │
│  │  │  (Real/fake distributions,      │   │  │
│  │  │   KL divergence, step counter)  │   │  │
│  │  └─────────────────────────────────┘   │  │
│  └────────────────┬───────────────────────┘  │
│                   │                          │
│  ┌────────────────▼───────────────────────┐  │
│  │           Splunk Tools                 │  │
│  │  (REST API, saved searches, metrics)   │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Project Structure

```
SplunkEnergyAttack/
└── SplunkResearch/
    ├── resources/
    │   ├── logtypes.py                 # Log type definitions (38 types)
    │   ├── section_logtypes.py         # Rule-to-event-code mappings (27 rules)
    │   ├── log_generator_resources.py  # Log templates and variations
    │   └── state_span.py              # State space definitions
    └── src/
        ├── custom_splunk/             # Gymnasium environment package
        │   ├── custom_splunk/
        │   │   └── envs/
        │   │       └── custom_splunk_env.py  # Base RL environment
        │   └── setup.py
        ├── wrappers/
        │   ├── action.py              # Action wrappers (Action8, Action11, etc.)
        │   ├── reward.py              # Multi-component reward computation
        │   └── state.py               # State normalization and observation
        ├── experiment_manager_new.py   # Experiment orchestration
        ├── splunk_tools.py            # Splunk API integration
        ├── log_generator.py           # Fake log generation and injection
        ├── time_manager.py            # Episode and time window management
        ├── callbacks.py               # Training callbacks and metrics logging
        ├── energy_profile_final.py    # Energy profiling experiments
        ├── policy.py                  # Custom RL policies
        ├── env_utils.py               # Environment utilities
        ├── cpu_prediction_nb.ipynb    # CPU prediction analysis
        ├── result_analysis.ipynb      # Results visualization
        └── splunk_tools_notebook.ipynb # Splunk tools demo
```

## Installation

### Prerequisites

- Python 3.8+
- A running Splunk instance with:
  - Security detection rules (saved searches) configured
  - HTTP Event Collector (HEC) enabled
  - Windows Event Log data indexed

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<owner>/SplunkEnergyAttack.git
   cd SplunkEnergyAttack
   ```

2. Install the custom Gymnasium environment:
   ```bash
   cd SplunkResearch/src/custom_splunk
   pip install -e .
   ```

3. Install dependencies:
   ```bash
   pip install stable-baselines3 sb3-contrib gymnasium torch splunklib pandas numpy joblib psutil python-dotenv requests elasticsearch
   ```

4. Create an environment file at `SplunkResearch/.env`:
   ```env
   SPLUNK_HOST=<your_splunk_host>
   SPLUNK_PORT=<splunk_management_port>
   SPLUNK_USERNAME=<username>
   SPLUNK_PASSWORD=<password>
   INDEX_NAME=<target_index>
   HEC_TOKEN1=<hec_token>
   HEC_TOKEN2=<hec_token>
   EMAIL=<notification_email>
   EMAIL_PASSWORD=<email_password>
   ```

## Usage

### Running an Experiment

Configure and launch a training experiment via `experiment_manager_new.py`:

```python
from experiment_manager_new import ExperimentManager, ExperimentConfig
from custom_splunk.envs.custom_splunk_env import SplunkConfig

# Define environment configuration
env_config = SplunkConfig(
    rule_frequency=60,           # Rule check frequency (minutes)
    search_window=2880,          # Search window size (minutes)
    logs_per_minute=150,         # Baseline log ingestion rate
    additional_percentage=1,     # Extra log injection fraction
    action_duration=7200,        # Step duration (seconds)
    num_of_measurements=1,
    baseline_num_of_measurements=2,
    env_id="splunk_train-v32",
    end_time="12/31/2024:23:59:59"
)

# Define experiment configuration
experiment_config = ExperimentConfig(
    env_config=env_config,
    model_type="recurrent_ppo",  # RL algorithm
    policy_type="MlpLstmPolicy",
    learning_rate=0.0001,
    num_episodes=600000,
    n_steps=256,
    ent_coef=0.1,
)

# Run experiment
manager = ExperimentManager(base_dir="SplunkResearch/experiments")
results = manager.run_experiment(experiment_config)
```

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

### Configuration Parameters

**Environment (`SplunkConfig`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rule_frequency` | `60` | How often Splunk rules execute (minutes) |
| `search_window` | `2880` | Time range for detection searches (minutes) |
| `action_duration` | `7200` | Step duration in the RL loop (seconds) |
| `logs_per_minute` | `150` | Baseline log ingestion rate |
| `additional_percentage` | `1` | Fraction of extra logs to inject |

**Reward Weights (`ExperimentConfig`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_energy` | `0.5` | Weight for energy/CPU reward |
| `beta_alert` | `0.3` | Weight for alert accuracy reward |
| `gamma_dist` | `0.2` | Weight for distribution fidelity reward |

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
