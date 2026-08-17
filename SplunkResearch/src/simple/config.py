"""One dataclass holding everything a run needs. Defaults reproduce the
"barely good" model `train_20260129234645` (SAC, mock, 9 rules).

Load order: dataclass defaults -> optional YAML file -> CLI overrides. No global
singleton, no dot-notation resolver, no secrets layering. Splunk credentials are
read from the environment at use-time (see measure.LiveMeasurer), never stored here.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import os

_SPLUNK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class SimpleConfig:
    # --- experiment ---
    mode: str = "train"                 # train | eval
    algorithm: str = "sac"              # sac | ppo
    seed: int = 42
    live: bool = False                  # False = mock (predict), True = hit Splunk
    ip: int = 1                         # resolves SPLUNK_HOST_{ip} from .env (live only)
    splunk_port: int = 8089             # live only: Splunk management port
    secondary_host: str = "132.72.81.150"  # live only: host tag for injected events (matches legacy default)

    # --- time window (episode = window_size, step = action_duration) ---
    fake_start_datetime: str = "12/01/2024:00:00:00"
    end_time: str = "12/31/2024:23:59:59"
    window_size_min: int = 2880         # 48h episode
    action_duration_s: int = 14400      # 4h step  -> total_steps = 2880*60/14400 = 12
    rule_frequency_min: int = 120       # stride between candidate episode starts

    # --- action / injection ---
    max_logtypes: int = 20
    hosts_percentage: int = 100
    base_log_count: int = 20000         # logs injected per step (fixed volume)
    additional_percentage: float = 1.0
    softmax_temperature: float = 20.0
    diversity_factor: int = 30

    # --- reward (single formula: a*log1p(energy) - b*relu(alert-ta) - g*relu(kl-tk)) ---
    alpha: float = 0.34                 # energy weight
    beta: float = 0.33                  # alert (stealth) weight
    gamma: float = 0.33                 # distribution KL (stealth) weight
    tau_alert: float = 0.25             # allowed relative alert increase before penalty
    tau_kl: float = 0.22                # allowed KL divergence before penalty
    alert_sensitivity: float = 0.25     # how fast the (bounded) alert penalty saturates
    kl_sensitivity: float = 0.30        # how fast the (bounded) KL penalty saturates
    fake_dist_normalizer: float = 475796.0   # scales distribution features for the cpu models
    alert_normalizer: float = 203.0          # scales alert feature for the cpu models
    energy_epsilon: float = 1e-8
    dist_epsilon: float = 1e-8

    # --- RL training ---
    learning_rate: float = 3e-4
    num_episodes: int = 50000
    gamma_rl: float = 0.95
    buffer_size: int = 100000
    batch_size: int = 1024
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    ent_coef: str = "auto"
    log_std_init: float = -3.0
    use_sde: bool = True
    train_freq_episodes: int = 4        # SAC: train every N episodes
    # PPO-specific
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 256
    # eval / checkpoint
    n_eval_episodes: int = 10
    checkpoint_freq_steps: int = 10000

    # --- paths ---
    base_dir: str = _SPLUNK_ROOT
    top_logtypes_csv: str = os.path.join(_SPLUNK_ROOT, "resources", "top_logtypes.csv")
    dist_csv: str = os.path.join(_SPLUNK_ROOT, "resources", "all_dist_by_host.csv")
    cpu_models_dir: str = os.path.join(_SPLUNK_ROOT, "src")
    experiments_root: str = os.path.join(_SPLUNK_ROOT, "simple_experiments")

    # ------------------------------------------------------------------
    @property
    def total_steps(self) -> int:
        assert (self.window_size_min * 60) % self.action_duration_s == 0, \
            "window_size must be divisible by action_duration"
        return self.window_size_min * 60 // self.action_duration_s

    def to_public_dict(self) -> dict:
        """Serializable snapshot for provenance. Contains no secrets by design."""
        return asdict(self)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, argv: Optional[List[str]] = None) -> "SimpleConfig":
        """Build from defaults, then optional --config YAML, then CLI overrides."""
        parser = argparse.ArgumentParser(description="Simple DRL log-injection trainer")
        parser.add_argument("--config", type=str, default=None, help="Optional YAML overrides")
        # Every scalar field is exposable as --field; only add the commonly tuned ones.
        parser.add_argument("--mode", choices=["train", "eval"])
        parser.add_argument("--algorithm", choices=["sac", "ppo"])
        parser.add_argument("--live", action="store_true", default=None)
        parser.add_argument("--ip", type=int)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--num-episodes", type=int, dest="num_episodes")
        parser.add_argument("--learning-rate", type=float, dest="learning_rate")
        parser.add_argument("--alpha", type=float)
        parser.add_argument("--beta", type=float)
        parser.add_argument("--gamma", type=float)
        parser.add_argument("--tau-alert", type=float, dest="tau_alert")
        parser.add_argument("--tau-kl", type=float, dest="tau_kl")
        parser.add_argument("--model-name", type=str, dest="model_name", default=None,
                            help="For --mode eval: run dir name under experiments_root")
        parser.add_argument("--n-eval-episodes", type=int, dest="n_eval_episodes")
        args = parser.parse_args(argv)

        overrides = {}
        if args.config:
            import yaml
            with open(args.config) as fh:
                overrides.update(yaml.safe_load(fh) or {})

        valid = {f.name for f in dataclasses.fields(cls)}
        for k, v in vars(args).items():
            if k in ("config", "model_name"):
                continue
            if v is not None and k in valid:
                overrides[k] = v

        cfg = cls(**{k: v for k, v in overrides.items() if k in valid})
        cfg.model_name = args.model_name  # attached for the eval path
        return cfg
