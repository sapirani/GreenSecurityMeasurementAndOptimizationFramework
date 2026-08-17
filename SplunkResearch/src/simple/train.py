"""Entry point: build the env, train SAC or PPO, evaluate, save.

Run from the project root:

    python -m SplunkResearch.src.simple.train --algorithm sac --num-episodes 200
    python -m SplunkResearch.src.simple.train --mode eval --model-name train_20260814_...

This replaces ExperimentManager (1267 lines) + callbacks.py (371 lines). A run is
just a directory: config.yaml + tb/ + models/ + result.json. No experiments.csv,
no status lifecycle, no git hashing, no email, no signal handlers.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys

# Allow both `python -m SplunkResearch.src.simple.train` and direct src-on-path use.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from simple.config import SimpleConfig
from simple.env import SimpleEnv

logger = logging.getLogger("simple.train")

_COMPONENT_KEYS = ["reward", "energy_raw", "energy_term", "alert_metric",
                   "alert_penalty", "kl", "kl_penalty", "baseline_cpu",
                   "current_cpu", "injected_logs", "n_triggered_rules"]


class ComponentLoggerCallback(BaseCallback):
    """Record the reward breakdown to SB3's TensorBoard on episode end.

    One writer (SB3's own), a fixed whitelist of keys — no per-rule writer fleet.
    """

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "reward" in info:  # present only on the episode-end step
                for k in _COMPONENT_KEYS:
                    if k in info:
                        self.logger.record(f"reward/{k}", float(info[k]))
        return True


def build_model(cfg: SimpleConfig, env, tb_dir: str):
    common = dict(policy="MlpPolicy", env=env, learning_rate=cfg.learning_rate,
                  gamma=cfg.gamma_rl, verbose=1, seed=cfg.seed, tensorboard_log=tb_dir)
    if cfg.algorithm == "sac":
        return SAC(
            **common,
            buffer_size=cfg.buffer_size, batch_size=cfg.batch_size,
            ent_coef=cfg.ent_coef, use_sde=cfg.use_sde,
            train_freq=(cfg.train_freq_episodes, "episode"),
            policy_kwargs=dict(net_arch=cfg.net_arch, log_std_init=cfg.log_std_init),
        )
    if cfg.algorithm == "ppo":
        return PPO(
            **common,
            n_steps=cfg.ppo_n_steps, batch_size=cfg.ppo_batch_size,
            use_sde=cfg.use_sde, policy_kwargs=dict(net_arch=cfg.net_arch),
        )
    raise ValueError(f"Unknown algorithm: {cfg.algorithm}")


def evaluate(model, cfg: SimpleConfig, n_episodes: int) -> dict:
    env = SimpleEnv(cfg, eval=True)
    rewards, comps = [], {k: [] for k in _COMPONENT_KEYS}
    try:
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done, ep_r, last = False, 0.0, {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, _, info = env.step(action)
                ep_r += r
                last = info
            rewards.append(ep_r)
            for k in _COMPONENT_KEYS:
                if k in last:
                    comps[k].append(float(last[k]))
    finally:
        env.close()
    out = {"n_episodes": n_episodes,
           "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}
    out.update({f"mean_{k}": float(np.mean(v)) for k, v in comps.items() if v})
    return out


def _setup_logging(run_dir: str):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(os.path.join(run_dir, "experiment.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def main(argv=None):
    cfg = SimpleConfig.load(argv)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if cfg.mode == "eval":
        model_name = getattr(cfg, "model_name", None)
        if not model_name:
            raise SystemExit("--mode eval requires --model-name <run_dir>")
        run_dir = os.path.join(cfg.experiments_root, model_name)
        _setup_logging(run_dir)
        model_path = os.path.join(run_dir, "models", "final.zip")
        cls = SAC if cfg.algorithm == "sac" else PPO
        model = cls.load(model_path)
        results = evaluate(model, cfg, cfg.n_eval_episodes)
        with open(os.path.join(run_dir, f"eval_{ts}.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Eval results: %s", results)
        return results

    # --- train ---
    run_dir = os.path.join(cfg.experiments_root, f"train_{ts}")
    os.makedirs(os.path.join(run_dir, "models", "checkpoints"), exist_ok=True)
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    _setup_logging(run_dir)

    with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump(cfg.to_public_dict(), fh, sort_keys=True)
    logger.info("Run dir: %s", run_dir)

    env = Monitor(SimpleEnv(cfg, eval=False))
    model = build_model(cfg, env, tb_dir)

    total_timesteps = cfg.total_steps * cfg.num_episodes
    callbacks = [
        ComponentLoggerCallback(),
        CheckpointCallback(save_freq=cfg.checkpoint_freq_steps,
                           save_path=os.path.join(run_dir, "models", "checkpoints"),
                           name_prefix="ckpt"),
    ]
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    tb_log_name="train", log_interval=1)
        model.save(os.path.join(run_dir, "models", "final.zip"))
        if cfg.algorithm == "sac":
            model.save_replay_buffer(os.path.join(run_dir, "models", "replay_buffer.pkl"))
        results = evaluate(model, cfg, cfg.n_eval_episodes)
        with open(os.path.join(run_dir, "result.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Training complete. Eval: %s", results)
        return results
    finally:
        env.close()


if __name__ == "__main__":
    main()
