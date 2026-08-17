# `simple/` — a lean, debuggable rewrite of the DRL log-injection framework

A from-scratch reimplementation of the Splunk log-injection RL research, built to
be small and traceable. It reproduces the mechanics of the old "barely good" model
(`train_20260129234645`: SAC, mock energy, 9 rules) in **~800 lines** (mock core)
instead of the ~4,000 the legacy path spreads across a base env + 7 gym wrappers +
`ExperimentManager` + `callbacks.py` + a 157-key config.

The legacy code under `src/wrappers/`, `src/experiment_manager_new.py`, etc. is left
untouched for reference. Nothing here imports it except the live path, which reuses
the proven `SplunkTools` / `LogGenerator`.

## Why it's easier to debug

- **One flat `gym.Env`** (`env.py`), no wrapper stack. All episode state is plain
  instance attributes — what you see in a debugger is the whole state. No
  `EpisodeSharedState`, no property aliases, no stack-walking.
- **One reward formula** (`reward.py`), computed once, fully logged. The scalar
  returned by `step()` is *exactly* what lands in TensorBoard — there is no second
  "diagnostic" reward that silently disagrees (the specific trap in the old
  `wrappers/reward.py:839`). A unit test asserts `logged == returned`.
- **No dead machinery**: no adaptive Lagrange multipliers, no τ curriculum, no
  quota term, no `reward_mode` switch, no normalizer registries, no strategy ABCs.
- **Deterministic**: no `random`/`np.random` inside the reward path. Same env +
  same action ⇒ same reward (tested).
- **One TensorBoard writer**, not the old ~69-writer-per-run fleet.

## The loop

Each episode is `total_steps` (=12) steps. Per step the agent outputs a `Box(0,1)^35`:
softmax distribution over 26 log types + a per-rule trigger/diversity for the 9
rule-triggering types (verbatim "Action8" decode, `injection.py`). Plans and real
historical log counts are accumulated; at the last step we measure and reward:

```
energy = max((cpu - cpu_base)/cpu_base, 0)                 # detection CPU cost
alert  = mean_rule max((a - a_base)/max(a_base,1), 0)       # alert loudness
kl     = KL(real_mix || real+injected_mix)                  # distribution stealth
reward = α·log1p(energy) − β·tanh_hinge(alert,τa,sa) − γ·tanh_hinge(kl,τk,sk)
```

The stealth penalties are **tanh-bounded to [0,1]** on purpose — an unbounded
penalty makes the tradeoff unlearnable (a large alert violation swamps energy).

## Mock vs live (`measure.py`)

One switch, `cfg.live`:

- **mock** (default, tested): per-rule XGBoost CPU regressors (`src/cpu_model_*.joblib`)
  + predicted alerts (`baseline + injected diversity`). No Splunk. Uses
  `resources/all_dist_by_host.csv` for the real distribution. Baseline and current
  CPU use the *same* estimator (unlike the legacy mock path, which compared a model
  estimate against a real measurement).
- **live** (`live.py`, **untested — needs a Splunk host**): renders + injects logs
  (`receivers/stream`), waits for indexing, profiles the rules via
  `/services/profiler_api`. Reuses `SplunkTools`/`LogGenerator`. Smoke-test one
  episode before trusting it.

## Run

```bash
# from the project root, in the py310_modelenv conda env
python -m SplunkResearch.src.simple.train --algorithm sac --num-episodes 50000
python -m SplunkResearch.src.simple.train --algorithm ppo --num-episodes 20000
python -m SplunkResearch.src.simple.train --mode eval --model-name train_<ts>

# live (only against a real host with SPLUNK_HOST_{ip} in ../.env):
python -m SplunkResearch.src.simple.train --algorithm sac --num-episodes 5000 --live --ip 1

# tests
python SplunkResearch/src/simple/tests/test_simple.py
```

A run is a directory under `SplunkResearch/simple_experiments/train_<ts>/`:
`config.yaml`, `experiment.log`, `models/{final.zip,replay_buffer.pkl,checkpoints/}`,
`tb/`, `result.json`. The directory *is* the record — no `experiments.csv`.

## Files

| File | Lines | Role |
|---|---|---|
| `rules.py` | ~77 | The 9 rules, their log types, expected alerts; log-type space loader |
| `config.py` | ~134 | One `SimpleConfig` dataclass (defaults = the "barely good" run) + YAML/CLI |
| `injection.py` | ~91 | Action → injection plan (verbatim Action8 decode) |
| `realdist.py` | ~49 | Real log-type distribution per window from the static CSV |
| `measure.py` | ~113 | `MockMeasurer` (regressors) + the `Measurer` interface + `make_measurer` |
| `reward.py` | ~93 | The single reward formula + KL + bounded hinge |
| `env.py` | ~173 | The one flat `gym.Env` |
| `train.py` | ~168 | Build env, train SAC/PPO, evaluate, save — replaces ExperimentManager |
| `live.py` | ~123 | Live Splunk injection + profiling (untested) |

## Known limitation

The infrastructure trains and the reward is well-scaled, but a *good* policy is not
guaranteed by a short run: because the stealth penalties saturate, a policy parked
at high-energy/high-alert sees little gradient toward stealth. The legacy code used
a τ-curriculum to address this. Options if convergence stalls: anneal `tau_alert`/
`tau_kl` from loose to tight, lower `alert_sensitivity`/`kl_sensitivity`, or
raise `beta`/`gamma`. These are the *only* knobs — tune them against the fully
logged reward components, which is the whole point of this rewrite.
