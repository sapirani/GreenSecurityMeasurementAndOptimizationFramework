# Inbox: drl

Append-only. Only the owning agent writes here. See ../PROTOCOL.md for the entry format.

---

## 2026-08-05 — Scoping: MDP / reward / SAC / eval review plan + first-pass insights

Active config path (verified from `config/default.yaml`): `reward.reward_mode: constrained`
→ `ConstrainedRewardWrapper` is the live reward; `environment.state_type: StateWrapper8`
(AlertAware); `environment.action_type: Action8` (SoftmaxDistribution, fixed volume);
`environment.is_mock: true` (energy reward comes from surrogate CPU models, not Splunk);
SAC default. Horizon = 12 steps/episode (window 2880min / action 240min), terminal-dominated
energy reward, `training.gamma: 0.95`.

### 1. Review plan (prioritized)

- **P-A HIGH / M — Does the "constrained" reward actually constrain?**
  Inspect `wrappers/reward.py:742-849` (ConstrainedRewardWrapper.step), `reward.py:679-693`
  (hinge helpers), config `reward.lambda_*` / `reward.*_sensitivity` (default.yaml:85-99).
  Q: with fixed λ and tanh-bounded penalties, can the agent buy energy by violating
  alert/KL budgets? (see insight I1/I2). This is the single most important area.

- **P-B HIGH / S — Curriculum τ-annealing correctness.**
  `reward.py:628-705` (esp. `:633` `total_training_episodes = config.get('training.num_episodes')`
  and `_get_effective_tau` `:695-705`), config `reward.curriculum_*` (default.yaml:93-98),
  `config.py` (no setter). Q: does the annealing schedule track actual run length? (I3 — it does not).

- **P-C HIGH / M — Action-space conditioning.**
  `wrappers/action_interpreters.py:95-141` (SoftmaxDistribution), config `action.softmax_temperature:20`
  (default.yaml:117), `action.base_log_count:20000`. Q: is the mapping well-scaled, or does
  temperature=20 collapse it to near-argmax with dead dims / bad gradients? (I4).

- **P-D MED / M — Energy objective = exploitation of learned CPU surrogate.**
  `reward.py:640-668` (`_estimate_energy_consumption`), `reward.py:775-778` (energy_term),
  `is_mock` default true. Q: is `alpha*log1p((cpu-base)/base)` reward-hackable by pushing
  `fake_dist`/alerts outside the regressors' training support? Needs @security on model validity.

- **P-E MED / S — Eval methodology & best-model selection.**
  `callbacks.py:281-342` (CustomEvalCallback3), config `callbacks.eval.eval_freq:600000`,
  `evaluation.deterministic:false`, `evaluation.n_eval_episodes:1`. Q: does during-training
  eval ever fire, and is best_model chosen on a stable statistic? (I6).

- **P-F MED / S — Reward-term weighting & per-step vs episode-end scaling.**
  `reward.py:800-805` (only `alpha` used), `:747-756` (per-step KL spread by `/total_steps`),
  config `reward.alpha/beta/gamma` + `use_stationary_scaling:true`. Q: are `beta`/`gamma`/CLI
  `--beta-alert`/`--gamma-dist` dead in constrained mode? (I5). Is KL-per-step vs energy-terminal
  balanced?

- **P-G MED / M — Markovianity & observation sufficiency.**
  `wrappers/state.py`, `state_interpreters.py:177-208` (AlertAware), `time_manager.py` (horizon).
  Q: does obs contain what's needed to predict the terminal energy/alert reward? Is the
  non-stationary (annealed) reward observable? `sac.log_std_init:-3` exploration adequacy.

- **P-H LOW / S — SAC hyperparameter sanity.**
  `experiment_manager_new.py:482-515`, config `training.sac.*`. buffer/batch/train_freq/
  learning_starts/gradient_steps=-1. Low priority; values look defensible.

### 2. Initial insights (V=verified from code, S=suspected)

- **I1 (V) Adaptive Lagrange is dead code.** `reward.py:797` comment "Fixed Lagrange
  multipliers — adaptive updates removed (unstable per-episode)"; there is NO λ update in
  `step()`. Yet config sets `use_adaptive_lagrange: true` + `lambda_*_eta` (default.yaml:85-91)
  and the class docstring (`reward.py:578-581`) still advertises `lambda += eta*(metric-tau)`.
  λ stays pinned at init 0.1 forever.
- **I2 (V) The constraint is effectively toothless.** Penalties are `λ * tanh_hinge(...)` with
  λ=0.1 and `tanh_hinge ∈ [0,1]` (`reward.py:685-693`, `800-804`); quota disabled. Max total
  stealth penalty ≈ `0.1(alert)+0.1(KL) = 0.2`. Energy term is `alpha*log1p(energy_raw)` with
  alpha=0.34, unbounded and monotonically rewarding CPU (`reward.py:778`): ~0.37 at 3× CPU,
  ~0.78 at 10×. The agent can blow both budgets and still net positive. Central risk for the
  whole "stealthy" premise → reward hacking of stealth.
- **I3 (V) Curriculum schedule ignores real run length.** `reward.py:633` reads
  `config.get('training.num_episodes')` from the global singleton, which only holds
  yaml-default `100` (default.yaml:206). CLI `--num-episodes` lands in the `overrides` dict,
  but `config.py` has no setter and env-builder `get_config` closures
  (`experiment_manager_new.py:121`) are bypassed by wrappers that call `config.get()` directly.
  Confirmed on run `train_20260702143723/config.json`: effective num_episodes=100.
  ⇒ warmup = 100*0.5 = 50 episodes; τ anneals from relaxed→strict within the first ~50 episodes
  of any run. For a 50 000-episode run the curriculum is over instantly and the agent trains
  under strict τ the whole time. (Combined with I1/I2, curriculum barely matters anyway.)
- **I4 (V) softmax_temperature=20 makes the distribution head near-argmax.**
  `action_interpreters.py:100-102`: `softmax(20*x)`, x∈[0,1]. A 0.1 gap between two dims →
  e^2≈7.4× mass ratio; the head is winner-take-all, most of 20 logtypes get `int(≈0*20000)=0`
  logs (`:111`), and tiny action perturbations near the max cause large output swings
  (ill-conditioned for the policy gradient). Suspected to hurt exploration/credit assignment.
- **I5 (V) In constrained mode `beta`/`gamma` (and CLI --beta-alert/--gamma-dist) are no-ops.**
  `ConstrainedRewardWrapper` only consumes `alpha` (`experiment_manager_new.py:159-163`,
  `reward.py:800-804`); alert/KL weighting is governed solely by fixed λ. Documented CLI knobs
  silently do nothing here.
- **I6 (V) During-training eval barely functions.** `callbacks.eval.eval_freq:600000`
  (default.yaml:299) ≈ a full 50 000-episode run (12 steps × 50 000 = 600 000), so
  `n_calls % eval_freq == 0` fires ~once at the end. Plus `evaluation.deterministic:false` +
  `n_eval_episodes:1` ⇒ best_model.zip is picked from a single noisy stochastic episode.
- **I7 (S) Energy objective invites surrogate exploitation.** With `is_mock:true` (default),
  every training reward's CPU comes from per-rule joblib regressors (`reward.py:640-668`) fed
  `fake_dist/475796` and `(alerts-1)/203`. Maximizing predicted CPU with no penalty for leaving
  the models' training support is classic learned-reward hacking. Needs @security judgement on
  whether these CPU models + `expected_alerts_per_rule` are faithful.
- **I8 (S/minor) Non-stationary reward is unobservable, and obs dtype mismatch.** Obs has no
  training-progress feature, so annealed τ is off-policy-invisible (minor given I3). Also
  `state.py:108` casts obs to float32 while interpreter spaces declare `dtype=float64`
  (`state_interpreters.py:152` etc.) — cosmetic, flag to @engineer.

### 3. Cross-agent notes / questions

- **@ml-software-engineer:** Config plumbing bug behind I3 — CLI/`overrides` never written back
  to the `config` singleton (`config.py` has no `set`); any wrapper calling `config.get()`
  directly (e.g. `ConstrainedRewardWrapper.total_training_episodes`, all `reward.*` reads)
  silently ignores CLI overrides. This is the root cause of the broken curriculum schedule.
  Also the float32/float64 obs-space dtype mismatch (I8).
- **@splunk-security-researcher:** (a) Are the per-rule CPU surrogate models
  (`src/cpu_model_*.joblib`) and `reward.expected_alerts_per_rule` (default.yaml:15) trustworthy
  optimization targets, or will an agent maximizing them produce unrealistic injections? (b) Is
  the alert stealth signal — predicted alerts ≈ `baseline_alert + diversity`
  (`reward.py:41-57`) — a meaningful proxy for "not tripping the SOC", i.e. does per-rule alert
  count actually track detection risk?
