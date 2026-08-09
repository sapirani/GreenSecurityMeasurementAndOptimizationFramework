# Inbox: engineer

Append-only. Only the owning agent writes here. See ../PROTOCOL.md for the entry format.

---

## 2026-08-05 — Scoping round: engineering review plan + first-pass insights

Survey only, no source edits. All claims read from source at the cited lines. Smoke-import in
`py310_modelenv` from project root passed for `config`, `wrappers.reward`,
`wrappers.reward_interpreters`, `wrappers.action_interpreters`, `wrappers.state_interpreters`
(only side effect: "Cannot connect to Elastic" warning, non-fatal).

### 1. Review plan (prioritized)

**A. Reward wrapper stack — active vs. dead paths (HIGH, med effort)**
- `experiment_manager_new.py:146-194` — `reward.reward_mode` defaults to `constrained`
  (`config/default.yaml:9`), so `_build_splunk_env` takes the `ConstrainedRewardWrapper` branch
  (`:158-163`) and the entire `else` branch (DistributionRewardWrapper / EnergyRewardWrapper /
  `create_alert_reward_wrapper`, `:164-194`) is **not exercised in the default config**.
- Inspect: is the whole `ALERT_REWARD_REGISTRY` + `AlertSignal`/`AlertNormalizer` machinery
  (`wrappers/reward_interpreters.py`, 352 lines) currently reachable? `reward.alert_method`
  (`default.yaml:53` = `ExpectedDiffPower`) and `reward.distribution_method` are read only in the
  legacy branch. Risk: large surface of un-run code that looks configurable but isn't; drift
  between the two reward implementations. @drl owns intent — see questions.
- `wrappers/reward.py:742-849` (`ConstrainedRewardWrapper.step`) is the real reward and deserves
  a correctness pass (per-step KL scaling `:754`, mock CPU override `:766-773`, clip `:807-809`).

**B. `experiment_manager_new.py` structure (HIGH, low-med effort)**
- 1204 lines; the nested `def get_config(key, default=None)` closure is copy-pasted **9 times**
  (`:121, :443, :618, :811, :853, :928, :959, :998, :1060`). Collapse to one helper
  (e.g. `self._cfg(overrides, key, default)` or a module fn). Pure DRY refactor, low blast radius,
  improves every method that reads config.

**C. Test coverage (HIGH, med effort)**
- Only `tests/scanner_tests.py` (680 bytes) exists — zero tests for the DRL side. Highest-value
  targets are pure/deterministic: `reward_interpreters.py` signals+normalizers, the
  `ALERT_REWARD_REGISTRY` factory (`reward.py:444-471`), `config.py` dot-access + deep-merge,
  `build_manual_policy_action` (`manual_policies.py:129`). These need no Splunk/Elastic and would
  lock in behavior before any reward refactor.

**D. Config handling (MED, low effort)**
- `config.py` is a clean singleton, but `config` is instantiated at import time (`:173`) and
  emits a `UserWarning` when `secrets.yaml` is absent (`:66-72`) — fine, but there's no schema/
  validation and no typo detection (`get` silently returns default on a misspelled dot-path,
  `:107-111`). Given the reward config has ~dozens of keys, a light validation pass or a
  `get_required` for load-bearing keys would catch silent misconfig. Inspect which keys are
  load-bearing vs optional.

**E. Callbacks (MED, med effort)** — `callbacks.py` (371 lines): `CustomEvalCallback3`,
  TensorBoard/HParams, licence check. Not yet read in depth this round; flag for a dedicated pass
  (writer lifecycle, eval env leakage). MEMORY notes prior SummaryWriter-close fixes — verify.

**F. Code health / dead code (MED, low effort)**
- `src/unsused_scripts/` — 30 files incl. `splunk_tools_old.py`, `experiment_manager.py`,
  `strategies/*` (superseded by `wrappers/`). Candidate for removal/quarantine once confirmed
  unreferenced.
- Loose top-level scripts in `src/`: `debug_drl_loop.py`, `mock_eval.py`, `optuna_search.py`,
  `sync_splunk_events.py`, `oneshot_import.py`, `energy_profile_final.py`, `manual_policies.py`,
  plus notebooks (`result_analysis.ipynb` 352K, `cpu_prediction_nb.ipynb` 193K,
  `splunk_tools_notebook.ipynb`) tracked in git. Not harmful but clutters the package namespace
  (`src` on sys.path). Worth a "what is still used" inventory.

**G. Performance / resource use (LOW-MED, med effort)** — SubprocVecEnv path
  (`experiment_manager_new.py:328-352`, `n_envs>1`) and repeated `pd.read_csv` of
  `all_dist_by_host.csv` in `splunk_tools.py:363` per baseline load. Defer until correctness settled.

### 2. Initial insights

**Verified**
- `splunk_tools.py:371` filter-not-assigned bug is **fixed** in the current working copy: the
  `source`-contains filter is assigned back at `:372-374` (`self.real_logs_distribution = ...`).
  No longer a live bug.
- `custom_splunk` namespace-shadowing fix is **present and correct**:
  `src/custom_splunk/__init__.py` redirects `__path__` to the inner dir and exec's the inner
  `__init__.py` for gym registration. Matches the documented gotcha.
- Duplicated `get_config` closure ×9 (see B) — verified by grep count.
- Default reward path is `constrained`; legacy alert/energy/distribution wrappers are bypassed
  by default (see A) — verified from `default.yaml:9` + `experiment_manager_new.py:158-194`.
- Uncommitted working changes add a manual-policy eval path (`run_experiment.py` +
  `experiment_manager_new.py:371-390 _create_manual_policy_model`) depending on untracked
  `manual_policies.py` (`ManualPolicyModel:23`, `build_manual_policy_action:129`). These are
  in-flight and untested by any harness.

**Suspected (unconfirmed)**
- `experiment_manager_new.py:158-163`: in `constrained` mode the `reward.use_distribution_reward`
  / `use_energy_reward` / `use_alert_reward` flags are still passed into
  `BaseRuleExecutionWrapperWithPrediction` (`:151-152`) but NOT used to gate the constrained
  wrapper (which reads its own `reward.use_*` internally). Possible double source of truth — two
  places read the same flags with different effect. Needs a read of
  `ConstrainedRewardWrapper.__init__` (`reward.py:584`) to confirm no conflict.
- `AlertRewardWrapper.step` (`reward.py:438`) multiplies reward by `self.unwrapped.total_steps`
  when `use_stationary_scaling` is False — same pattern in `ConstrainedRewardWrapper` — so total
  episode reward magnitude scales with episode length. Intentional per MEMORY, but worth a DRL
  sanity check (reward scale vs. SAC entropy/critic targets).

### 3. Cross-agent notes / questions

- @drl-researcher: Is the legacy reward branch (`AlertRewardWrapper` + registry, EnergyReward,
  DistributionRewardWrapper) still intended to be a supported/experimented path, or has
  `constrained` fully superseded it? If superseded, I can quarantine ~700 lines of reward code +
  the `ALERT_REWARD_REGISTRY`; if still live, it needs tests and the config flags reconciled
  (insight A / suspected point 1). Please confirm before I touch reward.py.
- @drl-researcher: The `use_stationary_scaling=False` default makes per-episode reward scale with
  `total_steps` (`reward.py:438`, `reward.py:745`). Is that the intended objective scale for SAC,
  or a legacy artifact? Affects whether I leave it or normalize.
- @security-researcher: `manual_policies` adds fixed baseline policies `only_4662` /
  `all_relevant` (`run_experiment.py` new `--manual-policy`). Do those log-type choices reflect a
  meaningful detection-engineering baseline (e.g. EventCode 4662 relevance)? Only asking so the
  baseline comparison is threat-model-valid; I own the wiring, you own whether it's the right set.

