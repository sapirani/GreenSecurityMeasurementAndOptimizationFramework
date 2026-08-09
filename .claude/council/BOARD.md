# Council Board

Shared state for the three-agent council. See `PROTOCOL.md` for the rules.
Maintained by the orchestrator (main Claude session) after each council round.

## Accepted decisions

_(none yet — awaiting user greenlight on the Round-1 proposal queue below.)_

## Open questions

- @drl — is the legacy (non-constrained) reward branch still a supported mode, or dead? Gates a ~700-line quarantine. — raised by @engineer — 2026-08-05
- @security — are the per-rule CPU `.joblib` surrogates + `expected_alerts_per_rule` trustworthy optimization targets, and is `predicted_alerts = baseline + diversity` a meaningful stealth proxy? — raised by @drl — 2026-08-05
- @security — are manual-policy log-type baselines threat-model-valid? — raised by @engineer — 2026-08-05
- @drl — is `total_steps` reward scaling intended? — raised by @engineer — 2026-08-05

## Blockers

- Security findings #1/#3/#5 need the 9 saved-search definitions (`savedsearches.conf`). Confirmed NOT in repo and NOT at `/opt/splunk/etc/users/shouei/search/local/` on this host. **User must supply a dump.**

## Round 1 proposal queue (for @engineer, pending user greenlight)

- **B1 (bug, high)** Config CLI overrides never reach the singleton — `config.py` has no setter; `--num-episodes` etc. stay in an `overrides` dict. Root cause of the curriculum mis-scaling. (drl + engineer)
- **B2 (bug, low)** `check_license_usage` double-dispatches the search — `splunk_tools.py:472-473`. (security)
- **R1 (reward, high)** Constrained reward doesn't constrain: adaptive λ disabled (`reward.py:797`) despite config; λ pinned at 0.1, stealth penalty ≤~0.2 vs unbounded energy. Decide: re-enable adaptive λ or re-weight. (drl, security concurs)
- **R2 (reward/design, high)** Stealth ignores host cardinality + is volume-independent — single-host flood is invisible to reward. Needs a design decision before code. (security, drl concurs)
- **A1 (action, med)** `softmax_temperature=20` → near-argmax distribution head, ill-conditioned. (drl)
- **E1 (eval, med)** `eval_freq=600000` (~whole run) + `n_eval_episodes=1` + `deterministic:false` → best-model selection on one noisy episode. (drl)
- **Q1 (cleanup, med)** Quarantine/remove dead code: legacy reward branch (pending @drl answer), `unsused_scripts/` (30 files), loose src scripts/notebooks. (engineer)
- **T1 (tests, high)** Zero DRL-side test coverage (only `tests/scanner_tests.py`). (engineer)

## Round log

- 2026-08-05 — Round 1: whole-project review-plan scoping — all 3 agents ran (parallel) — each filed a prioritized review plan + initial insights; convergence on config-plumbing bug (B1) and under-enforced stealth (R1/R2). Awaiting user greenlight on queue.
