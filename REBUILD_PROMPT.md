# Project Brief: Adversarial Log-Injection RL Framework for SIEM Energy Analysis

## Context & Motivation

Security Information and Event Management (SIEM) systems like Splunk continuously run detection rules (saved searches) over incoming log data. Each rule execution consumes CPU, memory, and ultimately energy. The cost of running a rule is not constant — it depends on the *volume and shape* of the data the rule has to scan and correlate. This creates an under-studied attack surface: an adversary who can influence what gets logged may be able to inflate the computational (and therefore energy) cost of security monitoring itself, without ever triggering an obvious incident.

I want to build a research framework that studies this phenomenon by training a reinforcement-learning agent to inject synthetic log events into a Splunk instance in a way that **maximizes the CPU/energy cost of detection rules while remaining stealthy**. The purpose is defensive research: to quantify the vulnerability, understand the effectiveness-vs-detectability trade-off, and inform more resource-resilient monitoring design.

## Primary Goal

Train an RL agent whose actions control the injection of fake (but realistic) Windows Event Logs into a monitored Splunk index, learning a policy that drives up the resource consumption of a set of Splunk detection rules while staying below detectability thresholds.

## Objectives (the multi-objective the agent optimizes)

1. **Maximize energy/CPU impact** — increase the measured resource cost of running the target detection rules relative to a no-injection baseline.
2. **Preserve alert stealth** — do not meaningfully change how many alerts the detection rules fire compared to baseline; a spike in alerts would expose the manipulation.
3. **Preserve distributional stealth** — keep the injected log stream statistically close to the organic/background log distribution (e.g. by log type, event code, host), so the injection doesn't stand out to distribution-based anomaly detection.
4. **Respect an injection budget** — the volume of injected logs relative to real traffic should stay within a plausible, bounded ratio.

These objectives conflict (more injected volume → more energy but worse stealth), so the core research question is how to formulate and balance them. The framework should make the trade-off tunable and measurable, not hard-code a single balance point.

## Environment & Data Realities

- **Target system:** live Splunk instances reachable over REST/HEC. There are multiple hosts, selectable at runtime; each experiment targets one host.
- **Log domain:** Windows Event Logs (`wineventlog:security`, `wineventlog:system`, etc.), spanning dozens of log types and event codes. Injected logs must be template-realistic — plausible field values, timestamps within a defined episode time window, spread across a configurable set of hosts.
- **Detection rules:** a fixed set of ~9 named Splunk saved searches (e.g. new-local-admin detection, Kerberoasting, ransomware-service detection, AD replication anomalies). Each has an expected baseline alert rate and its own cost profile.
- **Measurement:** the framework must be able to measure the resource cost of rule execution (CPU/energy), and also compare injected-vs-baseline alert counts and log distributions. Running real detection rules is *slow and expensive*, so there needs to be a fast path for iteration.
- **Baselines:** a shared, reusable baseline measurement of un-tampered rule cost and organic log distribution, established once and referenced across experiments.
- **Compute:** training runs both locally and on a SLURM GPU cluster (single-GPU jobs, tens of GB RAM). Expect long training runs (tens of thousands of episodes) plus separate evaluation runs of a trained policy.

## Functional Requirements

- **Modes:** train a new policy, evaluate a saved policy, and continue-training (retrain) from a saved policy.
- **Fast-iteration mode:** a way to run the full loop *without* paying for real detection-rule execution — i.e. substitute the expensive measurement step with a learned/predicted cost model while keeping injection, observation, and reward logic identical. Everything except the rule-execution measurement should behave the same in fast mode vs. real mode.
- **Configurability without code changes:** all experiment parameters — objective weights, thresholds/budgets, algorithm choice, host selection, time windows, log rates — should be drivable from configuration and command-line overrides, with credentials kept separate from committed config.
- **Reproducibility & tracking:** every experiment should be self-contained and auditable — captured configuration snapshot, code version, status lifecycle (running → completed/failed/interrupted), logs, saved models/checkpoints, and metrics suitable for TensorBoard-style monitoring.
- **Extensibility:** it should be straightforward to add a new way of interpreting actions, a new observation/state representation, or a new reward/stealth formulation without rewriting the core training loop. The right abstraction boundaries are yours to design.
- **Baselines for comparison:** include non-learned reference policies (e.g. random and simple hand-crafted injection strategies) so trained agents can be evaluated against meaningful baselines.
- **Graceful operation:** long cluster jobs must handle interruption/termination cleanly and leave experiments in a recoverable, correctly-labeled state.

## Constraints

- **Language/stack:** Python. RL via an established library ecosystem (your choice, but it should support continuous-control algorithms since injection volumes/distributions are naturally continuous). Splunk interaction via its REST/HEC APIs.
- **Safety & scope:** this is authorized defensive research against instances I control. Injection targets only my own test indices/hosts. No credentials in source or committed config.
- **Cost discipline:** real Splunk measurement is the bottleneck; the design must minimize how often it's needed (batching, caching baselines, the fast-iteration mode above).
- **Separation of concerns:** experiment orchestration, environment/simulation logic, Splunk I/O, log generation, and configuration should be cleanly separable so pieces can be tested and swapped in isolation.

## Deliverables

1. A runnable training/evaluation pipeline meeting the objectives and modes above.
2. Configuration system with sane documented defaults plus per-experiment overrides.
3. A realistic synthetic-log generator for the Windows Event Log domain.
4. Baseline-measurement tooling and at least the fast-iteration (predicted-cost) path.
5. Experiment tracking/output layout that makes runs reproducible and comparable.
6. A short design document explaining the RL formulation you chose (state, action, reward, algorithm) and *why*, plus how to extend each part.

## What I'm Leaving to You

I deliberately have **not** specified the RL formulation (MDP/state/action/reward math), the wrapper/module architecture, the specific algorithm, the config schema, or the directory layout. Propose these. Where a decision has meaningful trade-offs (e.g. how to encode the stealth constraints, on-policy vs. off-policy, how to model injection actions, how the fast-cost model is trained), lay out the options and give me a recommendation before committing. Start by proposing an overall design and RL formulation, and let's align on that before you build.
