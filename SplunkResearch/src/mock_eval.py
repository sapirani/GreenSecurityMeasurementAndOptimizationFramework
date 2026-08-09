#!/usr/bin/env python3
"""
mock_eval.py — Load a trained DRL log-injection model and run it in mock mode.

Prints human-readable descriptions of each action taken, e.g.:
  "5,234 logs of EventCode 4662 with 15 diversity  → AD Replication Attack"

Usage (from project root):
    python -m SplunkResearch.src.mock_eval
  or
    python SplunkResearch/src/mock_eval.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/home/shouei/GreenSecurityMeasurementAndOptimizationFramework"
SPLUNK_ROOT  = os.path.join(PROJECT_ROOT, "SplunkResearch")
SRC_DIR      = os.path.join(SPLUNK_ROOT, "src")

for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(
    SPLUNK_ROOT,
    "host_132.72.80.159_experiments/runs"
    "/train_20260224000500/models/checkpoints"
    "/checkpoint_1400000_steps.zip",
)
TOP_LOGTYPES_CSV = os.path.join(SPLUNK_ROOT, "resources/top_logtypes.csv")

# ── Experiment configuration (from config.json overrides) ─────────────────────
MAX_LOGTYPES        = 20
LOG_TYPES_FILTER    = ["wineventlog:security", "wineventlog:system"]
N_EPISODES          = 30        # number of mock episodes to run
N_STEPS_PER_EPISODE = 12       # sac.steps_per_episode
N_RULES             = 9
BASE_LOG_COUNT      = 20_000
DIVERSITY_FACTOR    = 30
SOFTMAX_TEMPERATURE = 20
SIGMOID_SHARPNESS   = 10
ADDITIONAL_PCT      = 1.0      # additional_percentage

# Rule names (from config.json)
RULE_NAMES = [
    "Windows Event For Service Disabled",
    "Detect New Local Admin Account",
    "ESCU Network Share Discovery Via Dir Command Rule",
    "Known Services Killed by Ransomware",
    "Non Chrome Process Accessing Chrome Default Dir",
    "Kerberoasting SPN Request With RC4 Encryption",
    "Clop Ransomware Known Service Name",
    "Windows AD Replication Request Initiated from Unsanctioned Location",
    "ESCU Windows Rapid Authentication On Multiple Hosts Rule",
]

# Relevant logtypes per rule (from section_logtypes.py)
RULE_LOGTYPES: dict[str, list[tuple[str, str]]] = {
    "Windows Event For Service Disabled":                             [("wineventlog:system",   "7040")],
    "Detect New Local Admin Account":                                  [("wineventlog:security", "4732")],
    "ESCU Network Share Discovery Via Dir Command Rule":               [("wineventlog:security", "5140")],
    "Known Services Killed by Ransomware":                            [("wineventlog:system",   "7036")],
    "Non Chrome Process Accessing Chrome Default Dir":                 [("wineventlog:security", "4663")],
    "Kerberoasting SPN Request With RC4 Encryption":                   [("wineventlog:security", "4769")],
    "Clop Ransomware Known Service Name":                              [("wineventlog:system",   "7045")],
    "Windows AD Replication Request Initiated from Unsanctioned Location": [("wineventlog:security", "4662")],
    "ESCU Windows Rapid Authentication On Multiple Hosts Rule":        [("wineventlog:security", "4624")],
}

# Reverse map: logtype tuple → list of rule indices it targets
LOGTYPE_TO_RULE_INDICES: dict[tuple[str, str], list[int]] = {}
for rule_idx, rule in enumerate(RULE_NAMES):
    for lt in RULE_LOGTYPES.get(rule, []):
        LOGTYPE_TO_RULE_INDICES.setdefault(lt, []).append(rule_idx)

# Human-readable Windows Event Code descriptions
EVENT_DESCRIPTIONS: dict[str, str] = {
    "4624":  "Logon Success",
    "4625":  "Logon Failure",
    "4627":  "Group Membership",
    "4634":  "Account Logoff",
    "4648":  "Explicit Credential Logon",
    "4662":  "AD Object Operation",
    "4663":  "File / Object Access Attempt",
    "4672":  "Special Privileges Logon",
    "4688":  "Process Creation",
    "4697":  "Service Installed",
    "4698":  "Scheduled Task Created",
    "4699":  "Scheduled Task Deleted",
    "4700":  "Scheduled Task Enabled",
    "4701":  "Scheduled Task Disabled",
    "4702":  "Scheduled Task Updated",
    "4720":  "User Account Created",
    "4732":  "Member Added to Security Group",
    "4735":  "Security Group Changed",
    "4769":  "Kerberos Service Ticket Requested",
    "4799":  "Security-Enabled Group Membership Enum",
    "4907":  "Audit Policy Changed",
    "5038":  "Code Integrity Check",
    "5058":  "Key File Operation",
    "5059":  "Key Migration Operation",
    "5061":  "Cryptographic Operation",
    "5140":  "Network Share Object Access",
    "5379":  "Credential Manager Credential Read",
    "7036":  "Service Control Manager — State Change",
    "7040":  "Service Start Type Changed",
    "7045":  "New Service Installed",
    # System / generic
    "44":    "PnP Activity",
    "7":     "Kernel Event",
    "1112":  "Group Policy Refresh",
    "108":   "Event Log Init",
    "1500":  "Windows Firewall Event",
}


# ── Logtype list construction (mirrors experiment_manager_new.py) ─────────────

def load_top_logtypes() -> list[tuple[str, str]]:
    """Return the top MAX_LOGTYPES log types by historical count."""
    df = pd.read_csv(TOP_LOGTYPES_CSV)
    df = df[df["source"].str.lower().isin(LOG_TYPES_FILTER)]
    df = df.sort_values("count", ascending=False)
    raw = df[["source", "EventCode"]].values.tolist()[:MAX_LOGTYPES]
    return [(r[0].lower(), str(r[1])) for r in raw]


def build_logtype_lists(
    top_logtypes: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Mirror SplunkEnv.__init__: merge relevant + top, sorted."""
    relevant = sorted(
        {lt for rule in RULE_NAMES for lt in RULE_LOGTYPES.get(rule, [])}
    )
    merged = sorted(list(dict.fromkeys(relevant + top_logtypes)))
    return merged, relevant


# ── Observation builder (mirrors StateWrapper8 / AlertAwareInterpreter) ───────

def build_mock_observation(
    n_logtypes: int,
    step: int,
    total_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build a plausible mock observation for StateWrapper8.

    Shape: (n_logtypes*2 + 3 + n_rules*2,)
      [real_dist(n), fake_dist(n), total_ep_logs_norm, inserted_norm, step_ratio,
       expected_baseline(n_rules), trigger_exposure(n_rules)]
    """
    progress = (step + 1) / total_steps

    # Real distribution: Dirichlet-like, skewed toward common types
    weights = rng.exponential(scale=2.0, size=n_logtypes)
    real_dist = weights / weights.sum()

    # Fake (injected) distribution: grows with episode progress
    fake_raw = rng.exponential(scale=0.5, size=n_logtypes) * progress
    fake_dist = fake_raw / (fake_raw.sum() + 1e-10)

    # Volume metrics
    total_ep_logs  = int(BASE_LOG_COUNT * step * 10)   # ~real env logs
    inserted_logs  = int(BASE_LOG_COUNT * ADDITIONAL_PCT * step)
    total_ep_norm  = min(total_ep_logs / 500_000, 1.0)
    inserted_norm  = min((inserted_logs + total_ep_logs) / 500_000, 1.0)
    step_ratio     = step / total_steps

    # Alert-aware features (StateWrapper8 extras)
    # expected_alerts_per_rule from config, normalised by alert_normalize_factor=100
    expected_alerts_raw = np.array([4, 0.7, 0, 4, 0, 0, 0, 0, 0], dtype=float)
    expected_baseline   = expected_alerts_raw / 100.0

    # Agent's self-computed exposure (grows slightly with episode)
    trigger_exposure = rng.uniform(0.0, progress * 0.6, size=N_RULES)

    obs = np.concatenate([
        real_dist,
        fake_dist,
        [total_ep_norm, inserted_norm, step_ratio],
        expected_baseline,
        trigger_exposure,
    ]).astype(np.float64)

    return obs


# ── Action interpretation (mirrors SmoothTriggerInterpreter) ──────────────────

def interpret_smooth_trigger(
    raw_action: np.ndarray,
    top_logtypes: list[tuple[str, str]],
    relevant_logtypes: list[tuple[str, str]],
) -> tuple[dict, np.ndarray, dict]:
    """
    Pure replication of SmoothTriggerInterpreter.interpret().

    Returns:
      plan        — {key: {count, diversity, is_trigger, logtype, prob}}
      distribution — softmax probabilities over top_logtypes
      step_diag   — {logtype: {raw_div, trigger_prob, softmax_prob, log_count, is_trigger}}
                    only populated for relevant_logtypes
    """
    n_top = len(top_logtypes)

    # Softmax over first n_top dims
    dist_raw = raw_action[:n_top]
    scaled   = SOFTMAX_TEMPERATURE * dist_raw
    exp_vals = np.exp(scaled - np.max(scaled))
    distribution = exp_vals / exp_vals.sum()

    diversity_list = raw_action[n_top:]
    num_logs = ADDITIONAL_PCT * BASE_LOG_COUNT

    plan: dict = {}
    step_diag: dict = {}
    for i, logtype in enumerate(top_logtypes):
        log_count = int(distribution[i] * num_logs)
        is_trigger = 0
        diversity  = 0

        if logtype in relevant_logtypes:
            idx     = relevant_logtypes.index(logtype)
            raw_div = float(diversity_list[idx])
            raw_div_effective = 0.0 if log_count == 0 else raw_div
            # Sigmoid trigger
            trigger_prob = 1.0 / (1.0 + np.exp(-SIGMOID_SHARPNESS * (raw_div_effective - 0.5)))
            is_trigger   = int(trigger_prob > 0.5)
            if is_trigger:
                diversity = int(raw_div * DIVERSITY_FACTOR)

            step_diag[logtype] = {
                "raw_div":      raw_div,            # policy output (before zero-ing for empty logs)
                "raw_div_eff":  raw_div_effective,  # effective value used in sigmoid
                "trigger_prob": trigger_prob,
                "softmax_prob": float(distribution[i]),
                "log_count":    log_count,
                "is_trigger":   is_trigger,
            }

        diversity = max(1, min(diversity, max(log_count, 1)))

        key = f"{logtype[0]}_{logtype[1]}_{is_trigger}"
        plan[key] = {
            "count":      log_count,
            "diversity":  diversity,
            "is_trigger": is_trigger,
            "logtype":    logtype,
            "prob":       float(distribution[i]),
        }

    return plan, distribution, step_diag


# ── Pretty printing ───────────────────────────────────────────────────────────

def _logtype_label(lt: tuple[str, str]) -> str:
    src = "Security" if "security" in lt[0] else "System"
    return f"EventCode {lt[1]:>6}  ({src})"


def print_step(
    step_num: int,
    plan: dict,
    distribution: np.ndarray,
    top_logtypes: list[tuple[str, str]],
) -> None:
    total = sum(v["count"] for v in plan.values())
    malicious = {k: v for k, v in plan.items() if v["is_trigger"] and v["count"] > 0}
    benign    = {k: v for k, v in plan.items() if not v["is_trigger"] and v["count"] > 0}

    line = "─" * 70
    print(f"\n{line}")
    print(f"  STEP {step_num:>2}  │  Total: {total:>6,} logs injected  │  "
          f"Active types: {len(malicious) + len(benign)}")
    print(line)

    # ── Malicious injections ──────────────────────────────────────────────────
    if malicious:
        print(f"\n  ◆ ATTACK  ({len(malicious)} malicious log type(s))")
        for info in sorted(malicious.values(), key=lambda x: -x["count"]):
            lt   = info["logtype"]
            ec   = lt[1]
            desc = EVENT_DESCRIPTIONS.get(ec, f"Event {ec}")
            rule_hits = LOGTYPE_TO_RULE_INDICES.get(lt, [])
            rule_str  = "  ← " + " + ".join(
                f"Rule[{i}]: {RULE_NAMES[i]}" for i in rule_hits
            ) if rule_hits else ""

            print(
                f"    → {info['count']:>6,} logs of {ec:<6}  "
                f"with {info['diversity']:>2} diversity variants  │  "
                f"p={info['prob']:.3f}"
            )
            print(f"       {desc}{rule_str}")
    else:
        print("\n  ◆ ATTACK  (none triggered this step)")

    # ── Benign cover injections (top 5) ──────────────────────────────────────
    if benign:
        top5 = sorted(benign.values(), key=lambda x: -x["count"])[:5]
        print(f"\n  ◇ COVER   (top 5 of {len(benign)} benign types for camouflage)")
        for info in top5:
            lt   = info["logtype"]
            ec   = lt[1]
            desc = EVENT_DESCRIPTIONS.get(ec, f"Event {ec}")
            print(
                f"    · {info['count']:>6,} logs of {ec:<6}  │  "
                f"p={info['prob']:.3f}  │  {desc}"
            )

    # ── Distribution stats ────────────────────────────────────────────────────
    entropy = -np.sum(distribution * np.log(distribution + 1e-12))
    top_idx = int(np.argmax(distribution))
    top_lt  = top_logtypes[top_idx]
    top_pct = distribution[top_idx] * 100
    print(f"\n  Distribution entropy: {entropy:.3f} nats  │  "
          f"Dominant: EventCode {top_lt[1]} @ {top_pct:.1f}%")


def print_episode_summary(
    all_plans: list[dict],
    top_logtypes: list[tuple[str, str]],
    relevant_logtypes: list[tuple[str, str]],
) -> None:
    """Aggregate stats across the full mock episode."""
    agg: dict[tuple[str, str], dict] = {}
    for plan in all_plans:
        for info in plan.values():
            lt = info["logtype"]
            if lt not in agg:
                agg[lt] = {
                    "count": 0, "max_diversity": 0,
                    "triggered_steps": 0, "is_relevant": lt in relevant_logtypes,
                }
            agg[lt]["count"]         += info["count"]
            agg[lt]["max_diversity"]  = max(agg[lt]["max_diversity"], info["diversity"])
            if info["is_trigger"]:
                agg[lt]["triggered_steps"] += 1

    total_all   = sum(v["count"] for v in agg.values())
    attack_types = {k: v for k, v in agg.items() if v["triggered_steps"] > 0}
    active_types = {k: v for k, v in agg.items() if v["count"] > 0}

    h = "═" * 70
    print(f"\n{h}")
    print("  EPISODE SUMMARY")
    print(h)
    print(f"\n  Total logs injected : {total_all:>10,}")
    print(f"  Active log types    : {len(active_types)}")
    print(f"  Attack log types    : {len(attack_types)}")

    if attack_types:
        print(f"\n  ◆ Attack log types used (across {len(all_plans)} steps):")
        for lt, v in sorted(attack_types.items(), key=lambda x: -x[1]["count"]):
            ec        = lt[1]
            desc      = EVENT_DESCRIPTIONS.get(ec, f"Event {ec}")
            rule_idxs = LOGTYPE_TO_RULE_INDICES.get(lt, [])
            rule_str  = " + ".join(RULE_NAMES[i] for i in rule_idxs)
            print(
                f"    • EventCode {ec:<6}  │  {v['count']:>8,} logs total  │  "
                f"max diversity {v['max_diversity']:>2}  │  "
                f"triggered {v['triggered_steps']}/{len(all_plans)} steps"
            )
            print(f"      Desc:  {desc}")
            if rule_str:
                print(f"      Rule:  {rule_str}")

    if active_types:
        cover = {k: v for k, v in active_types.items() if v["triggered_steps"] == 0}
        print(f"\n  ◇ Benign / cover log types ({len(cover)} unique):")
        for lt, v in sorted(cover.items(), key=lambda x: -x[1]["count"])[:8]:
            ec   = lt[1]
            desc = EVENT_DESCRIPTIONS.get(ec, f"Event {ec}")
            print(f"    · EventCode {ec:<6}  │  {v['count']:>8,} logs  │  {desc}")

    print(f"\n{h}\n")


# ── Multi-episode summary ─────────────────────────────────────────────────────

def print_multi_episode_summary(
    per_episode_plans: list[list[dict]],
    top_logtypes: list[tuple[str, str]],
    relevant_logtypes: list[tuple[str, str]],
) -> None:
    """Aggregate attack / cover stats across all episodes."""
    n_ep = len(per_episode_plans)
    # {logtype: {total_count, triggered_steps, total_episodes_active}}
    agg: dict[tuple[str, str], dict] = {}

    for ep_plans in per_episode_plans:
        seen_this_ep: set[tuple[str, str]] = set()
        for plan in ep_plans:
            for info in plan.values():
                lt = info["logtype"]
                if lt not in agg:
                    agg[lt] = {
                        "total_count": 0,
                        "triggered_steps": 0,
                        "episodes_active": 0,
                        "is_relevant": lt in relevant_logtypes,
                    }
                agg[lt]["total_count"]    += info["count"]
                if info["is_trigger"]:
                    agg[lt]["triggered_steps"] += 1
                seen_this_ep.add(lt)
        for lt in seen_this_ep:
            if agg[lt]["total_count"] > 0:
                agg[lt]["episodes_active"] += 1

    attack_types = {k: v for k, v in agg.items() if v["triggered_steps"] > 0}
    cover_types  = {k: v for k, v in agg.items()
                    if v["total_count"] > 0 and v["triggered_steps"] == 0}

    h = "╔" + "═" * 68 + "╗"
    f = "╚" + "═" * 68 + "╝"
    print(f"\n{h}")
    print(f"  MULTI-EPISODE SUMMARY  ({n_ep} episodes × {N_STEPS_PER_EPISODE} steps)")
    print(f"{f}")
    total_logs = sum(v["total_count"] for v in agg.values())
    print(f"\n  Total logs injected across all episodes: {total_logs:>12,}")
    print(f"  Unique active log types                : {len(agg)}")
    print(f"  Attack log types (triggered ≥1 step)   : {len(attack_types)}")

    if attack_types:
        print(f"\n  ◆ Attack types (sorted by total count):")
        for lt, v in sorted(attack_types.items(), key=lambda x: -x[1]["total_count"]):
            ec        = lt[1]
            desc      = EVENT_DESCRIPTIONS.get(ec, f"Event {ec}")
            rule_idxs = LOGTYPE_TO_RULE_INDICES.get(lt, [])
            rule_str  = " + ".join(RULE_NAMES[i] for i in rule_idxs)
            print(
                f"    • EventCode {ec:<6}  │  {v['total_count']:>10,} logs total  │  "
                f"triggered {v['triggered_steps']:>3} steps  │  "
                f"active in {v['episodes_active']}/{n_ep} episodes"
            )
            print(f"      {desc}" + (f"  ←  {rule_str}" if rule_str else ""))

    if cover_types:
        print(f"\n  ◇ Top-8 cover types by total count:")
        for lt, v in sorted(cover_types.items(), key=lambda x: -x[1]["total_count"])[:8]:
            ec   = lt[1]
            desc = EVENT_DESCRIPTIONS.get(ec, f"Event {ec}")
            print(
                f"    · EventCode {ec:<6}  │  {v['total_count']:>10,} logs  │  "
                f"active in {v['episodes_active']}/{n_ep} episodes  │  {desc}"
            )

    print(f"\n{'═' * 70}\n")


# ── Trigger decision analysis ─────────────────────────────────────────────────

def _ascii_hist(values: list[float], n_bins: int = 10, width: int = 30) -> str:
    """Return a single-line ASCII histogram of values in [0, 1], marking 0.5."""
    bins = [0] * n_bins
    for v in values:
        b = min(int(v * n_bins), n_bins - 1)
        bins[b] += 1
    max_count = max(bins) or 1
    bar_chars = []
    threshold_bin = n_bins // 2  # bin that straddles 0.5
    for b, cnt in enumerate(bins):
        bar_len = int(cnt / max_count * width)
        char = "█" if b < threshold_bin else "░"
        bar_chars.append(char * bar_len if bar_len else ("│" if b == threshold_bin else "·"))
    return "|" + "|".join(bar_chars) + f"|  (0←{'─'*8}→1, threshold at ┤)"


def print_trigger_analysis(
    all_step_diags: list[dict],   # flat list; each dict: {lt: step_diag_entry}
    relevant_logtypes: list[tuple[str, str]],
) -> None:
    """
    Explain, per relevant logtype, why the agent did or did not trigger it.

    The trigger gate is a sigmoid on raw_div with threshold 0.5:
      trigger_prob = sigmoid(SIGMOID_SHARPNESS * (raw_div - 0.5))
    So raw_div > 0.5  →  trigger, raw_div < 0.5  →  no trigger.
    """
    # Collect per-logtype lists of raw_div_eff, softmax_prob, is_trigger
    per_lt: dict[tuple[str, str], dict] = {lt: {
        "raw_divs": [], "raw_divs_eff": [], "trigger_probs": [],
        "softmax_probs": [], "log_counts": [], "triggered": [],
    } for lt in relevant_logtypes}

    for diag in all_step_diags:
        for lt, d in diag.items():
            if lt not in per_lt:
                continue
            per_lt[lt]["raw_divs"].append(d["raw_div"])
            per_lt[lt]["raw_divs_eff"].append(d["raw_div_eff"])
            per_lt[lt]["trigger_probs"].append(d["trigger_prob"])
            per_lt[lt]["softmax_probs"].append(d["softmax_prob"])
            per_lt[lt]["log_counts"].append(d["log_count"])
            per_lt[lt]["triggered"].append(d["is_trigger"])

    total_steps = len(all_step_diags)
    h = "─" * 70
    print(f"\n{'═' * 70}")
    print("  TRIGGER DECISION ANALYSIS  (why each relevant logtype was / wasn't triggered)")
    print(f"  Sigmoid gate: trigger iff  raw_div > 0.5  "
          f"(sharpness={SIGMOID_SHARPNESS}, so Δ=0.05 → prob ≈ 0.38→0.62)")
    print(f"{'═' * 70}")

    for lt in relevant_logtypes:
        d = per_lt[lt]
        if not d["raw_divs"]:
            continue

        ec   = lt[1]
        desc = EVENT_DESCRIPTIONS.get(ec, f"Event {ec}")
        rule_idxs = LOGTYPE_TO_RULE_INDICES.get(lt, [])
        rule_str  = " + ".join(RULE_NAMES[i] for i in rule_idxs)

        raw_arr  = np.array(d["raw_divs"])
        eff_arr  = np.array(d["raw_divs_eff"])
        prob_arr = np.array(d["softmax_probs"])
        trig_arr = np.array(d["triggered"])
        cnt_arr  = np.array(d["log_counts"])

        n_observed   = len(raw_arr)
        n_triggered  = int(trig_arr.sum())
        n_zero_count = int((cnt_arr == 0).sum())    # steps where log_count=0 → raw_div zeroed

        above_thresh_raw = (raw_arr > 0.5).sum()    # policy output > 0.5 (ignoring zero-count)
        above_thresh_eff = (eff_arr > 0.5).sum()    # effective (after zeroing empty steps)

        print(f"\n  EventCode {ec}  —  {desc}")
        if rule_str:
            print(f"  Rule: {rule_str}")
        print(h)

        # raw_div statistics
        print(f"  raw_div  (policy output, dim {relevant_logtypes.index(lt)} of diversity head):")
        print(f"    mean={raw_arr.mean():.4f}  std={raw_arr.std():.4f}  "
              f"min={raw_arr.min():.4f}  max={raw_arr.max():.4f}  "
              f"median={np.median(raw_arr):.4f}")
        print(f"    Policy output > 0.5 : {above_thresh_raw}/{n_observed} steps "
              f"({100*above_thresh_raw/n_observed:.1f}%)")
        print(f"    Zeroed (log_count=0): {n_zero_count}/{n_observed} steps "
              f"({100*n_zero_count/n_observed:.1f}%)  ← softmax gave this LT 0 logs")
        print(f"    Effective > 0.5     : {above_thresh_eff}/{n_observed} steps "
              f"({100*above_thresh_eff/n_observed:.1f}%)  ← actually triggered")

        # softmax probability stats
        print(f"  Softmax probability (volume allocation):")
        print(f"    mean={prob_arr.mean():.4f}  std={prob_arr.std():.4f}  "
              f"min={prob_arr.min():.4f}  max={prob_arr.max():.4f}")
        mean_count = cnt_arr.mean()
        print(f"    → mean log_count per step: {mean_count:.1f}  "
              f"(zeroed {n_zero_count} steps)")

        # Verdict
        print(f"  Triggered: {n_triggered}/{n_observed} steps  "
              f"({100*n_triggered/n_observed:.1f}%)")

        # Root cause
        reasons = []
        if raw_arr.mean() < 0.5:
            reasons.append(
                f"policy diversity output is below threshold on average "
                f"(mean {raw_arr.mean():.3f} < 0.5)"
            )
        if n_zero_count > n_observed * 0.3:
            reasons.append(
                f"softmax allocated 0 logs in {n_zero_count}/{n_observed} steps, "
                f"zeroing raw_div → sigmoid < 0.5"
            )
        if raw_arr.max() < 0.5:
            reasons.append("policy NEVER output raw_div > 0.5 — this LT is suppressed by the policy")
        elif above_thresh_raw > 0 and above_thresh_eff == 0:
            reasons.append(
                f"policy did output raw_div > 0.5 in {above_thresh_raw} steps, "
                f"but log_count was 0 every time → zeroed out"
            )

        if reasons:
            print(f"  Root cause:")
            for r in reasons:
                print(f"    • {r}")
        elif n_triggered == n_observed:
            print(f"  Root cause: always triggered — policy consistently outputs raw_div > 0.5")

        # ASCII histogram of raw_div values
        hist_str = _ascii_hist(d["raw_divs"])
        print(f"  Distribution of raw_div: {hist_str}")
        print(f"  (each segment = [0,0.1),[0.1,0.2),...  "
              f"█=below threshold, ░=above threshold)")

    print(f"\n{'═' * 70}\n")


# ── State / observation analysis ──────────────────────────────────────────────

def _decode_obs_features(
    obs: np.ndarray,
    n_logtypes: int,
    top_logtypes: list[tuple[str, str]],
) -> dict[str, float]:
    """
    Decode a flat observation vector into a named-feature dict.

    Layout (StateWrapper8 / AlertAwareInterpreter):
      [real_dist(n), fake_dist(n), total_ep_logs_norm, inserted_norm, step_ratio,
       expected_baseline(N_RULES), trigger_exposure(N_RULES)]
    """
    n = n_logtypes
    features: dict[str, float] = {}

    # Global volume / timing
    features["step_ratio"]        = float(obs[2*n + 2])
    features["total_ep_logs_norm"] = float(obs[2*n])
    features["inserted_norm"]      = float(obs[2*n + 1])

    # Per-rule alert features
    for r in range(N_RULES):
        short = RULE_NAMES[r].split()[-1]   # last word as compact label
        features[f"expected_baseline[{r}:{short}]"] = float(obs[2*n + 3 + r])
        features[f"trigger_exposure[{r}:{short}]"]  = float(obs[2*n + 3 + N_RULES + r])

    # Per-logtype distribution features
    for i, lt in enumerate(top_logtypes):
        ec = lt[1]
        features[f"real_dist[{ec}]"] = float(obs[i])
        features[f"fake_dist[{ec}]"] = float(obs[n + i])

    return features


def print_state_analysis(
    all_step_obs:   list[np.ndarray],
    all_step_diags: list[dict],
    relevant_logtypes: list[tuple[str, str]],
    top_logtypes:      list[tuple[str, str]],
    n_logtypes:        int,
) -> None:
    """
    For each relevant logtype, compare the observation features at steps where
    the agent triggered it vs steps where it did not.

    Ranks features by |mean_triggered − mean_not_triggered| so the most
    discriminative state dimensions appear first.
    """
    assert len(all_step_obs) == len(all_step_diags)

    h  = "─" * 70
    print(f"\n{'═' * 70}")
    print("  STATE ANALYSIS  (what observation features correlate with each trigger decision)")
    print(f"  Observation layout: real_dist({n_logtypes}) | fake_dist({n_logtypes}) | "
          f"vol×3 | expected_baseline({N_RULES}) | trigger_exposure({N_RULES})")
    print(f"{'═' * 70}")

    for lt in relevant_logtypes:
        ec   = lt[1]
        desc = EVENT_DESCRIPTIONS.get(ec, f"Event {ec}")
        rule_idxs = LOGTYPE_TO_RULE_INDICES.get(lt, [])
        rule_str  = " + ".join(RULE_NAMES[i] for i in rule_idxs)

        triggered_obs:     list[dict] = []
        not_triggered_obs: list[dict] = []

        for obs, diag in zip(all_step_obs, all_step_diags):
            if lt not in diag:
                continue
            feat = _decode_obs_features(obs, n_logtypes, top_logtypes)
            if diag[lt]["is_trigger"]:
                triggered_obs.append(feat)
            else:
                not_triggered_obs.append(feat)

        n_trig     = len(triggered_obs)
        n_not_trig = len(not_triggered_obs)

        print(f"\n  EventCode {ec}  —  {desc}")
        if rule_str:
            print(f"  Rule: {rule_str}")
        print(f"  Steps: {n_trig} triggered  /  {n_not_trig} not triggered")
        print(h)

        if n_trig == 0:
            print("  (never triggered — no comparison possible)")
            continue
        if n_not_trig == 0:
            print("  (always triggered — no comparison possible)")
            continue

        # Build feature arrays
        all_keys = list(triggered_obs[0].keys())
        trig_mat     = np.array([[d[k] for k in all_keys] for d in triggered_obs])
        not_trig_mat = np.array([[d[k] for k in all_keys] for d in not_triggered_obs])

        trig_mean     = trig_mat.mean(axis=0)
        not_trig_mean = not_trig_mat.mean(axis=0)
        trig_std      = trig_mat.std(axis=0)
        not_trig_std  = not_trig_mat.std(axis=0)
        delta         = trig_mean - not_trig_mean

        # Rank by |Δ|, show top 10
        ranked = sorted(enumerate(all_keys), key=lambda x: -abs(delta[x[0]]))

        print(f"  {'Feature':<42} {'triggered':>10} {'not-trig':>10} {'Δ':>8}  direction")
        print(f"  {'─'*42} {'─'*10} {'─'*10} {'─'*8}  {'─'*12}")
        for idx, key in ranked[:10]:
            d = delta[idx]
            direction = ("↑ higher when triggered" if d > 0.005
                         else "↓ lower when triggered" if d < -0.005
                         else "≈ no difference")
            print(
                f"  {key:<42} "
                f"{trig_mean[idx]:>7.4f}±{trig_std[idx]:.3f} "
                f"{not_trig_mean[idx]:>7.4f}±{not_trig_std[idx]:.3f} "
                f"{d:>+8.4f}  {direction}"
            )

        # Highlight the single most predictive feature
        best_idx, best_key = ranked[0]
        best_d = delta[best_idx]
        print(f"\n  Most discriminative feature: '{best_key}'  Δ={best_d:+.4f}")
        if abs(best_d) < 0.005:
            print("  → No single observation feature strongly predicts this trigger.")
            print("    The decision is likely driven by the policy's internal bias")
            print("    (raw_div systematically near threshold) rather than state cues.")

    print(f"\n{'═' * 70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from stable_baselines3 import SAC
    import gymnasium as gym

    banner = "═" * 70
    print(f"\n{banner}")
    print("  DRL LOG-INJECTION MODEL — MOCK EVALUATION")
    print(f"  Checkpoint: {os.path.basename(CHECKPOINT)}")
    print(banner)

    # 1. Compute logtype lists exactly as experiment_manager_new.py does
    print("\n  [1/4] Building log-type lists from historical data...")
    top20 = load_top_logtypes()
    top_logtypes, relevant_logtypes = build_logtype_lists(top20)
    n_logtypes = len(top_logtypes)

    print(f"        Merged top logtypes  : {n_logtypes}")
    print(f"        Relevant (attack) LTs: {len(relevant_logtypes)}")
    print(f"        Merged list (sorted) :")
    for i, lt in enumerate(top_logtypes):
        marker = " ★" if lt in relevant_logtypes else "  "
        desc   = EVENT_DESCRIPTIONS.get(lt[1], "")
        print(f"          [{i:>2}]{marker} {lt[0]}_{lt[1]:<6}  {desc}")

    # 2. Compute space dimensions
    # StateWrapper8 (AlertAwareInterpreter): n_logtypes*2 + 3 + n_rules*2
    obs_dim = n_logtypes * 2 + 3 + N_RULES * 2
    # SmoothTrigger: len(top_logtypes) + len(relevant_logtypes)
    act_dim = n_logtypes + len(relevant_logtypes)

    print(f"\n  [2/4] Space dimensions:")
    print(f"        Observation dim : {obs_dim}  (= {n_logtypes}*2 + 3 + {N_RULES}*2)")
    print(f"        Action dim      : {act_dim}  (= {n_logtypes} + {len(relevant_logtypes)})")

    # 3. Load model
    print(f"\n  [3/4] Loading SAC checkpoint...")
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,),  dtype=np.float64)
    act_space = gym.spaces.Box(low=0.0, high=1.0, shape=(act_dim,),  dtype=np.float32)

    model = SAC.load(
        CHECKPOINT,
        custom_objects={
            "observation_space": obs_space,
            "action_space":      act_space,
        },
    )
    print(f"        Policy type: {type(model.policy).__name__}")
    print(f"        Device:      {model.device}")

    # 4. Run mock episodes
    print(f"\n  [4/4] Running {N_EPISODES} mock episodes ({N_STEPS_PER_EPISODE} steps each)...")

    per_episode_plans: list[list[dict]] = []
    all_step_diags:    list[dict]       = []   # flat; one entry per step across all episodes
    all_step_obs:      list[np.ndarray] = []   # flat; paired with all_step_diags

    for ep in range(N_EPISODES):
        ep_banner = "┄" * 70
        print(f"\n{ep_banner}")
        print(f"  EPISODE {ep + 1} / {N_EPISODES}")
        print(ep_banner)

        rng       = np.random.default_rng(seed=ep)   # different seed per episode
        all_plans: list[dict] = []

        for step in range(N_STEPS_PER_EPISODE):
            obs = build_mock_observation(n_logtypes, step, N_STEPS_PER_EPISODE, rng)
            raw_action, _ = model.predict(obs, deterministic=True)
            plan, distribution, step_diag = interpret_smooth_trigger(
                raw_action, top_logtypes, relevant_logtypes
            )
            all_plans.append(plan)
            all_step_diags.append(step_diag)
            all_step_obs.append(obs)
            print_step(step + 1, plan, distribution, top_logtypes)

        print_episode_summary(all_plans, top_logtypes, relevant_logtypes)
        per_episode_plans.append(all_plans)

    if N_EPISODES > 1:
        print_multi_episode_summary(per_episode_plans, top_logtypes, relevant_logtypes)

    print_trigger_analysis(all_step_diags, relevant_logtypes)
    print_state_analysis(all_step_obs, all_step_diags, relevant_logtypes, top_logtypes, n_logtypes)


if __name__ == "__main__":
    main()
