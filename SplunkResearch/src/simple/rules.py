"""Static research data: the detection rules, their log types, and the log-type
space the agent injects into.

This replaces the config-driven `reward.rule_names` / `expected_alerts_per_rule`
lists and the `section_logtypes` / `top_logtypes.csv` plumbing that used to be
threaded through the env and every TensorBoard writer. It is plain data — no
config lookups, no side effects.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd

LogType = Tuple[str, str]  # (source, eventcode), both lower-cased strings

# ---------------------------------------------------------------------------
# The 9 detection rules, in the canonical order used by the "barely good" run.
# `expected_alert` is the stationary per-episode baseline alert count for the
# rule (from reward.expected_alerts_per_rule = [4, 0.7, 0, 4, 0, 0, 0, 0, 0]).
# `logtype` is the single Windows event that, when injected with malicious
# variants, makes the rule fire (from resources/section_logtypes.py).
# ---------------------------------------------------------------------------

RULES: List[Dict] = [
    {"name": "Windows Event For Service Disabled",                          "logtype": ("wineventlog:system",   "7040"), "expected_alert": 4.0},
    {"name": "Detect New Local Admin Account",                             "logtype": ("wineventlog:security", "4732"), "expected_alert": 0.7},
    {"name": "ESCU Network Share Discovery Via Dir Command Rule",          "logtype": ("wineventlog:security", "5140"), "expected_alert": 0.0},
    {"name": "Known Services Killed by Ransomware",                        "logtype": ("wineventlog:system",   "7036"), "expected_alert": 4.0},
    {"name": "Non Chrome Process Accessing Chrome Default Dir",            "logtype": ("wineventlog:security", "4663"), "expected_alert": 0.0},
    {"name": "Kerberoasting SPN Request With RC4 Encryption",             "logtype": ("wineventlog:security", "4769"), "expected_alert": 0.0},
    {"name": "Clop Ransomware Known Service Name",                         "logtype": ("wineventlog:system",   "7045"), "expected_alert": 0.0},
    {"name": "Windows AD Replication Request Initiated from Unsanctioned Location", "logtype": ("wineventlog:security", "4662"), "expected_alert": 0.0},
    {"name": "ESCU Windows Rapid Authentication On Multiple Hosts Rule",   "logtype": ("wineventlog:security", "4624"), "expected_alert": 0.0},
]

# Rules whose alert count does NOT scale with injected diversity — injecting
# more variants does not create more alerts (matches AlertPredictor's special
# case at wrappers/reward.py:50).
DIVERSITY_INSENSITIVE_RULES = {"ESCU Windows Rapid Authentication On Multiple Hosts Rule"}

RULE_NAMES: List[str] = [r["name"] for r in RULES]
EXPECTED_ALERTS: Dict[str, float] = {r["name"]: r["expected_alert"] for r in RULES}
RULE_LOGTYPE: Dict[str, LogType] = {r["name"]: r["logtype"] for r in RULES}

# Log types that actually trigger a rule (sorted, deduped) — the "relevant"
# subset. The action's per-rule trigger/diversity dims are indexed by this list.
RELEVANT_LOGTYPES: List[LogType] = sorted({r["logtype"] for r in RULES})


def logtype_key(lt: LogType, is_trigger: int = 0) -> str:
    """Canonical dict key for a log type, e.g. 'wineventlog:system_7040_1'."""
    return f"{lt[0]}_{lt[1]}_{is_trigger}"


def load_top_logtypes(csv_path: str,
                      sources=("wineventlog:security", "wineventlog:system"),
                      max_logtypes: int = 20) -> List[LogType]:
    """Build the injection log-type space: the `max_logtypes` most common event
    codes among `sources`, unioned with the relevant (rule-triggering) types,
    sorted and deduped. Mirrors experiment_manager_new.py:125-130 +
    custom_splunk_env.py:122-124 so dimensions match the old runs.
    """
    df = pd.read_csv(csv_path)
    df = df[df["source"].str.lower().isin(sources)]
    top = (df.sort_values("count", ascending=False)[["source", "EventCode"]]
             .values.tolist()[:max_logtypes])
    top = [(str(s).lower(), str(ec)) for s, ec in top]
    merged = sorted(dict.fromkeys(RELEVANT_LOGTYPES + top))
    return merged


DEFAULT_TOP_LOGTYPES_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resources", "top_logtypes.csv",
)
