"""Plot baseline alerts per rule over time with train/eval split.

NOTE: The CSV data uses rolling 48h windows with a 4h step (overlapping).
We plot alerts at the window midpoint to represent the signal faithfully
without double-counting via daily aggregation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

BASELINE_DIR = Path(__file__).parent.parent / "host_132.72.81.150_experiments" / "baseline"

# Load data
train = pd.read_csv(BASELINE_DIR / "baseline_splunk_train-v32_2880_132_72_81_150.csv")
eval_ = pd.read_csv(BASELINE_DIR / "baseline_splunk_eval-v32_2880_132_72_81_150.csv")

# Concatenate
df = pd.concat([train, eval_], ignore_index=True)

# Parse times; use window midpoint as the representative timestamp
df["start"] = pd.to_datetime(df["start_time"], format="%m/%d/%Y:%H:%M:%S")
df["end"] = pd.to_datetime(df["end_time"], format="%m/%d/%Y:%H:%M:%S")
df["mid"] = df["start"] + (df["end"] - df["start"]) / 2
df.sort_values("mid", inplace=True)

# Eval period boundary
eval_start = pd.to_datetime(eval_["start_time"], format="%m/%d/%Y:%H:%M:%S").min()

# All rules
all_rules = sorted(df["search_name"].unique())

# Shorten rule names for legend
name_map = {
    "Known Services Killed by Ransomware": "Known Services Killed",
    "Windows Event For Service Disabled": "Service Disabled",
    "ESCU Windows Rapid Authentication On Multiple Hosts Rule": "Rapid Auth Multi-Host",
    "Detect New Local Admin Account": "New Local Admin",
    "Clop Ransomware Known Service Name": "Clop Ransomware",
    "ESCU Network Share Discovery Via Dir Command Rule": "Net Share Discovery",
    "Kerberoasting SPN Request With RC4 Encryption": "Kerberoasting RC4",
    "Non Chrome Process Accessing Chrome Default Dir": "Non-Chrome Access",
    "Windows AD Replication Request Initiated from Unsanctioned Location": "AD Replication Unsanctioned",
}

# Plot
fig, ax = plt.subplots(figsize=(16, 7))

colors = plt.cm.tab10(np.linspace(0, 1, len(all_rules)))

for i, rule in enumerate(all_rules):
    rule_data = df[df["search_name"] == rule].copy()
    label = name_map.get(rule, rule)
    # Resample to daily mean to smooth the overlapping 48h windows
    rule_ts = rule_data.set_index("mid")["alert"].resample("1D").mean()
    ax.plot(rule_ts.index, rule_ts.values, label=label,
            color=colors[i], linewidth=1.5, alpha=0.85)

# Eval dashed line
ax.axvline(x=eval_start, color="red", linestyle="--", linewidth=2, label="Eval period starts")

# Formatting
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Alerts (daily mean per 48h window)", fontsize=12)
ax.set_title("Baseline Alerts per Rule Over Time", fontsize=14)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45, ha="right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = Path(__file__).parent / "baseline_alerts_per_rule.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.close()
