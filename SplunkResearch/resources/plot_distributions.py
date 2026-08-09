#!/usr/bin/env python3
"""Analyze and plot log type distribution and alert counts over time (2-day windows).

Queries Splunk for:
1. Event count per source (log type) in 2-day buckets
2. Alert count per detection rule in 2-day buckets

Usage (from project root):
    python -m SplunkResearch.resources.plot_distributions --ip 3 --index new_main
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "src" / ".env")

import splunklib.client as client
import splunklib.results as results


TOP_LOGTYPES_CSV = Path(__file__).resolve().parent / "top_logtypes.csv"
TOP_LOGTYPES_SOURCES = ["wineventlog:security", "wineventlog:system"]
TOP_LOGTYPES_LIMIT = 50


def load_top_logtypes(csv_path=TOP_LOGTYPES_CSV, sources=TOP_LOGTYPES_SOURCES, limit=TOP_LOGTYPES_LIMIT):
    """Load (source, EventCode) pairs from top_logtypes.csv — same selection as the env."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["source"].str.lower().isin(sources)]
    df = df.sort_values("count", ascending=False).head(limit)
    return [(row["source"], str(row["EventCode"])) for _, row in df.iterrows()]


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

SHORT_NAMES = {
    "ESCU - Known Services Killed by Ransomware - Rule": "Known Services Killed",
    "ESCU - Windows Event For Service Disabled - Rule": "Service Disabled",
    "ESCU - Windows Rapid Authentication On Multiple Hosts - Rule": "Rapid Auth Multi-Host",
    "ESCU - Detect New Local Admin account - Rule": "New Local Admin",
    "ESCU - Clop Ransomware Known Service Name - Rule": "Clop Ransomware",
    "ESCU - Network Share Discovery Via Dir Command - Rule": "Net Share Discovery",
    "ESCU - Kerberoasting spn request with RC4 encryption - Rule": "Kerberoasting RC4",
    "ESCU - Non Chrome Process Accessing Chrome Default Dir - Rule": "Non-Chrome Access",
    "ESCU - Windows AD Replication Request Initiated from Unsanctioned Location - Rule": "AD Replication Unsanctioned",
}


def connect(host, port=8089, username="splunk", password=None):
    if password is None:
        from SplunkResearch.src.config import config
        password = config.get("splunk.password", "")
    return client.connect(host=host, port=port, username=username, password=password, autologin=True)


def run_export(service, query, earliest="0", latest="now"):
    """Run an export search and return list of result dicts."""
    kwargs = {"earliest_time": earliest, "latest_time": latest, "count": 0}
    reader = results.JSONResultsReader(
        service.jobs.export(query, **kwargs, output_mode="json"))
    rows = []
    for item in reader:
        if isinstance(item, results.Message):
            continue
        rows.append(dict(item))
    return rows


def query_log_types(service, index, span="2d", top_logtypes=None):
    """Get event count per EventCode in time buckets, restricted to top_logtypes."""
    if not top_logtypes:
        top_logtypes = load_top_logtypes()

    # Group EventCodes per source so we can build one OR-clause per source.
    by_source = {}
    for src, ec in top_logtypes:
        by_source.setdefault(src, []).append(ec)
    clauses = [
        f'(source="{src}" (' + " OR ".join(f"EventCode={ec}" for ec in ecs) + "))"
        for src, ecs in by_source.items()
    ]
    filter_expr = " OR ".join(clauses)

    query = (f'search index={index} host!=132.72.81.150 host!=dt-splunk '
             f'({filter_expr}) '
             f'| bin _time span={span} '
             f'| stats count by _time EventCode source')
    print(f"Running EventCode distribution query for {len(top_logtypes)} top logtypes...")
    rows = run_export(service, query)
    print(f"  Got {len(rows)} rows")
    return rows


def _resolve_macros(service, query):
    """Resolve Splunk macros in a search query."""
    import re
    macro_pattern = re.compile(r'`(\w+)`')
    macros_found = macro_pattern.findall(query)
    for macro_name in macros_found:
        try:
            # Try to get macro definition
            response = service.get(f"/servicesNS/-/-/admin/macros/{macro_name}",
                                   output_mode="json")
            body = json.loads(response.body.read())
            definition = body["entry"][0]["content"]["definition"]
            query = query.replace(f"`{macro_name}`", definition)
        except Exception:
            # Common macros fallback
            known = {
                "wineventlog_system": 'source="WinEventLog:System"',
                "wineventlog_security": 'source="WinEventLog:Security"',
            }
            if macro_name in known:
                query = query.replace(f"`{macro_name}`", known[macro_name])
    return query


def query_alerts(service, index, rule_names, span="2d"):
    """For each saved search rule, resolve its query and count alerts per time bucket."""
    saved_searches = {}
    for ss in service.saved_searches:
        saved_searches[ss.name] = ss
    alert_data = {}

    for rule in rule_names:
        if rule not in saved_searches:
            print(f"  Warning: rule '{rule}' not found, skipping")
            continue

        ss = saved_searches[rule]
        base_search = ss.content.get("search", "")
        if not base_search:
            print(f"  Warning: rule '{rule}' has empty search, skipping")
            continue

        # Resolve macros and override index
        resolved = _resolve_macros(service, base_search)

        # Strip everything after the first | stats/table/rename (keep only the base filter)
        # so we can add our own bin + stats
        import re
        match = re.search(r'\|\s*(stats|table|rename|eval|transaction|bucket|dedup)', resolved)
        if match:
            base_filter = resolved[:match.start()].strip()
        else:
            base_filter = resolved.strip()

        # Prepend index
        wrapped = f'search index={index} {base_filter} | bin _time span={span} | stats count by _time'

        print(f"  Querying: {SHORT_NAMES.get(rule, rule)}...")
        try:
            rows = run_export(service, wrapped)
            alert_data[rule] = rows
            total = sum(int(r.get("count", 0)) for r in rows)
            print(f"    -> {total} total alerts, {len(rows)} time windows")
        except Exception as e:
            print(f"    Error: {e}")

    return alert_data


def _build_dataframe(rows, key_field="EventCode", allowed_keys=None):
    """Build a pandas DataFrame: rows=time windows, columns=(EventCode, source) labels.

    If allowed_keys is provided as a set of (source_lower, eventcode_str), rows are
    filtered to that set so the plot is restricted to top_logtypes.
    """
    import pandas as pd
    records = {}
    for r in rows:
        t = r.get("_time", "")
        key = str(r.get(key_field, "unknown"))
        source = r.get("source", "")
        if allowed_keys is not None:
            if (source.lower(), key) not in allowed_keys:
                continue
        label = f"{key} ({source.split(':')[-1]})" if source else key
        count = int(r.get("count", 0))
        if t not in records:
            records[t] = {}
        records[t][label] = records[t].get(label, 0) + count

    df = pd.DataFrame.from_dict(records, orient="index").fillna(0).sort_index()
    import re
    cleaned = [re.sub(r"\s*[A-Z]{2,5}$", "", str(t)).strip() for t in df.index]
    df.index = pd.to_datetime(cleaned, errors="coerce")
    df = df[df.index.notna()]
    # Sort columns by total count descending
    df = df[df.sum().sort_values(ascending=False).index]
    return df


def _kl_divergence(p, q):
    """Symmetric KL divergence between two distributions."""
    eps = 1e-10
    p = p / (p.sum() + eps) + eps
    q = q / (q.sum() + eps) + eps
    return 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))


def find_similar_splits(df, alert_df=None, min_train_days=60, min_test_days=30, step_days=14):
    """Find train/test date splits where EventCode + alert distributions are most similar.

    Slides a split point across the timeline, computing symmetric KL divergence
    between the train and test EventCode distributions (and alert distributions
    if provided). Returns sorted list of (split_date, kl_eventcode, kl_alert, kl_combined).
    """
    import pandas as pd
    dates = df.index
    start, end = dates.min(), dates.max()
    min_train = pd.Timedelta(days=min_train_days)
    min_test = pd.Timedelta(days=min_test_days)
    step = pd.Timedelta(days=step_days)

    splits = []
    split_date = start + min_train
    while split_date <= end - min_test:
        train = df[df.index < split_date].sum()
        test = df[df.index >= split_date].sum()
        kl_ec = _kl_divergence(train.values, test.values)

        kl_al = 0.0
        if alert_df is not None and len(alert_df) > 0:
            a_train = alert_df[alert_df.index < split_date].sum()
            a_test = alert_df[alert_df.index >= split_date].sum()
            if a_train.sum() > 0 and a_test.sum() > 0:
                kl_al = _kl_divergence(a_train.values, a_test.values)

        kl_combined = kl_ec + kl_al
        splits.append((split_date, kl_ec, kl_al, kl_combined))
        split_date += step

    splits.sort(key=lambda x: x[3])
    return splits


def plot_log_types(rows, output_path, alert_df=None, top_logtypes=None):
    """Plot EventCode distribution over time (restricted to top_logtypes) and train/test similarity."""
    if not rows:
        print("No EventCode data to plot")
        return None

    import pandas as pd
    allowed = None
    if top_logtypes:
        allowed = {(src.lower(), str(ec)) for src, ec in top_logtypes}
    df = _build_dataframe(rows, "EventCode", allowed_keys=allowed)
    print(f"  EventCode matrix: {df.shape[0]} time windows x {df.shape[1]} top_logtype EventCodes")

    # Top 15 EventCodes for plotting
    top_n = 15
    top_cols = df.columns[:top_n].tolist()
    other = df[df.columns[top_n:]].sum(axis=1) if len(df.columns) > top_n else None

    # === Plot 1: EventCode distribution over time ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    plot_df = df[top_cols].copy()
    if other is not None:
        plot_df["Other"] = other
    plot_df.plot.area(ax=ax1, alpha=0.8, linewidth=0.5)
    ax1.set_title(f"EventCode Distribution Over Time — top_logtypes ({df.shape[1]} codes, 2-day windows)", fontsize=14)
    ax1.set_ylabel("Event Count")
    ax1.legend(loc="upper left", fontsize=6, ncol=3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Bar chart of totals
    totals = df.sum().head(top_n + (1 if other is not None else 0))
    if other is not None:
        totals["Other"] = other.sum()
    totals = totals.sort_values(ascending=True)
    totals.plot.barh(ax=ax2, color=plt.cm.tab20(np.linspace(0, 1, len(totals))))
    ax2.set_xlabel("Total Event Count")
    ax2.set_title("Total Events by EventCode", fontsize=14)
    for i, (name, v) in enumerate(totals.items()):
        ax2.text(v + totals.max() * 0.01, i, f"{int(v):,}", va="center", fontsize=8)

    plt.tight_layout()
    out = output_path / "eventcode_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # === Plot 2: Train/test similarity analysis ===
    print("\n=== Train/Test Split Similarity Analysis ===")
    splits = find_similar_splits(df, alert_df, min_train_days=60, min_test_days=30, step_days=7)

    if not splits:
        print("  Not enough data for split analysis")
        return df

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    split_dates = [s[0] for s in splits]
    kl_ec = [s[1] for s in splits]
    kl_al = [s[2] for s in splits]
    kl_comb = [s[3] for s in splits]

    ax1.plot(split_dates, kl_ec, label="EventCode KL", color="blue", linewidth=1.5)
    ax1.plot(split_dates, kl_al, label="Alert KL", color="red", linewidth=1.5)
    ax1.plot(split_dates, kl_comb, label="Combined KL", color="green", linewidth=2)

    # Mark top 5 best splits
    for i, (d, kec, kal, kc) in enumerate(splits[:5]):
        ax1.axvline(x=d, color="green", alpha=0.3, linestyle="--")
        ax1.annotate(f"#{i+1}: {d.strftime('%Y-%m-%d')}\nKL={kc:.4f}",
                     xy=(d, kc), fontsize=7, ha="center",
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    ax1.set_title("Train/Test Split Quality (lower KL = more similar distributions)", fontsize=14)
    ax1.set_ylabel("Symmetric KL Divergence")
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Best split detail: compare distributions
    best_date = splits[0][0]
    train_dist = df[df.index < best_date].sum()
    test_dist = df[df.index >= best_date].sum()
    train_norm = train_dist / train_dist.sum()
    test_norm = test_dist / test_dist.sum()

    compare = pd.DataFrame({"Train": train_norm, "Test": test_norm}).head(top_n)
    compare.plot.bar(ax=ax2, alpha=0.8)
    ax2.set_title(f"Best Split: {best_date.strftime('%Y-%m-%d')} — EventCode Distribution Comparison", fontsize=13)
    ax2.set_ylabel("Proportion")
    ax2.set_xlabel("EventCode")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_path / "train_test_split_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Print top 10 splits
    print("\nTop 10 recommended train/test splits:")
    print(f"{'Rank':<5} {'Split Date':<15} {'KL EventCode':<15} {'KL Alert':<12} {'KL Combined':<12} {'Train Days':<12} {'Test Days':<10}")
    data_start = df.index.min()
    data_end = df.index.max()
    for i, (d, kec, kal, kc) in enumerate(splits[:10]):
        train_days = (d - data_start).days
        test_days = (data_end - d).days
        print(f"{i+1:<5} {d.strftime('%Y-%m-%d'):<15} {kec:<15.4f} {kal:<12.4f} {kc:<12.4f} {train_days:<12} {test_days:<10}")

    return df


def plot_alerts(alert_data, output_path):
    """Plot alert counts per rule over time + totals bar chart. Returns alert DataFrame."""
    import pandas as pd
    if not alert_data:
        print("No alert data to plot")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(alert_data), 1)))

    rule_totals = {}
    for i, (rule, rows) in enumerate(alert_data.items()):
        if not rows:
            continue
        times, counts = [], []
        for r in sorted(rows, key=lambda x: x.get("_time", "")):
            t = r.get("_time", "")
            if t:
                times.append(datetime.fromisoformat(t[:19]))
                counts.append(int(r.get("count", 0)))
        if times:
            label = SHORT_NAMES.get(rule, rule[:60])
            total = sum(counts)
            rule_totals[rule] = total
            ax1.plot(times, counts, marker="o", markersize=3,
                     label=f"{label} ({total:,})",
                     color=colors[i], linewidth=1.5)

    ax1.set_title("Alert Counts per Detection Rule (2-day windows)", fontsize=14)
    ax1.set_ylabel("Alert Count")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    if rule_totals:
        sorted_rules = sorted(rule_totals.items(), key=lambda x: x[1], reverse=True)
        names = [SHORT_NAMES.get(r, r[:60]) for r, _ in sorted_rules]
        totals = [t for _, t in sorted_rules]

        y_pos = range(len(names))
        ax2.barh(y_pos, totals, color=colors[:len(names)])
        ax2.set_yticks(list(y_pos))
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel("Total Alert Count")
        ax2.set_title("Total Alerts by Rule", fontsize=14)
        ax2.invert_yaxis()
        for i, v in enumerate(totals):
            ax2.text(v + max(totals) * 0.01, i, f"{v:,}", va="center", fontsize=9)

    plt.tight_layout()
    out = output_path / "alert_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Build alert DataFrame for split analysis
    records = {}
    for rule, rule_rows in alert_data.items():
        label = SHORT_NAMES.get(rule, rule[:40])
        for r in rule_rows:
            t = r.get("_time", "")
            if t:
                if t not in records:
                    records[t] = {}
                records[t][label] = int(r.get("count", 0))
    if records:
        alert_df = pd.DataFrame.from_dict(records, orient="index").fillna(0).sort_index()
        # Strip timezone abbreviations (e.g. "IDT", "IST") before parsing
        alert_df.index = pd.to_datetime(
            alert_df.index.map(lambda x: x.rsplit(" ", 1)[0] if x[-1].isalpha() else x))
        return alert_df
    return None


def main():
    parser = argparse.ArgumentParser(description="Log type and alert distribution analysis")
    parser.add_argument("--ip", type=int, default=3, help="Splunk host (maps to SPLUNK_HOST_N env var)")
    parser.add_argument("--index", default="new_main")
    parser.add_argument("--span", default="2d", help="Time bucket span")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--skip-logtypes", action="store_true", help="Skip log type analysis")
    parser.add_argument("--skip-alerts", action="store_true", help="Skip alert analysis")
    args = parser.parse_args()

    host = os.environ.get(f"SPLUNK_HOST_{args.ip}")
    if not host:
        print(f"Error: SPLUNK_HOST_{args.ip} not set in .env")
        sys.exit(1)

    from SplunkResearch.src.config import config
    username = config.get("splunk.username", "splunk")
    password = config.get("splunk.password", "")

    output_path = Path(args.output) if args.output else Path(f"SplunkResearch/host_{host}_experiments/baseline")
    output_path.mkdir(parents=True, exist_ok=True)

    svc = connect(host, args.port, username, password)
    print(f"Connected to {host}, index={args.index}, span={args.span}")

    # Run alerts first so we can use the alert DataFrame in the split analysis
    alert_df = None
    if not args.skip_alerts:
        print("\n=== Alert Distribution ===")
        alert_data = query_alerts(svc, args.index, RULE_NAMES, args.span)
        alert_df = plot_alerts(alert_data, output_path)
        with open(output_path / "alert_data.json", "w") as f:
            json.dump(alert_data, f, indent=2)

    if not args.skip_logtypes:
        print("\n=== EventCode Distribution (top_logtypes) ===")
        top_lts = load_top_logtypes()
        print(f"Loaded {len(top_lts)} top_logtypes from {TOP_LOGTYPES_CSV.name}")
        log_rows = query_log_types(svc, args.index, args.span, top_logtypes=top_lts)
        plot_log_types(log_rows, output_path, alert_df=alert_df, top_logtypes=top_lts)
        with open(output_path / "eventcode_data.json", "w") as f:
            json.dump(log_rows, f, indent=2)

    print(f"\nDone! Output in {output_path}")


if __name__ == "__main__":
    main()
