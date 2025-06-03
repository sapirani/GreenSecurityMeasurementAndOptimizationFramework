import re
import numpy as np
import pandas as pd
from datetime import datetime

def parse_log_datetime(log_line):
    if not isinstance(log_line, str):
        return None
    match = re.search(r"Start time:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", log_line)
    if match:
        dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp()
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+", log_line)
    if match:
        dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    return None



import re

def parse_iterations(log_path):
    iterations = []
    current_start_line = None
    current_end_line = None
    rule = quantity = diversity = None

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()

            if 'Start time:' in line:
                # Close previous iteration
                if current_start_line:
                    iterations.append({
                        "start": current_start_line,
                        "end": current_end_line or "EOF",
                        "rule": rule,
                        "quantity": quantity,
                        "diversity": diversity
                    })

                current_start_line = line
                current_end_line = None
                rule = quantity = diversity = None

            elif 'Generating logs for rule:' in line and current_start_line:
                match = re.search(
                    r"rule:\s*(.*?),\s*quantity:\s*(\d+),\s*diversity:\s*(\d+)", line)
                if match:
                    rule = match.group(1)
                    quantity = int(match.group(2))
                    diversity = int(match.group(3))

            elif 'Cleaning environment' in line:
                if current_start_line:
                    current_end_line = line
                    iterations.append({
                        "start": current_start_line,
                        "end": current_end_line,
                        "rule": rule,
                        "quantity": quantity,
                        "diversity": diversity
                    })
                    current_start_line = None
                    current_end_line = None
                    rule = quantity = diversity = None

    # Final open iteration
    if current_start_line:
        iterations.append({
            "start": current_start_line,
            "end": "EOF",
            "rule": rule,
            "quantity": quantity,
            "diversity": diversity
        })

    return iterations



def aggregate_cpu_usage(iterations, cpu_csv_path):
    cpu_df = pd.read_csv(cpu_csv_path)
    cpu_df.columns = [col.strip().lower().replace(" ", "_") for col in cpu_df.columns]
    cpu_df['timestamp'] = cpu_df['timestamp_per_10_seconds']
    # Normalize timestamp units if needed (e.g., milliseconds to seconds)
    if cpu_df["timestamp"].max() > 1e12:
        cpu_df["timestamp"] = cpu_df["timestamp"] / 1000.0

    results = []
    for it in iterations:
        start_ts = parse_log_datetime(it["start"])
        end_ts = parse_log_datetime(it["end"]) if it["end"] != "EOF" else float("inf")

        if start_ts is None or end_ts is None:
            continue

        # Filter the relevant cpu samples
        mask = (cpu_df["timestamp"] >= start_ts) & (cpu_df["timestamp"] <= end_ts)
        subset = cpu_df[mask]

        if not subset.empty:
            cpu_integral = np.trapz(subset["average_of_cpu_percent"].values, x=subset["timestamp"].values)
        else:
            cpu_integral = 0.0

        results.append({
            "start_time": it["start"],
            "end_time": it["end"],
            "rule": it["rule"],
            "quantity": it["quantity"],
            "diversity": it["diversity"],
            "cpu_integral": cpu_integral
        })

    return pd.DataFrame(results)


def main():
    log_path = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/energy_profile_final.log"     # ⬅️ Replace with your real log file
    cpu_csv_path = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/Inspector (2).csv"      # ⬅️ Replace with your real CPU CSV
    output_csv = "aggregated_output.csv"

    print("[*] Parsing iterations from log...")
    iterations = parse_iterations(log_path)

    print(f"[*] Found {len(iterations)} iterations.")
    print("[*] Aggregating CPU usage...")
    result_df = aggregate_cpu_usage(iterations, cpu_csv_path)

    print(f"[*] Writing results to {output_csv}")
    result_df.to_csv(output_csv, index=False)
    print("[✓] Done.")


if __name__ == "__main__":
    main()
