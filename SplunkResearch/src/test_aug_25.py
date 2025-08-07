import splunklib.client as client
import splunklib.results as results
import psutil
import threading
import time
import json
import csv
import logging
from datetime import datetime
import pytz

class SplunkRuleRunner:
    def __init__(self, host, port, username, password, scheme='https'):
        self.service = client.connect(
            host=host,
            port=port,
            username=username,
            password=password,
            scheme=scheme
        )
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger("SplunkRuleRunner")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("splunk_rule_runner.log")
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def run_rules(self, rule_names, earliest_time=None, latest_time=None, export_csv=None, parallel=True):
        results = []
        threads = []

        def run_rule(rule):
            try:
                saved_search = self.service.saved_searches[rule]
                was_disabled = saved_search["disabled"] != "0"

                # Enable if originally disabled
                if was_disabled:
                    self.logger.info(f"Temporarily enabling rule: {rule}")
                    saved_search.update(disabled=False)
                    saved_search.refresh()
                dispatch_args = {}
                if earliest_time: dispatch_args["dispatch.earliest_time"] = earliest_time
                if latest_time: dispatch_args["dispatch.latest_time"] = latest_time

                self.logger.info(f"Dispatching rule: {rule}")
                cpu_before = psutil.cpu_times()._asdict()

                job = saved_search.dispatch(**dispatch_args)

                while not job.is_done():
                    time.sleep(.5)
                cpu_after = psutil.cpu_times()._asdict()

                job.refresh()

                metric = {
                    "rule_name": rule,
                    "sid": job.sid,
                    "run_time_sec": job["runDuration"],
                    # "cpu_time_sec": job["performance.systemTotalCPUSeconds"],
                    "event_count": job["eventCount"],
                    "result_count": job["resultCount"],
                    "scan_count": job["scanCount"],
                    "timestamp": datetime.now().isoformat(),
                    "cpu_user_time": cpu_after["user"] - cpu_before["user"],
                    "cpu_system_time": cpu_after["system"] - cpu_before["system"]
                }

                self.logger.info(f"Completed rule: {rule} — {metric}")
                results.append(metric)
            except Exception as e:
                self.logger.error(f"Error running rule '{rule}': {str(e)}")
            finally:
                # Re-disable rule if it was originally disabled
                try:
                    if was_disabled:
                        self.logger.info(f"Re-disabling rule: {rule}")
                        saved_search.update(disabled=True)
                        saved_search.refresh()
                except Exception as e:
                    self.logger.error(f"Failed to restore disabled state for rule '{rule}': {str(e)}")
            

        if parallel:
            for rule in rule_names:
                t = threading.Thread(target=run_rule, args=(rule,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
        else:
            for rule in rule_names:
                run_rule(rule)

        if export_csv and results:
            self.export_metrics_csv(results, export_csv)

        return results

    def export_metrics_csv(self, metrics, filename):
        keys = metrics[0].keys()
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics)
        self.logger.info(f"Exported metrics to {filename}")

# ==== EXAMPLE USAGE ====
if __name__ == "__main__":
    runner = SplunkRuleRunner(
        host="localhost",
        port=8089,
        username="shouei",
        password="123456789",
        scheme="https"
    )

    rules = [
        "Detect New Local Admin Account",
        "Known Services Killed by Ransomware",
        "Windows Event For Service Disabled",
        "Kerberoasting spn request with RC4 encryption",
        "Clop Ransomware Known Service Name",
        "Non Chrome Process Accessing Chrome Default Dir",
        'ESCU Network Share Discovery Via Dir Command Rule',
        'Windows AD Replication Request Initiated from Unsanctioned Location',
        'ESCU Windows Rapid Authentication On Multiple Hosts Rule'
    ]


    tz = pytz.timezone("Asia/Jerusalem")  # or your timezone
    earliest = tz.localize(datetime(2024, 6, 19, 20, 0, 0)).isoformat()
    latest = tz.localize(datetime(2024, 6, 21, 20, 0, 0)).isoformat()

    metrics = runner.run_rules(
        rule_names=rules,
        earliest_time=earliest,
        latest_time=latest,
        export_csv="splunk_rule_metrics.csv",
        parallel=True
    )

    print(json.dumps(metrics, indent=2))
