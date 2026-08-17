"""Live measurement: inject the episode's logs into Splunk and profile the rules.

Faithful port of the injection + measurement flow from wrappers/action.py:214-306
(inject_episodic_logs / collect_configs / flush_logs) and the profiler dispatch in
splunk_tools.py (run_saved_searches). It REUSES the proven SplunkTools and
LogGenerator rather than reimplementing them.

STATUS: untested — requires a reachable Splunk host (SPLUNK_HOST_{ip} in
SplunkResearch/.env). The mock path is the default and is fully tested; run with
live=True only against a real host, and smoke-test one episode first.

Baseline vs current use the SAME profiler over the same window: measure the clean
window, inject, wait for indexing, measure again. Energy = relative CPU increase.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict

import requests

from .measure import Measurement
from .rules import RULE_NAMES, EXPECTED_ALERTS

logger = logging.getLogger(__name__)


class LiveMeasurer:
    def __init__(self, cfg):
        self.cfg = cfg
        # Imported lazily so the mock path never needs the old, Splunk-coupled stack.
        from splunk_tools import SplunkTools
        from log_generator import LogGenerator
        from .rules import load_top_logtypes
        self.splunk = SplunkTools(RULE_NAMES, cfg.rule_frequency_min, ip=cfg.ip)
        # top_logtypes rebuilt here to feed the generator's template space.
        self.top_logtypes = load_top_logtypes(cfg.top_logtypes_csv, max_logtypes=cfg.max_logtypes)
        self.log_generator = LogGenerator(self.top_logtypes)
        logger.warning("LiveMeasurer active — this path is UNTESTED. Smoke-test one "
                       "episode before trusting results.")

    # ------------------------------------------------------------------
    def _episode_window(self, env):
        s = env._episode_start
        import datetime
        e = s + datetime.timedelta(minutes=self.cfg.window_size_min)
        return (s.strftime("%m/%d/%Y:%H:%M:%S"), e.strftime("%m/%d/%Y:%H:%M:%S"))

    def _profile(self, time_range) -> Dict[str, Dict]:
        """Run every saved search over `time_range`; return {rule: {cpu, alert}}."""
        metrics = asyncio.run(self.splunk.run_saved_searches(time_range))
        out = {}
        for m in metrics:
            out.setdefault(m.search_name, {"cpu": 0.0, "alert": 0.0})
            out[m.search_name]["cpu"] += float(m.cpu)
            out[m.search_name]["alert"] += float(m.results_count)
        return out

    def _collect_configs(self, env):
        configs = []
        for logs_to_inject, w_start, w_end in env.episode_plans:
            tr = (w_start.strftime("%m/%d/%Y:%H:%M:%S"), w_end.strftime("%m/%d/%Y:%H:%M:%S"))
            for key, info in logs_to_inject.items():
                if info["count"] == 0:
                    continue
                src, ec, istrig = key.split("_")
                configs.append({"logsource": src, "eventcode": ec, "istrigger": istrig,
                                "time_range": tr, "num_logs": info["count"],
                                "diversity": info["diversity"], "injection_id": 0})
        return configs

    def _inject(self, env):
        configs = self._collect_configs(env)
        if not configs:
            return
        prefix = self.splunk.log_file_prefix
        for batch in self.log_generator.generate_massive_stream(configs, batch_size=50000):
            buckets: Dict[str, list] = {}
            for log_entry, source in batch:
                buckets.setdefault(source, []).append(log_entry)
            for source, logs in buckets.items():
                self.splunk.write_logs_to_monitor(logs, source)
        self._flush(os.path.join(prefix, "wineventlog:security.txt"), "Security")
        self._flush(os.path.join(prefix, "wineventlog:system.txt"), "System")

    def _flush(self, path, log_type):
        if not os.path.exists(path):
            return
        port = getattr(self.cfg, "splunk_port", 8089)
        endpoint = f"https://{self.splunk.splunk_host}:{port}/services/receivers/stream"
        params = {"index": self.splunk.index_name,
                  "sourcetype": f"WinEventLog:{log_type}",
                  "host": getattr(self.cfg, "secondary_host", self.splunk.splunk_host)}
        with open(path, "rb") as fh:
            resp = requests.post(endpoint, auth=(self.splunk.splunk_username, self.splunk.splunk_password),
                                 params=params, headers={"x-splunk-input-mode": "streaming"},
                                 data=fh, verify=False, timeout=60)
        resp.raise_for_status()
        # Truncate on success only: the file is appended to across episodes and is
        # never otherwise cleared, so a failed POST leaves data to retry next flush
        # while a successful one must not re-send the same bytes again.
        open(path, "w").close()

    # ------------------------------------------------------------------
    def measure(self, env) -> Measurement:
        time_range = self._episode_window(env)

        baseline = self._profile(time_range)
        self._inject(env)
        import time
        time.sleep(5)  # allow indexing; the original polls a count query — see action.py:243
        current = self._profile(time_range)

        baseline_alert = {r: baseline.get(r, {}).get("alert", EXPECTED_ALERTS[r]) for r in RULE_NAMES}
        current_alert = {r: current.get(r, {}).get("alert", baseline_alert[r]) for r in RULE_NAMES}
        baseline_cpu = sum(baseline.get(r, {}).get("cpu", 0.0) for r in RULE_NAMES)
        current_cpu = sum(current.get(r, {}).get("cpu", 0.0) for r in RULE_NAMES)
        return Measurement(baseline_cpu, current_cpu, baseline_alert, current_alert)

    def close(self):
        """Delete injected fake logs to prevent bleed into the next run."""
        try:
            self.splunk.delete_fake_logs()
        except Exception as e:  # noqa: BLE001
            logger.warning("clean-up failed: %s", e)
