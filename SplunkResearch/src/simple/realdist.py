"""Real log-type distribution per time window, from a static CSV.

Replaces splunk_tools.load_real_logs_distribution_bucket + get_real_distribution
(splunk_tools.py:352-797) for the mock path: no Splunk query, just a time slice
of resources/all_dist_by_host.csv aggregated by (source, eventcode).
"""
from __future__ import annotations

import datetime
import logging
from typing import Dict, List

import pandas as pd

from .rules import LogType

logger = logging.getLogger(__name__)


class RealDistributionProvider:
    """Loads the host distribution CSV once, serves per-window log-type counts."""

    def __init__(self, csv_path: str):
        logger.info("Loading real log distribution from %s", csv_path)
        df = pd.read_csv(csv_path)
        df["_time"] = pd.to_datetime(df["_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        df = df.dropna(subset=["_time"])
        df = df[df["source"].str.contains("Security|System", case=False, regex=True)]
        df["source"] = df["source"].str.lower()
        df["EventCode"] = df["EventCode"].astype(str)
        df["count"] = df["count"].astype("int64")
        self._df = df.set_index("_time").sort_index()
        logger.info("Loaded %d distribution rows spanning %s..%s",
                    len(self._df), self._df.index.min(), self._df.index.max())

    def window_counts(self, start_dt: datetime.datetime, end_dt: datetime.datetime,
                      top_logtypes: List[LogType]) -> Dict[LogType, int]:
        """Aggregated real counts for each top log type within [start, end]."""
        sl = self._df.loc[pd.Timestamp(start_dt):pd.Timestamp(end_dt)]
        counts = {lt: 0 for lt in top_logtypes}
        if sl.empty:
            return counts
        agg = sl.groupby(["source", "EventCode"])["count"].sum()
        wanted = set(top_logtypes)
        for (src, ec), c in agg.items():
            lt = (src, ec)
            if lt in wanted:
                counts[lt] = int(c)
        return counts
