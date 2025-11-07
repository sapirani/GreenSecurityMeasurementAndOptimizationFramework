from dataclasses import asdict
from datetime import timedelta, datetime, timezone
from threading import Lock
from typing import Optional, List, Union

import pandas as pd
from dateutil.tz import UTC
from river.compose import TransformerUnion
from river import utils, stats
from river.feature_extraction import Agg
from river.utils.rolling import Rollable

from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults

HOSTNAME_FIELD = "hostname"
ALL_HOSTS = "all_hosts"


class TimezoneAwareTimeRolling(utils.TimeRolling):
    def __init__(self, obj: Rollable, period: timedelta):
        super().__init__(obj, period)
        self._latest = datetime(1, 1, 1, tzinfo=UTC)

    def flush_expired(self):
        """
        This function is mostly copied from TimeRolling update function.
        The original implementation is updated only when a new element is inserted.
        This function enables the removal of old events, even though no new elements are inserted.
        """
        # Update latest time marker
        self._latest = datetime.now(timezone.utc)

        # Find all expired points (older than now - period)
        i = 0
        for ti, (argsi, kwargsi) in zip(self._timestamps, self._datum):
            if ti > self._latest - self.period:
                break
            self.obj.revert(*argsi, **kwargsi)
            i += 1

        # Remove expired events
        if i > 0:
            self._timestamps = self._timestamps[i:]
            self._datum = self._datum[i:]


# TODO: CONSIDER ADDING THE WINDOW SIZE AS A PART OF THE INDEX / NAME
class CustomAgg(Agg):
    @property
    def state(self) -> pd.Series:
        for time_rolling in self._groups.values():
            time_rolling.flush_expired()    # Assuming our TimezoneAwareTimeRolling is used

        if not self.by:
            return pd.Series(
                (stat.get() for stat in self._groups.values()),
                index=pd.Index([ALL_HOSTS]),
                name=self._feature_name,
            )
        return super().state


class TimeAwareTransformerUnion(TransformerUnion):
    def learn_one(self, x, t=None):
        for transformer in self.transformers.values():
            transformer.learn_one(x, t)


class DRLState:
    """
    This version of DRL state is relatively naive.
    It uses the metrics gained from the entire nodes and processes, and aggregate them all together.
    It means that the agent would not be able to isolate the consumption of each node / process from each other,
    hence, will be limited for taking cluster-level decisions only.
    """

    # TODO: CONSIDER USING WEIGHTS INSIDE TIME WINDOWS (SO NEWER RESULTS WILL BE MORE SIGNIFICANT)
    def __init__(
            self,
            time_windows_seconds: List[int],
            split_by: Union[str, list[str], None] = None,
    ):
        self.split_by = split_by
        self.time_windows = [timedelta(seconds=window_seconds) for window_seconds in time_windows_seconds]
        self.lock = Lock()

        # TODO: INCORPORATE AGGREGATIONS SUCH AS CPU INTEGRAL AND ENERGY CONSUMPTION
        field_names_to_average = ["cpu_percent_sum_across_cores", "total_memory_percent"]
        field_names_to_sum = [
            "disk_read_count", "disk_write_count",
            "disk_read_kb", "disk_write_kb",
            "disk_read_time", "disk_write_time",
            "packets_sent", "packets_received",
            "network_kb_sent", "network_kb_received"
        ]

        self.time_aware_transformer = TimeAwareTransformerUnion()
        self.build_state_transformer(field_names_to_average, field_names_to_sum)

    def build_state_transformer(self, field_names_to_average: List[str], field_names_to_sum: List[str]):
        all_aggregators = []
        for time_window in self.time_windows:
            for field_name_to_average in field_names_to_average:
                all_aggregators.append(
                    CustomAgg(
                        on=field_name_to_average, by=self.split_by,
                        how=TimezoneAwareTimeRolling(stats.Mean(), period=time_window)
                    )
                )

                all_aggregators.append(
                    CustomAgg(
                        on=field_name_to_average, by=self.split_by,
                        how=TimezoneAwareTimeRolling(stats.Var(), period=time_window)
                    )
                )

            for field_name_to_sum in field_names_to_sum:
                all_aggregators.append(
                    CustomAgg(
                        on=field_name_to_sum, by=self.split_by,
                        how=TimezoneAwareTimeRolling(stats.Sum(), period=time_window)
                    )
                )

        self.time_aware_transformer = TimeAwareTransformerUnion(*all_aggregators)

    def update_state(
            self,
            iteration_raw_results: IterationRawResults,
            iteration_aggregation_results: Optional[IterationAggregatedResults]
    ):
        """
        This function considers only the system iteration raw results, and preserve an state the represents
        the load on the entire cluster (for now, it is not being split by the cluster's hosts)

        Simplified assumptions for now:
        1. We are not considering the variety in hosts
        2. We are not considering process data (except for energy consumption that should be estimated at the host
        level later on, not at the process level)
        3. The representation considers the cluster load as a whole, without dividing consumption per each node
        4. The average of CPU consumption implicitly assumes that all nodes contain the same amount of CPU cores
        (we may incorporate the number of CPU cores in the future)
        5. Same for the memory usage
        6. The metrics are being summed and there is no upper bound. The DRL might learn better when all metrics
        are represented as low numbers between 0 and 1.
        """

        timestamp = iteration_raw_results.metadata.timestamp
        hostname = iteration_raw_results.metadata.session_host_identity.hostname

        system_raw = asdict(iteration_raw_results.system_raw_results)
        system_raw[HOSTNAME_FIELD] = hostname

        with self.lock:
            self.time_aware_transformer.learn_one(system_raw, t=timestamp)

    @property
    def state(self) -> pd.Series:
        """
        Note: some metrics may result in negative value that is very close to 0, due to numeric errors
        """
        with self.lock:
            return pd.concat(
                [transformer.state for transformer in self.time_aware_transformer.transformers.values()],
                axis=1
            )
