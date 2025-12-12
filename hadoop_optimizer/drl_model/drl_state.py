from dataclasses import asdict
from datetime import timedelta, datetime, timezone
from threading import Lock
from typing import Optional, List, Union, cast, Dict, Any

import pandas as pd
from dateutil.tz import UTC
from river.compose import TransformerUnion
from river import utils, stats
from river.feature_extraction import Agg
from river.utils.rolling import Rollable

from DTOs.aggregated_results_dtos.cpu_integral_result import CPUIntegralResult
from DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from DTOs.aggregation_types import AggregationType
from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.drl_model.consts.state_telemetry import DRLTelemetryType

HOSTNAME_FIELD = "hostname"
ALL_HOSTS = "all_hosts"


class StateNotReadyException(Exception):
    """Raised when the minimal required telemetry has not yet been retrieved by the DRL"""
    def __init__(self):
        super().__init__("minimal required telemetry is not yet available")


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
    def __init__(self, on: str, by: str | list[str] | None,
                 how: stats.base.Univariate | utils.Rolling | utils.TimeRolling):

        super().__init__(on, by, how)
        if not isinstance(how, TimezoneAwareTimeRolling):
            raise ValueError("CustomAgg supports only TimezoneAwareTimeRolling as the 'how' argument")

        self._feature_name += f"_window_{int(how.period.total_seconds())}_seconds"

    @property
    def state(self) -> pd.Series:
        for time_rolling in self._groups.values():
            time_rolling.flush_expired()

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
        self.known_session_host_identities = []

        # TODO: use enum + add "system" to all of the metrics
        # TODO: SCALE THE TELEMETRY VALUES SOMEHOW
        field_names_to_average = ["total_memory_percent"]
        field_names_to_sum = [
            "disk_read_count", "disk_write_count",
            "disk_read_kb", "disk_write_kb",
            "disk_read_time", "disk_write_time",
            "packets_sent", "packets_received",
            "network_kb_sent", "network_kb_received",

            # Aggregations
            DRLTelemetryType.SYSTEM_CPU_INTEGRAL,
            DRLTelemetryType.SYSTEM_TOTAL_ENERGY_MWH,
            DRLTelemetryType.SYSTEM_CPU_ENERGY_MWH,
            DRLTelemetryType.SYSTEM_RAM_ENERGY_MWH,
            DRLTelemetryType.SYSTEM_DISK_READ_ENERGY_MWH,
            DRLTelemetryType.SYSTEM_DISK_WRITE_ENERGY_MWH,
            DRLTelemetryType.SYSTEM_NETWORK_RECEIVED_ENERGY_MWH,
            DRLTelemetryType.SYSTEM_NETWORK_SENT_ENERGY_MWH
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

    @staticmethod
    def __extract_aggregations(
            iteration_aggregation_results: Optional[IterationAggregatedResults]
    ) -> Dict[str, Any]:
        system_cpu_integral = cast(
            CPUIntegralResult,
            iteration_aggregation_results.system_results[AggregationType.CPUIntegral]
        ).cpu_integral

        energy_model_result = cast(
            EnergyModelResult,
            iteration_aggregation_results.system_results[AggregationType.SystemEnergyModelAggregator]
        )

        return {
            DRLTelemetryType.SYSTEM_CPU_INTEGRAL: system_cpu_integral,
            DRLTelemetryType.SYSTEM_TOTAL_ENERGY_MWH: energy_model_result.energy_mwh,
            DRLTelemetryType.SYSTEM_CPU_ENERGY_MWH: energy_model_result.cpu_energy_consumption,
            DRLTelemetryType.SYSTEM_RAM_ENERGY_MWH: energy_model_result.ram_energy_consumption,
            DRLTelemetryType.SYSTEM_DISK_READ_ENERGY_MWH: energy_model_result.disk_io_read_energy_consumption,
            DRLTelemetryType.SYSTEM_DISK_WRITE_ENERGY_MWH: energy_model_result.disk_io_write_energy_consumption,
            DRLTelemetryType.SYSTEM_NETWORK_RECEIVED_ENERGY_MWH: energy_model_result.network_io_received_energy_consumption,
            DRLTelemetryType.SYSTEM_NETWORK_SENT_ENERGY_MWH: energy_model_result.network_io_sent_energy_consumption,
        }

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
        if iteration_raw_results.metadata != iteration_aggregation_results.iteration_metadata:
            raise ValueError("Received inconsistent metadata between raw results and aggregations")

        # ignore first iteration as the aggregations need more than one sample
        if iteration_raw_results.metadata.session_host_identity not in self.known_session_host_identities:
            # TODO: think of when we can remove old session-host identities
            self.known_session_host_identities.append(iteration_raw_results.metadata.session_host_identity)
            return

        timestamp = iteration_raw_results.metadata.timestamp
        hostname = iteration_raw_results.metadata.session_host_identity.hostname

        # TODO: leverage process telemetry somehow
        system_raw = asdict(iteration_raw_results.system_raw_results)
        aggregations_results = self.__extract_aggregations(iteration_aggregation_results)

        complete_sample = {
            HOSTNAME_FIELD: hostname,
            **system_raw,
            **aggregations_results,
        }

        with self.lock:
            self.time_aware_transformer.learn_one(complete_sample, t=timestamp)

    # TODO: ENSURE THAT WHEN WE SPLIT BY HOSTNAME AND RECEIVE DATA FROM 2 HOSTS, EVERYTHING WORKS AS EXPECTED
    def __extract_state_telemetry_entries(self) -> pd.DataFrame:
        # TODO: AVOID NUMERIC ERRORS THAT RESULT IN VERY SMALL NEGATIVE NUMBERS.
        #  MAYBE IT CAN BE COMBINED WITH THE VALUES SCALING ALTOGETHER
        with self.lock:
            telemetry_entries = pd.concat(
                [transformer.state for transformer in self.time_aware_transformer.transformers.values()],
                axis=1
            )

            if telemetry_entries.empty:
                raise StateNotReadyException()

            return telemetry_entries

    @staticmethod
    def __extract_state_job_entries(job_properties: JobProperties) -> pd.Series:
        return pd.Series(job_properties.dict())

    def retrieve_state_entries(self, task_properties: JobProperties) -> pd.Series:
        """
        Note: some metrics may result in negative value that is very close to 0, due to numeric errors
        """
        telemetry_entries = self.__extract_state_telemetry_entries().stack()
        job_properties_entries = self.__extract_state_job_entries(task_properties)
        return pd.concat([telemetry_entries, job_properties_entries])
