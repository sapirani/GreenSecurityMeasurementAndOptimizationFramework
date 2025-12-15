from dataclasses import asdict
from datetime import timedelta
from threading import Lock
from typing import Optional, List, Union, cast, Dict, Any
import pandas as pd
from river import stats, utils
from DTOs.aggregated_results_dtos.cpu_integral_result import CPUIntegralResult
from DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from DTOs.aggregation_types import AggregationType
from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.drl_model.config.telemetry_fields import field_names_to_average, field_names_to_sum
from hadoop_optimizer.drl_model.consts.general import HOSTNAME_FIELD, SYSTEM_PREFIX
from hadoop_optimizer.drl_model.consts.state_telemetry import DRLTelemetryType
from hadoop_optimizer.river_extensions.custom_agg import CustomAgg
from hadoop_optimizer.river_extensions.time_aware_transformer_union import TimeAwareTransformerUnion
from hadoop_optimizer.river_extensions.timezone_aware_time_rolling import TimezoneAwareTimeRolling


class StateNotReadyException(Exception):
    """Raised when the minimal required telemetry has not yet been retrieved by the DRL"""
    def __init__(self):
        super().__init__("minimal required telemetry is not yet available")


class DRLState:
    """
    This version of DRL state is relatively naive.
    It uses the metrics gained from the entire nodes and processes, and aggregate them all together.
    It means that the agent would not be able to isolate the consumption of each node / process from each other,
    hence, will be limited for taking cluster-level decisions only.
    """

    def __init__(
            self,
            time_windows_seconds: List[int],
            split_by: Union[str, list[str], None] = None,
    ):
        self.split_by = split_by
        self.time_windows = [timedelta(seconds=window_seconds) for window_seconds in time_windows_seconds]
        self.lock = Lock()
        self.known_session_host_identities = []
        self.time_aware_transformer = TimeAwareTransformerUnion()

        self.build_state_transformer()

    @staticmethod
    def __time_aware_aggregator_factory(
            field_to_aggregate: str,
            split_by: Union[str, List[str], None],
            statistic: Union[stats.base.Univariate, utils.Rolling, utils.TimeRolling],
            period: timedelta
    ):
        """ A wrapper function that uses our custom-build time-aware aggregations """
        return CustomAgg(
            on=field_to_aggregate, by=split_by,
            how=TimezoneAwareTimeRolling(statistic, period=period)
        )

    def build_state_transformer(self):
        """
        This function builds all aggregators over all selected time windows
        (based on the configured fields to aggregate, statistics to apply, and selected time windows).
        :return: time-aware transformer that holds all aggregations
        """
        all_aggregators = []
        for time_window in self.time_windows:
            for field_name_to_average in field_names_to_average:
                all_aggregators.append(
                    self.__time_aware_aggregator_factory(
                        field_to_aggregate=field_name_to_average,
                        split_by=self.split_by,
                        statistic=stats.Mean(),
                        period=time_window
                    )
                )

                all_aggregators.append(
                    self.__time_aware_aggregator_factory(
                        field_to_aggregate=field_name_to_average,
                        split_by=self.split_by,
                        statistic=stats.Var(),
                        period=time_window
                    )
                )

            for field_name_to_sum in field_names_to_sum:
                all_aggregators.append(
                    self.__time_aware_aggregator_factory(
                        field_to_aggregate=field_name_to_sum,
                        split_by=self.split_by,
                        statistic=stats.Sum(),
                        period=time_window
                    )
                )

        self.time_aware_transformer = TimeAwareTransformerUnion(*all_aggregators)

    @staticmethod
    def __extract_aggregations(
            iteration_aggregation_results: Optional[IterationAggregatedResults]
    ) -> Dict[str, Any]:
        """
        :return: a dictionary the maps the aggregation name to its value based on the dataclass input
        """
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
        system_raw = {
            f"{SYSTEM_PREFIX}{telemetry_name}": telemetry_val for telemetry_name, telemetry_val in
            asdict(iteration_raw_results.system_raw_results).items()
        }
        aggregations_results = self.__extract_aggregations(iteration_aggregation_results)

        complete_sample = {
            HOSTNAME_FIELD: hostname,
            **system_raw,
            **aggregations_results,
        }

        with self.lock:
            self.time_aware_transformer.learn_one(complete_sample, t=timestamp)

    def __extract_state_telemetry_entries(self) -> pd.DataFrame:
        """
        This turns the raw and aggregated data regarding the load on the system to an embedding space
        that summaries this data.
        This embedding is used as a part of the state space of the DRL.
        """
        # TODO: AVOID NUMERIC ERRORS THAT RESULT IN VERY SMALL NEGATIVE NUMBERS.
        #  MAYBE IT CAN BE COMBINED WITH THE VALUES SCALING ALTOGETHER

        # TODO: SCALE THE TELEMETRY VALUES SOMEHOW (to avoid huge values, such as number of disk reads)
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
        """
        This turns the selected job properties to an embedding space that summaries this data.
        This embedding is used as a part of the state space of the DRL.
        """
        return pd.Series(job_properties.dict())

    def retrieve_state_entries(self, task_properties: JobProperties) -> pd.Series:
        """
        This function returns the entire state, which is consisted of:
        1. a vector representing the load on the system
        2. a vector representing the selected job that should be running in the system soon
        Note: some metrics may result in negative value that is very close to 0, due to numeric errors
        """
        telemetry_entries = self.__extract_state_telemetry_entries().stack()
        job_properties_entries = self.__extract_state_job_entries(task_properties)
        return pd.concat([telemetry_entries, job_properties_entries])
