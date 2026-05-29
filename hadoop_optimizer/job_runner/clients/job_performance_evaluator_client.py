from datetime import datetime
from typing import Optional
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_execution_performance import JobExecutionPerformance
from DTOs.logging.consts import IndexName
from common.drl_telemetry.energy_tracker import EnergyTracker
from elastic_reader.consts import AggregationStrategy
from elastic_consumers.elastic_aggregations_logger import ElasticAggregationsLogger
from hadoop_optimizer.drl_envs.training.training_env import EpisodeContext
from hadoop_optimizer.job_runner.clients.job_runner_client import HadoopJobRunnerClient
from elastic_reader.elastic_reader_service import ElasticReaderService
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput, ReadingMode


class HadoopJobPerformanceEvaluatorClient:
    def __init__(
            self,
            hadoop_job_runner_client: Optional[HadoopJobRunnerClient] = None,
    ):
        self.hadoop_job_runner_client = hadoop_job_runner_client or HadoopJobRunnerClient()

        self.energy_tracker = EnergyTracker()

        time_picker_input = TimePickerChosenInput(
                start=datetime.now(tz=datetime.now().astimezone().tzinfo),
                end=None,
                mode=ReadingMode.REALTIME
            )

        indices_to_read_from = [IndexName.PROCESS_METRICS, IndexName.SYSTEM_METRICS]

        elastic_aggregations_logger = ElasticAggregationsLogger(
            reading_mode=ReadingMode.REALTIME,
            log_extra_fields=set(EpisodeContext.model_fields.keys())
        )

        self.elastic_reader_service = ElasticReaderService(
            consumers=[self.energy_tracker, elastic_aggregations_logger],
            aggregation_strategy=AggregationStrategy.CALCULATE,
            time_picker_input=time_picker_input,
            indices_to_read_from=indices_to_read_from,
        )

    def __enter__(self):
        self.start()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.elastic_reader_service.start_in_background()

    def stop(self):
        self.elastic_reader_service.stop()

    def run_job(
        self,
        job_descriptor: JobDescriptor,
        execution_configuration: HadoopJobExecutionConfig,
        session_id: str,
        episode_context: EpisodeContext
    ) -> JobExecutionPerformance:
        """
        :raises:
            1.  requests.exceptions.HTTPError: 422 not implemented (typically when selected job execution config is invalid)
            2.  requests.exceptions.HTTPError: 500 internal server error (typically when Hadoop job can't run for some reason)
            3.  requests.exceptions.HTTPError: 501 not implemented (typically when Hadoop is not installed)
            4.  requests.exceptions.HTTPError: 503 gateway timeout (when job execution has passed time limit)
            5.  RuntimeError: in case that the caller did not call start beforehand / use the contextmanager
        """

        if not self.elastic_reader_service.is_running():
            raise RuntimeError("The Elastic Reader thread has not been started yet.")

        self.energy_tracker.reset_tracker(session_id, episode_context)
        result = self.hadoop_job_runner_client.run_job(
            job_descriptor=job_descriptor,
            execution_configuration=execution_configuration,
            session_id=session_id,
            episode_context=episode_context,
        )

        # TODO: SHOULD WE USE PER-HOST ENERGY CONSUMPTION HERE?
        energy_consumption = sum(self.energy_tracker.get_energy_consumption().values())

        return JobExecutionPerformance(
            running_time_sec=result.runtime_sec,
            energy_use_mwh=energy_consumption
        )
