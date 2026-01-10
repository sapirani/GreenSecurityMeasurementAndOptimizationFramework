from typing import Any, Optional
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_properties import JobProperties
from DTOs.hadoop.job_types import JobType
from hadoop_optimizer.drl_envs.abstract_hadoop_optimizer_env import AbstractOptimizerEnvInterface
from hadoop_optimizer.drl_telemetry.telemetry_manager import DRLTelemetryManager
from hadoop_optimizer.supported_jobs.supported_jobs_config import SupportedJobsConfig
from hadoop_optimizer.training_client.client import HadoopOptimizerTrainingClient
import numpy as np


class OptimizerTrainingEnv(AbstractOptimizerEnvInterface):
    def __init__(self, telemetry_manager: DRLTelemetryManager, training_client: HadoopOptimizerTrainingClient):
        super().__init__(telemetry_manager)
        self.training_client = training_client

        self.__episodic_job_descriptor: Optional[JobDescriptor] = None

        self.__episodic_baseline_running_time: Optional[float] = None
        self.__episodic_baseline_energy_consumption_mwh: Optional[float] = None

    def _custom_rendering(self):
        print("Episodic Job Type:", self.__episodic_job_descriptor.job_type.value)

    def _init_episodic_job(self, options: dict[str, Any] | None) -> JobProperties:
        if options:
            raise ValueError("Options are not expected in training mode")

        selected_job_type = self.__select_episodic_job_type(self.np_random)
        selected_input_size_gb = self.__select_input_size_gb(selected_job_type, self.np_random)
        self.__episodic_job_descriptor = JobDescriptor(job_type=selected_job_type, input_size_gb=selected_input_size_gb)

        default_execution_configuration = HadoopJobExecutionConfig()

        result = self.training_client.run_job(
            job_descriptor=self.__episodic_job_descriptor,
            execution_configuration=default_execution_configuration,
        )

        self.__episodic_baseline_running_time = result.runtime_sec
        self.__episodic_baseline_energy_consumption_mwh = self.__compute_baseline_energy_consumption()

        return SupportedJobsConfig.extract_job_properties(self.__episodic_job_descriptor)

    def __compute_baseline_energy_consumption(self) -> float:
        """
        The baseline job performance is not required in deployment, as are not calculating rewards
        """
        # TODO: IMPLEMENT, use self.telemetry_manager somehow
        return 0

    def _compute_reward(self, job_config: HadoopJobExecutionConfig, terminated: bool, truncated: bool) -> float:
        # todo: implement (run job using the client, measure performance and compute the reward)
        # todo: consider creating another class for the reward computations as defined in the slides

        result = self.training_client.run_job(
            job_descriptor=self.__episodic_job_descriptor,
            execution_configuration=job_config,
        )
        return (result.runtime_sec - self.__episodic_baseline_running_time) / self.__episodic_baseline_running_time

    @staticmethod
    def __select_episodic_job_type(np_random: np.random.Generator) -> JobType:
        # TODO: SELECT EPISODIC JOB BASED ON A SMART LOGIC TAILORED TO THE TRAINING PROGRESSION
        supported_jobs = SupportedJobsConfig.get_all_jobs()
        selected_job_index = np_random.integers(0, len(supported_jobs), dtype=int)
        return supported_jobs[selected_job_index]

    @staticmethod
    def __select_input_size_gb(selected_job_type: JobType, np_random: np.random.Generator) -> float:
        # TODO: SELECT EPISODIC INPUT SIZE BASED ON A SMART LOGIC TAILORED TO THE TRAINING PROGRESSION
        supported_input_size_gb = SupportedJobsConfig.get_supported_input_size_gb(selected_job_type)
        selected_input_size_index = np_random.integers(0, len(supported_input_size_gb), dtype=int)
        return supported_input_size_gb[selected_input_size_index]
