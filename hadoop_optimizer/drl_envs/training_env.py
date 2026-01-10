from typing import Any, Optional
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_properties import JobProperties
from DTOs.hadoop.job_types import JobType
from hadoop_optimizer.drl_envs.abstract_hadoop_optimizer_env import AbstractOptimizerEnvInterface
from hadoop_optimizer.drl_telemetry.telemetry_manager import DRLTelemetryManager
from hadoop_optimizer.training_client.client import HadoopOptimizerTrainingClient


class OptimizerTrainingEnv(AbstractOptimizerEnvInterface):
    def __init__(self, telemetry_manager: DRLTelemetryManager, training_client: HadoopOptimizerTrainingClient):
        super().__init__(telemetry_manager)
        self.training_client = training_client

        self.__episodic_job_descriptor: Optional[JobDescriptor] = None

        self.__episodic_baseline_running_time: Optional[float] = None
        self.__episodic_baseline_energy_consumption_mwh: Optional[float] = None

    def _init_episodic_job(self, options: dict[str, Any] | None) -> JobProperties:
        if options:
            raise ValueError("Options are not expected in training mode")

        selected_job_type = self.__select_episodic_job_type(self.np_random_seed)
        selected_input_size_gb = self.__select_input_size_gb(selected_job_type, self.np_random_seed)
        self.__episodic_job_descriptor = JobDescriptor(job_type=selected_job_type, input_size_gb=selected_input_size_gb)

        default_execution_configuration = HadoopJobExecutionConfig()

        result = self.training_client.run_job(
            job_descriptor=self.__episodic_job_descriptor,
            execution_configuration=default_execution_configuration,
        )

        self.__episodic_baseline_running_time = result.runtime_sec
        self.__episodic_baseline_energy_consumption_mwh = self.__compute_baseline_energy_consumption()

        return self.__extract_job_properties(self.__episodic_job_descriptor)

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

    # TODO: CONSIDER TRANSFERRING THESE FUNCTIONS FROM NOW ON INTO A SEPARATE CLASS DEDICATED FOR TRANSLATIONS
        # BETWEEN JOB TYPES TO PROPERTIES + DEFINITION, AND KNOWING THE AVAILABLE INPUT SIZES FOR EACH JOB
    @staticmethod
    def __select_episodic_job_type(np_random_seed: int) -> JobType:
        # TODO: SELECT IT RANDOMLY / WITH SOME SMART LOGIC BASED ON THE TRAINING PROGRESSION
        return JobType.word_count

    @staticmethod
    def __select_input_size_gb(selected_job_type: JobType, np_random_seed: int) -> float:
        # TODO: SELECT BASED ON THE AVAILABLE INPUT SIZES FOR EACH JOB
        return 10

    @staticmethod
    def __extract_job_properties(job_descriptor: JobDescriptor):
        # TODO: LEVERAGE THE JOB TYPE TO DEFINE THE CPU AND IO SCALE
        return JobProperties(
            input_size_gb=job_descriptor.input_size_gb,
            cpu_bound_scale=0.5,
            io_bound_scale=0.5,
        )
