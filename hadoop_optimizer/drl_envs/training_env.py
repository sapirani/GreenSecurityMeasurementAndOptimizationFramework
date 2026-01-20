from dataclasses import asdict
from typing import Any, Optional
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_execution_performance import JobExecutionPerformance
from DTOs.hadoop.job_properties import JobProperties
from DTOs.hadoop.job_types import JobType
from DTOs.hadoop.training_metadata import TrainingMetadata
from DTOs.hadoop.training_step_results import TrainingStepResults
from hadoop_optimizer.drl_envs.abstract_hadoop_optimizer_env import AbstractOptimizerEnvInterface
from hadoop_optimizer.drl_telemetry.energy_tracker import EnergyTracker
from hadoop_optimizer.drl_telemetry.telemetry_aggregator import TelemetryAggregator
from hadoop_optimizer.reward.reward_calculator import RewardCalculator
from hadoop_optimizer.supported_jobs.supported_jobs_config import SupportedJobsConfig
from hadoop_optimizer.training_client.client import HadoopOptimizerTrainingClient
import numpy as np
from logging import Logger


class OptimizerTrainingEnv(AbstractOptimizerEnvInterface):
    def __init__(
            self,
            telemetry_aggregator: TelemetryAggregator,
            training_client: HadoopOptimizerTrainingClient,
            energy_tracker: EnergyTracker,
            reward_calculator: RewardCalculator,
            train_id: str,
            training_results_logger: Logger
    ):
        super().__init__(telemetry_aggregator)
        self.training_client = training_client
        self.energy_tracker = energy_tracker
        self.reward_calculator = reward_calculator
        self.train_id = train_id
        self.training_results_logger = training_results_logger

        self.__episodic_job_descriptor: Optional[JobDescriptor] = None

    def _custom_rendering(self):
        print("Episodic Baseline Performance:", self.reward_calculator.baseline_performance)
        print("Current Job Performance:", self.reward_calculator.last_job_performance)
        print("Total Reward:", self.reward_calculator.last_reward)
        print("Episodic Job Type:", self.__episodic_job_descriptor.job_type.value)

    def __run_job_and_measure_performance(
            self,
            job_config: HadoopJobExecutionConfig,
            *,
            is_baseline: bool = False
    ) -> JobExecutionPerformance:

        self.energy_tracker.reset_tracker(self.train_id)
        training_metadata = TrainingMetadata(
            episode_num=self.episode_counter,
            step_num=self.step_count,
            is_baseline=is_baseline
        )
        result = self.training_client.run_job(
            job_descriptor=self.__episodic_job_descriptor,
            execution_configuration=job_config,
            session_id=self.train_id,
            scanner_extras=training_metadata,
        )
        # TODO: SHOULD WE USE PER-HOST ENERGY CONSUMPTION HERE?
        energy_consumption = sum(self.energy_tracker.get_energy_consumption().values())
        job_performance = JobExecutionPerformance(
            running_time_sec=result.runtime_sec,
            energy_use_mwh=energy_consumption
        )

        training_step_results = TrainingStepResults(
            job_descriptor=self.__episodic_job_descriptor,
            job_config=job_config,
            training_metadata=training_metadata,
            job_performance=job_performance,
            training_id=self.train_id,
        )

        self.training_results_logger.info(
            "Summarised Training Step Results",
            extra=asdict(training_step_results)
        )

        return job_performance

    def _init_episodic_job(self, options: dict[str, Any] | None) -> JobProperties:
        if options:
            raise ValueError("Options are not expected in training mode")

        selected_job_type = self.__select_episodic_job_type(self.np_random)
        selected_input_size_gb = self.__select_input_size_gb(selected_job_type, self.np_random)
        self.__episodic_job_descriptor = JobDescriptor(job_type=selected_job_type, input_size_gb=selected_input_size_gb)

        default_execution_configuration = HadoopJobExecutionConfig()
        job_performance = self.__run_job_and_measure_performance(default_execution_configuration, is_baseline=True)
        self.reward_calculator.update_baseline_performance(job_performance)

        return SupportedJobsConfig.extract_job_properties(self.__episodic_job_descriptor)

    def _compute_reward(self, job_config: HadoopJobExecutionConfig, terminated: bool, truncated: bool) -> float:
        # TODO: THINK ABOUT WHAT TO DO WITH THE FIRST ITERATION THAT IS OUTPUTTING NON-RELEVANT ENERGY CONSUMPTION
        #   i think it happens only in the first episode
        job_performance = self.__run_job_and_measure_performance(job_config)
        computed_reward = self.reward_calculator.compute_reward(job_performance)
        self.render()
        return computed_reward

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
