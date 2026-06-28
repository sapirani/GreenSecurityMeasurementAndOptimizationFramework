from logging import Logger
from typing import Any, Optional

import numpy as np

from DTOs.hadoop.drl.job_properties import JobProperties
from DTOs.hadoop.drl.training.cached_results_utilization_policy import CachedResultsUtilizationPolicy
from DTOs.hadoop.drl.training.episode_context import EpisodeContext
from DTOs.hadoop.drl.training.extended_episode_context import ExtendedEpisodeContext
from DTOs.hadoop.drl.training.training_metadata import TrainingMetadata
from DTOs.hadoop.drl.training.training_step_results import TrainingStepResults
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_execution_performance import JobExecutionPerformance
from DTOs.hadoop.job_types import JobType
from hadoop_optimizer.common.drl_telemetry.telemetry_aggregator import TelemetryAggregator
from hadoop_optimizer.common.supported_jobs.supported_jobs_config import SupportedJobsConfig
from hadoop_optimizer.drl_envs.abstract_hadoop_optimizer_env import AbstractOptimizerEnvInterface
from hadoop_optimizer.drl_envs.training.reward.reward_calculator import RewardCalculator
from hadoop_optimizer.drl_envs.training.training_progress_tracker import TrainingProgressTracker
from hadoop_optimizer.job_runner.clients.cached_job_performance_evaluator_client import CachedHadoopJobPerformanceEvaluatorClient


class OptimizerTrainingEnv(AbstractOptimizerEnvInterface):
    def __init__(
            self,
            telemetry_aggregator: TelemetryAggregator,
            training_client: CachedHadoopJobPerformanceEvaluatorClient,
            reward_calculator: RewardCalculator,
            train_id: str,
            training_results_logger: Logger,
            training_progress_tracker: TrainingProgressTracker,
            cached_results_utilization_policy: CachedResultsUtilizationPolicy
    ):
        super().__init__(telemetry_aggregator)
        self.training_client = training_client
        self.training_client.start()
        self.reward_calculator = reward_calculator
        self.train_id = train_id
        self.training_results_logger = training_results_logger
        self.training_progress_tracker = training_progress_tracker
        self.cached_results_utilization_policy = cached_results_utilization_policy

        self.__episodic_job_descriptor: Optional[JobDescriptor] = None
        self.__current_step_performance: Optional[JobExecutionPerformance] = None
        self.__current_step_reward = None

    def _extra_step_init(self):
        assert self.__episodic_job_descriptor is not None
        self.training_progress_tracker.update_training_progress(self.__episodic_job_descriptor, self.episode_counter)

    def _custom_rendering(self):
        assert self.__episodic_job_descriptor is not None
        print("Episodic Job Type:", self.__episodic_job_descriptor.job_type.value)
        print("Training Progress Context:",
              self.training_progress_tracker.get_progress_context(self.__episodic_job_descriptor, is_baseline=False))
        print("Episodic Baseline Performance:", self.reward_calculator.baseline_performance)
        print("Current Job Performance:", self.__current_step_performance)
        print("Current Step Reward:", self.__current_step_reward)
        print("Episodic Cumulative Reward:", self._cumulative_reward + self.__current_step_reward)

    def __get_episode_context(
            self,
            *,
            is_baseline: bool = False,
            is_last_step: bool = False
    ) -> ExtendedEpisodeContext:
        return ExtendedEpisodeContext(
            episode_num=self.episode_counter,
            episode_step=self.step_count,
            is_baseline=is_baseline,
            is_last_step=is_last_step,
        )

    def __log_results(
            self,
            job_config: HadoopJobExecutionConfig,
            job_performance: JobExecutionPerformance,
            step_reward: Optional[float] = None,
            cumulative_reward: Optional[float] = None,
            *,
            is_baseline: bool = False,
            is_last_step: bool = False,
    ):
        """
        :param cumulative_reward: episodic reward (including the current step reward)
        """
        assert self.__episodic_job_descriptor is not None
        if is_baseline and step_reward:
            raise ValueError("Reward values are not expected when logging the baseline performance")

        progress_context = self.training_progress_tracker.get_progress_context(
            self.__episodic_job_descriptor,
            is_baseline=is_baseline
        )

        episode_context = self.__get_episode_context(is_baseline=is_baseline, is_last_step=is_last_step)

        training_step_results = TrainingStepResults(
            training_id=self.train_id,
            job_descriptor=self.__episodic_job_descriptor,
            job_config=job_config,
            training_metadata=TrainingMetadata(episode_context=episode_context, progress_context=progress_context),
            job_performance=job_performance,
            step_reward=step_reward,
            cumulative_reward=cumulative_reward
        )

        self.training_results_logger.info(
            "Summarized Training Step Results",
            extra=training_step_results.dict()
        )

    def __run_job_and_measure_performance(
            self,
            job_config: HadoopJobExecutionConfig,
            *,
            is_baseline: bool = False,
            is_last_step: bool = False
    ) -> JobExecutionPerformance:
        # TODO: IMPORTANT OPTIMIZATION OF CHECKING IF SOME RESULTS FOR THE SAME CONFIGURATION AND INPUT SIZE ALREADY
        #   EXIST IN THE TRAINING_DRL INDEX, AND RETURN THOSE RESULTS IMMEDIATELY
        #   NOTE: WE MAY ADD INTENTIONAL NOISE FOR THOSE RESULTS, AND WE SHOULD IMPLEMENT "SIMILARITY" MECHANISM
        #   (AS IF WE HAVE A VERY SIMILAR CONFIGURATION BUT NOT EXACTLY THE SAME IN THE DRL_TRAINING INDEX)
        #   THAT IS BEING MORE GRANULAR OVER STEPS.
        assert self.__episodic_job_descriptor is not None
        episode_context = self.__get_episode_context(is_baseline=is_baseline, is_last_step=is_last_step)

        return self.training_client.run_job(
            job_descriptor=self.__episodic_job_descriptor,
            execution_configuration=job_config,
            session_id=self.train_id,
            episode_context=EpisodeContext.from_episode_context(episode_context),
            space_ranges=self._get_space_ranges(self.job_config_space),
            # TODO: Consider making the following policy adaptive as the training progress
            cached_results_utilization_policy=self.cached_results_utilization_policy,
        )

    def _init_episodic_job(self, options: dict[str, Any] | None) -> JobProperties:
        if options:
            raise ValueError("Options are not expected in training mode")

        selected_job_type = self.__select_episodic_job_type(self.np_random)
        selected_input_size_gb = self.__select_input_size_gb(selected_job_type, self.np_random)
        self.__episodic_job_descriptor = JobDescriptor(job_type=selected_job_type, input_size_gb=selected_input_size_gb)

        default_execution_configuration = HadoopJobExecutionConfig()
        job_performance = self.__run_job_and_measure_performance(
            default_execution_configuration,
            is_baseline=True,
            is_last_step=False,
        )
        self.__log_results(default_execution_configuration, job_performance, is_baseline=True)
        self.reward_calculator.update_baseline_performance(job_performance)

        assert self.__episodic_job_descriptor is not None
        return SupportedJobsConfig.extract_job_properties(self.__episodic_job_descriptor)

    def _compute_reward(self, job_config: HadoopJobExecutionConfig, *, terminated: bool, truncated: bool) -> float:
        self.__current_step_performance = self.__run_job_and_measure_performance(job_config, is_last_step=terminated)
        assert self.__current_step_performance is not None

        self.__current_step_reward = self.reward_calculator.compute_reward(
            self.__current_step_performance,
            is_last_step=terminated,
            is_truncated=truncated
        )

        self.__log_results(
            job_config,
            self.__current_step_performance,
            self.__current_step_reward,
            self._cumulative_reward + self.__current_step_reward,
            is_baseline=False,
            is_last_step=terminated,
        )

        self.render()
        return self.__current_step_reward

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

    def close(self):
        self.training_client.stop()
