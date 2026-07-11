import math
import random
from datetime import datetime
import numpy as np
from elasticsearch import Elasticsearch
from pydantic import BaseModel

from DTOs.hadoop.consts import DocumentID, SimilarityScore
from DTOs.hadoop.drl.training.cached_results_utilization_policy import CachedResultsUtilizationPolicy
from DTOs.hadoop.drl.training.episode_context import EpisodeContext
from DTOs.hadoop.drl.training.training_step_results import TrainingStepResults
from enum import Enum
from elasticsearch_dsl import Search, Q
from DTOs.logging.consts import IndexName
from typing import Optional, Dict, List, Union, Tuple
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_execution_performance import JobExecutionPerformance
from DTOs.range import Range
from hadoop_optimizer.job_runner.clients.job_performance_evaluator_client import HadoopJobPerformanceEvaluatorClient
from hadoop_optimizer.common.utils import get_full_field_name, is_enum_argument

MAX_SIMILAR_RESULTS = 10000


class CachedHadoopJobPerformanceEvaluatorClient:
    """
    This class fetches previously performed experiments from Elasticsearch, to avoid waisting time when running
    similar jobs all over again
    """

    def __init__(
            self,
            elastic_url: str,
            elastic_user: str,
            elastic_password: str,
            job_performance_evaluator_client: Optional[HadoopJobPerformanceEvaluatorClient] = None,
            cached_results_utilization_policy: Optional[CachedResultsUtilizationPolicy] = None,
            search_since: Optional[datetime] = None,
            force_real_execution_probability: float = 0.001
    ):
        self.job_performance_evaluator_client = job_performance_evaluator_client or HadoopJobPerformanceEvaluatorClient()
        self.cached_results_utilization_policy = cached_results_utilization_policy or CachedResultsUtilizationPolicy()
        self.search_since = search_since or datetime.min
        self.force_real_execution_probability = force_real_execution_probability

        self.es_client = Elasticsearch(elastic_url, basic_auth=(elastic_user, elastic_password), verify_certs=False)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.job_performance_evaluator_client.start()

    def stop(self):
        self.job_performance_evaluator_client.stop()

    @staticmethod
    def _compute_similarity_weights(
            similarity_scores: Dict[DocumentID, float],
            temperature: float = 1.0
    ) -> Tuple[List[DocumentID], np.ndarray]:

        if temperature <= 0:
            raise ValueError("Temperature must be > 0")

        ids = list(similarity_scores)
        values = np.array([similarity_scores[_id] for _id in ids], dtype=float)

        min_v = values.min()
        max_v = values.max()

        if max_v > min_v:
            values = (values - min_v) / (max_v - min_v)

        # higher temperature = smoother distribution
        scaled = values / max(temperature, 1e-8)

        # Note: to prevent high values, we reduce the maximum score from all exponents
        weights = np.exp(scaled - np.max(scaled))
        weights /= weights.sum()

        return ids, weights

    @staticmethod
    def _weighted_mean_std(
            values: np.ndarray,
            weights: np.ndarray
    ) -> Tuple[float, float]:
        mean = float(np.average(values, weights=weights))
        variance = float(np.average((values - mean) ** 2, weights=weights))
        return mean, math.sqrt(variance)

    def _calc_simulated_job_performance(
            self,
            similar_execution_results: Dict[DocumentID, JobExecutionPerformance],
            similarity_scores: Dict[DocumentID, SimilarityScore],
            results_noise_scale: float
    ) -> JobExecutionPerformance:
        """
        Uses weighting that is proportional to the similarity score of each configuration relative to the
        properties of the requested job.
        """
        if not similar_execution_results:
            raise ValueError("Similar results must not be empty")

        if similarity_scores.keys() != similar_execution_results.keys():
            raise ValueError("Must received the same Document IDs in both similarity scores and execution results")

        document_ids, weights = self._compute_similarity_weights(
            similarity_scores,
            self.cached_results_utilization_policy.similarity_temperature
        )

        running_times_sec = np.array(
            [similar_execution_results[_id].running_time_sec for _id in document_ids],
            dtype=float
        )

        energies_mwh = np.array(
            [similar_execution_results[_id].energy_use_mwh for _id in document_ids],
            dtype=float
        )

        mean_running_time_sec, std_running_time_sec = self._weighted_mean_std(running_times_sec, weights)
        mean_energy_mwh, std_energy_mwh = self._weighted_mean_std(energies_mwh, weights)

        # TODO: CONSIDER LOGGING LARGE STANDARD DEVIATIONS AS WARNINGS
        # TODO: CONSIDER RUNNING THE JOB WITHOUT SIMULATIONS IF THE STANDARD DEVIATION IS TOO LARGE

        # Gaussian noise proportional to std
        running_time_sec_noise = float(
            np.random.normal(loc=0.0, scale=std_running_time_sec * results_noise_scale)
        )

        energy_mwh_noise = float(
            np.random.normal(loc=0.0, scale=std_energy_mwh * results_noise_scale)
        )

        return JobExecutionPerformance(
            running_time_sec=mean_running_time_sec + running_time_sec_noise,
            energy_use_mwh=mean_energy_mwh + energy_mwh_noise,
            simulated=True,
            std_running_time_sec=std_running_time_sec,
            std_energy_mwh=std_energy_mwh,
            selected_running_time_sec_noise=running_time_sec_noise,
            selected_energy_use_mwh_noise=energy_mwh_noise,
            similarity_weights=dict(zip(document_ids, weights)),
        )

    @staticmethod
    def _compute_similarity_score(
            desired_val: Union[float, int],
            similar_val: Union[float, int],
            param_range: Range
    ) -> SimilarityScore:
        soft_range = max(param_range.high - param_range.low, 1e-9)

        if abs(desired_val - similar_val) > soft_range:
            raise ValueError(
                f"Values diff should not exceed the parameter range size. \n"
                f"Diff: {desired_val}. \n"
                f"Range size: {soft_range} [{param_range.low},{param_range.high}]"
            )

        return SimilarityScore(1 - (abs(desired_val - similar_val) / soft_range))

    @staticmethod
    def _compute_similarity_scores(
            execution_configuration: HadoopJobExecutionConfig,
            similar_training_steps: Dict[DocumentID, TrainingStepResults],
            space_ranges: Dict[str, Range],
    ) -> Dict[DocumentID, SimilarityScore]:
        supported_field_names = sorted(set(HadoopJobExecutionConfig.model_fields).intersection(set(space_ranges)))

        if not supported_field_names:
            print("WARNING: no supported fields for computing similarity scores")
            return {}

        similarity_scores = {}
        for _id, training_step in similar_training_steps.items():
            similarity_score = sum(
                CachedHadoopJobPerformanceEvaluatorClient._compute_similarity_score(
                    getattr(execution_configuration, field_name),
                    getattr(training_step.job_config, field_name),
                    space_ranges[field_name],
                )
                for field_name in supported_field_names
            )
            similarity_scores[_id] = similarity_score / len(supported_field_names)

        return similarity_scores

    def _uses_highly_deviated_results(self, simulated_performance: JobExecutionPerformance) -> bool:
        if not simulated_performance.simulated:
            return False

        runtime_avg = simulated_performance.running_time_sec - simulated_performance.selected_running_time_sec_noise
        energy_avg = simulated_performance.energy_use_mwh - simulated_performance.selected_energy_use_mwh_noise
        return (
                (simulated_performance.std_running_time_sec / runtime_avg) * 100 >
                self.cached_results_utilization_policy.running_time_max_deviation_percent
        ) or (
                (simulated_performance.std_energy_mwh / energy_avg) * 100 >
                self.cached_results_utilization_policy.energy_max_deviation_percent
        )

    def run_job(
            self,
            job_descriptor: JobDescriptor,
            execution_configuration: HadoopJobExecutionConfig,
            session_id: str,
            episode_context: EpisodeContext,
            space_ranges: Dict[str, Range],
            cached_results_utilization_policy: Optional[CachedResultsUtilizationPolicy]
    ) -> JobExecutionPerformance:
        """
        If there are enough results of similar samples from Elasticsearch - use them as cache to simulate the result
        of the currently requested job. Otherwise, run the actual job and return its results.
        For simulated results - The weighted mean of fetched similar results is used as an estimator to the
        job's performance. The weighting is proportional to the similarity score of each sample relative to the
        properties of the requested job.
        An additional interfacial noise is added, which is propositional to the standard deviation of the
        fetched similar results.
        :raises:
            1.  requests.exceptions.HTTPError: 422 not implemented (typically when selected job execution config is invalid)
            2.  requests.exceptions.HTTPError: 500 internal server error (typically when Hadoop job can't run for some reason)
            3.  requests.exceptions.HTTPError: 501 not implemented (typically when Hadoop is not installed)
            4.  requests.exceptions.HTTPError: 503 gateway timeout (when job execution has passed time limit)
            5.  RuntimeError: in case that the caller did not call start beforehand / use the contextmanager
        """
        cached_results_utilization_policy = cached_results_utilization_policy or self.cached_results_utilization_policy
        assert cached_results_utilization_policy

        similar_training_steps = self._find_similar_training_steps(
            job_descriptor,
            execution_configuration,
            space_ranges,
            cached_results_utilization_policy.max_param_diff_percent
        )

        similar_execution_results = {
            _id: similar_training_step.job_performance for _id, similar_training_step in similar_training_steps.items()
        }

        # run another experiment and combine the new results
        if len(similar_execution_results) < cached_results_utilization_policy.min_required_similar_samples:
            return self.job_performance_evaluator_client.run_job(
                job_descriptor=job_descriptor,
                execution_configuration=execution_configuration,
                session_id=session_id,
                episode_context=episode_context,
            )

        if random.random() < self.force_real_execution_probability:
            return self.job_performance_evaluator_client.run_job(
                job_descriptor=job_descriptor,
                execution_configuration=execution_configuration,
                session_id=session_id,
                episode_context=episode_context,
            )

        similarity_scores = self._compute_similarity_scores(
            execution_configuration,
            similar_training_steps,
            space_ranges
        )

        simulated_performance = self._calc_simulated_job_performance(
            similar_execution_results,
            similarity_scores,
            cached_results_utilization_policy.results_noise_scale
        )

        if self._uses_highly_deviated_results(simulated_performance):
            print("Executing the requested job due to high deviation in similar simulated jobs")
            real_performance = self.job_performance_evaluator_client.run_job(
                job_descriptor=job_descriptor,
                execution_configuration=execution_configuration,
                session_id=session_id,
                episode_context=episode_context,
            )

            # Return real results along with simulation metadata to investigate cases with large deviations.
            return simulated_performance.model_copy(update={
                "running_time_sec": real_performance.running_time_sec,
                "energy_use_mwh": real_performance.energy_use_mwh,
                "simulated": False,
            })

        return simulated_performance

    @staticmethod
    def _calc_similarity_interval(
            desired_val: float,
            param_range: Range,
            max_param_diff_percent: float
    ):
        max_deviation = (max_param_diff_percent / 100) * (param_range.high - param_range.low)
        return Range(
            low=desired_val - max_deviation,
            high=desired_val + max_deviation,
        )

    @staticmethod
    def params_similarity_ranges(
            execution_configuration: HadoopJobExecutionConfig,
            space_ranges: Dict[str, Range],
            max_param_diff_percent: float  # TODO: consider an extension where each param defines its own max diff
    ) -> Dict[str, Range]:
        """
        Returs, for each config parameter, the range of values that are considered "similar" enough to
        the provided execution configuration.
        Note: booleans and are excluded as values must be the identical to be considered as "similar"
        :param execution_configuration: the configuration that we want to understand what are the "similarity ranges" for each of its parameters
        :param space_ranges: the lower value and higher value of each parameter range
        :param max_param_diff_percent: the relative percentage (compared to the scale of each parameter's range), where
            values that reside within this range are considered "similar".
        :return: a dictionary that maps each parameter name to the range of values representing "similar" enough" values.
        """
        similarity_ranges = {}
        for field_name, field in HadoopJobExecutionConfig.model_fields.items():
            if field.annotation is bool or is_enum_argument(field.annotation):
                continue

            if field_name in space_ranges:
                similarity_ranges[field_name] = CachedHadoopJobPerformanceEvaluatorClient._calc_similarity_interval(
                    getattr(execution_configuration, field_name),
                    space_ranges[field_name],
                    max_param_diff_percent
                )

        return similarity_ranges

    @staticmethod
    def full_field_name(field_name: str) -> str:
        """
        Assuming that all results were logged from TrainingStepResults
        """
        return get_full_field_name(field_name, TrainingStepResults)

    @staticmethod
    def model_dump_with_full_names(basemodel: BaseModel):
        return {
            CachedHadoopJobPerformanceEvaluatorClient.full_field_name(field): value
            for field, value in basemodel.model_dump().items()
        }

    def _find_matching_events(
            self,
            job_descriptor: JobDescriptor,
            config: HadoopJobExecutionConfig,
            similarity_ranges: Dict[str, Range],
            float_tolerance: float = 1e-6,
    ):
        """
        Finds not-simulated, similar results from Elasticsearch.
        """
        filters = []

        filters.append(
            Q(
                "range",
                timestamp={
                    "gte": self.search_since
                }
            )
        )

        # Flatten all fields
        query_fields = {
            self.full_field_name("simulated"): False,
            **self.model_dump_with_full_names(job_descriptor),
            **self.model_dump_with_full_names(config)
        }

        full_name_similarity_ranges = {
            CachedHadoopJobPerformanceEvaluatorClient.full_field_name(k): v for k, v in similarity_ranges.items()
        }

        for field_name, value in query_fields.items():
            if field_name in full_name_similarity_ranges:
                similarity_range = full_name_similarity_ranges[field_name]
                filters.append(
                    Q(
                        "range",
                        **{
                            field_name: {
                                "gte": float(similarity_range.low) - float_tolerance,
                                "lte": float(similarity_range.high) + float_tolerance,
                            }
                        },
                    )
                )

            elif isinstance(value, Enum):
                value = value.value
                filters.append(
                    Q(
                        "bool",
                        should=[
                            Q("term", **{field_name: value}),
                            Q("term", **{field_name: str(value).lower()}),
                        ],
                        minimum_should_match=1,
                    )
                )

            elif isinstance(value, float):
                # fallback: small tolerance if no allowed_range
                filters.append(
                    Q(
                        "range",
                        **{
                            field_name: {
                                "gte": value - float_tolerance,
                                "lte": value + float_tolerance,
                            }
                        },
                    )
                )

            else:
                filters.append(Q("term", **{field_name: value}))

        return (
            Search(using=self.es_client, index=IndexName.DRL_TRAINING)
            .query("bool", filter=filters)[:MAX_SIMILAR_RESULTS]
            .execute()
        )

    def _find_similar_training_steps(
            self,
            job_descriptor: JobDescriptor,
            execution_configuration: HadoopJobExecutionConfig,
            space_ranges: Dict[str, Range],
            max_param_diff_percent: float
    ) -> Dict[DocumentID, TrainingStepResults]:
        """
        This function finds similar results from the training results index in Elasticsearch.
        It fetches the relevant events and converts them into the original object that they were logged from.
        :return:
        """
        # compute allowed intervals for all numeric fields
        params_similarity_ranges = self.params_similarity_ranges(
            execution_configuration,
            space_ranges,
            max_param_diff_percent
        )

        hits = self._find_matching_events(
            job_descriptor=job_descriptor,
            config=execution_configuration,
            similarity_ranges=params_similarity_ranges
        )

        similar_training_results = {}
        for hit in hits:
            relevant_fields = {k: v for k, v in hit.to_dict().items() if k in TrainingStepResults.model_fields}
            similar_training_results[hit.meta.id] = TrainingStepResults.model_validate(relevant_fields)

        return similar_training_results
