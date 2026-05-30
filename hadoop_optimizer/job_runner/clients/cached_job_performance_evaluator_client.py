import inspect
import numpy as np
from elasticsearch import Elasticsearch
from DTOs.hadoop.drl.training.training_step_results import TrainingStepResults
from DTOs.hadoop.job_types import JobType
from enum import Enum
from elasticsearch_dsl import Search, Q
from DTOs.logging.consts import IndexName
from typing import Optional, Dict, List
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_execution_performance import JobExecutionPerformance
from DTOs.range import Range
from hadoop_optimizer.drl_envs.training.training_env import EpisodeContext
from hadoop_optimizer.job_runner.clients.job_performance_evaluator_client import HadoopJobPerformanceEvaluatorClient


class CachedHadoopJobPerformanceEvaluatorClient:
    def __init__(
            self,
            job_performance_evaluator_client: Optional[HadoopJobPerformanceEvaluatorClient] = None,
    ):
        self.job_performance_evaluator_client = job_performance_evaluator_client or HadoopJobPerformanceEvaluatorClient()

        # todo: inject as parameters
        ES_URL = "http://192.168.140.105:9200"
        ES_USER = "elastic"
        ES_PASS = "71BPiEiQ"

        self.es_client = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASS), verify_certs=False)

    def __enter__(self):
        self.start()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.job_performance_evaluator_client.start()

    def stop(self):
        self.job_performance_evaluator_client.stop()

    # TODO: ORGANIZE CODE
    def run_job(
        self,
        job_descriptor: JobDescriptor,
        execution_configuration: HadoopJobExecutionConfig,
        session_id: str,
        episode_context: EpisodeContext,
        space_ranges: Dict[str, Range],
        max_param_diff_percent: float = 33,
    ) -> JobExecutionPerformance:
        """
        # TODO: EXPLAIN WHAT WE ARE DOING HERE: FETCHING ONLY NOT SIMULATED RESULTS OF SIMILAR CONFIGURATIONS
            AND COMPUTE THE MEAN, IN THE CASE WE HAVE AT LEAST 3 SIMILAR TRAINING STEPS
        :raises:
            1.  requests.exceptions.HTTPError: 422 not implemented (typically when selected job execution config is invalid)
            2.  requests.exceptions.HTTPError: 500 internal server error (typically when Hadoop job can't run for some reason)
            3.  requests.exceptions.HTTPError: 501 not implemented (typically when Hadoop is not installed)
            4.  requests.exceptions.HTTPError: 503 gateway timeout (when job execution has passed time limit)
            5.  RuntimeError: in case that the caller did not call start beforehand / use the contextmanager
        """

        # TODO: FETCH RESULTS FROM SIMILAR JobDescriptor + HadoopJobExecutionConfig
        #  AVERAGE THEM
        #  NOTE THAT HAVE HAVE ENOUGH MEASUREMENTS AND THAT THE VARIANCE IS LOW
        #  IF VARIANCE IS HIGH - OUTPUT SOME WARNING AND MAYBE RUN ADDITIONAL EXPERIMEET
        #  IF THERE ARE ENOUGH SAMPLES AND VARIANCE IS LOW, JUST RETURN THE AVERAGE
        #  OTHERWISE, RETURN THE RESULTS THAT CONSIDER THE NEW RUN

        similar_training_steps = self._find_similar_training_steps(
            execution_configuration,
            space_ranges,
            max_param_diff_percent
        )

        similar_execution_results = [
            similar_training_step.job_performance for similar_training_step in similar_training_steps
        ]

        if len(similar_execution_results) < 3:  # run another experiment and combine the new results
            new_experiment_results = self.job_performance_evaluator_client.run_job(
                job_descriptor=job_descriptor,
                execution_configuration=execution_configuration,
                session_id=session_id,
                episode_context=episode_context,
            )

            return JobExecutionPerformance(
                running_time_sec=new_experiment_results.running_time_sec,
                energy_use_mwh=new_experiment_results.energy_use_mwh,
                simulated=False
            )

        # TODO: THINK OF THE CASE WHERE ONE OF THE RETREIVED CONFIGURATIONS IS THE EXACT SAME AS I WANTED, BUT I HAVE
        #   FEW MORE SIMILARITIES THAT ARE NOT THE EXACT THING. SHOULD I CONSIDER ONLY THE EXACT ONE OR ALL THE SIMILARS?
        evaluated_running_time_sec = [
            similar_execution_result.running_time_sec for similar_execution_result in similar_execution_results
        ]
        evaluated_energy_mwh = [
            similar_execution_result.energy_use_mwh for similar_execution_result in similar_execution_results
        ]

        mean_running_time_sec = float(np.mean(evaluated_running_time_sec))
        mean_energy_mwh = float(np.mean(evaluated_energy_mwh))
        # sample std (dataset is considered as a sample from a larger unknown population)
        std_running_time_sec = float(np.std(evaluated_running_time_sec, ddof=1))
        std_energy_mwh = float(np.std(evaluated_energy_mwh, ddof=1))

        # TODO: ADD RANDOM NOISE THAT ITS SCALE IS PROPORTIONATE TO THE STD

        return JobExecutionPerformance(
            running_time_sec=mean_running_time_sec,
            energy_use_mwh=mean_energy_mwh,
            simulated=True,
            std_running_time_sec=std_running_time_sec,
            std_energy_mwh=std_energy_mwh,
            number_of_simulation_participants=len(similar_execution_results)
        )

    @staticmethod
    def _allowed_param_interval(
            desired_val: float,
            param_range: Range,
            max_param_diff_percent: float
    ):
        factor = (max_param_diff_percent / 100) * (param_range.high - param_range.low)
        return Range(
            low=desired_val - factor,
            high=desired_val + factor,
        )

    @staticmethod
    def _allowed_params_interval(
            execution_configuration: HadoopJobExecutionConfig,
            space_ranges: Dict[str, Range],
            max_param_diff_percent: float
    ) -> Dict[str, Range]:
        allowed_ranges = {}
        for field_name, field in HadoopJobExecutionConfig.model_fields.items():
            annotation = field.annotation
            if annotation is bool or CachedHadoopJobPerformanceEvaluatorClient._is_enum_argument(annotation):
                continue

            if field_name in space_ranges:
                allowed_ranges[f'job_config.{field_name}'] = CachedHadoopJobPerformanceEvaluatorClient._allowed_param_interval(
                    getattr(execution_configuration, field_name),
                    space_ranges[field_name],
                    max_param_diff_percent
                )
        return allowed_ranges


    @staticmethod
    def _is_enum_argument(arg_type) -> bool:
        return inspect.isclass(arg_type) and issubclass(arg_type, Enum)

    def _find_matching_events(
            self,
            config: HadoopJobExecutionConfig,
            job_descriptor: JobDescriptor,
            allowed_ranges: Dict[str, Range],
            float_tolerance: float = 1e-6,
    ):
        filters = []

        filters.append(
            Q(
                "range",
                timestamp={
                    "gte": "2026-05-29T00:00:00"
                }
            )
        )

        filters.append(
            Q(
                "term",
                **{"job_performance.simulated": False}
            )
        )

        # Flatten all fields
        fields = [
            *(
                (f"job_descriptor.{field}", value)
                for field, value in job_descriptor.model_dump().items()
            ),
            *(
                (f"job_config.{field}", value)
                for field, value in config.model_dump().items()
            ),
        ]

        for field_name, value in fields:
            if field_name in allowed_ranges:
                r = allowed_ranges[field_name]
                filters.append(
                    Q(
                        "range",
                        **{
                            field_name: {
                                "gte": float(r.low) - float_tolerance,
                                "lte": float(r.high) + float_tolerance,
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
            .query("bool", filter=filters)[:10000]
            .execute()
        )

    def _find_similar_training_steps(
            self,
            execution_configuration: HadoopJobExecutionConfig,
            space_ranges: Dict[str, Range],
            max_param_diff_percent: float
    ) -> List[TrainingStepResults]:
        # compute allowed intervals for all numeric fields
        allowed_ranges = self._allowed_params_interval(execution_configuration, space_ranges, max_param_diff_percent)

        job_descriptor = JobDescriptor(job_type=JobType.word_count, input_size_gb=0.3)

        hits = self._find_matching_events(
            config=execution_configuration,
            job_descriptor=job_descriptor,
            allowed_ranges=allowed_ranges
        )

        similar_training_results = []
        for hit in hits:
            relevant_fields = {k: v for k, v in hit.to_dict().items() if k in TrainingStepResults.model_fields}
            similar_training_results.append(TrainingStepResults.model_validate(relevant_fields))

        return similar_training_results
