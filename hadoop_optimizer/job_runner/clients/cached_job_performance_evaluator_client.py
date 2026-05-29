from typing import Optional
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_execution_performance import JobExecutionPerformance
from hadoop_optimizer.drl_envs.training.training_env import EpisodeContext
from hadoop_optimizer.job_runner.clients.job_performance_evaluator_client import HadoopJobPerformanceEvaluatorClient


class CachedHadoopJobPerformanceEvaluatorClient:
    def __init__(
            self,
            job_performance_evaluator_client: Optional[HadoopJobPerformanceEvaluatorClient] = None,
    ):
        self.job_performance_evaluator_client = job_performance_evaluator_client or HadoopJobPerformanceEvaluatorClient()

    def __enter__(self):
        self.start()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.job_performance_evaluator_client.start()

    def stop(self):
        self.job_performance_evaluator_client.stop()

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

        # TODO: FETCH RESULTS FROM SIMILAR JobDescriptor + HadoopJobExecutionConfig
        #  AVERAGE THEM
        #  NOTE THAT HAVE HAVE ENOUGH MEASUREMENTS AND THAT THE VARIANCE IS LOW
        #  IF VARIANCE IS HIGH - OUTPUT SOME WARNING AND MAYBE RUN ADDITIONAL EXPERIMEET
        #  IF THERE ARE ENOUGH SAMPLES AND VARIANCE IS LOW, JUST RETURN THE AVERAGE
        #  OTHERWISE, RETURN THE RESULTS THAT CONSIDER THE NEW RUN


        return self.job_performance_evaluator_client.run_job(
            job_descriptor=job_descriptor,
            execution_configuration=execution_configuration,
            session_id=session_id,
            episode_context=episode_context,
        )
