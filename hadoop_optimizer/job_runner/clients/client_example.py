from human_id import generate_id

from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_types import JobType
from job_runner.clients.job_performance_evaluator_client import HadoopJobPerformanceEvaluatorClient
from DTOs.hadoop.drl.training.episode_context import EpisodeContext


def main():
    selected_id = generate_id(word_count=3)
    with HadoopJobPerformanceEvaluatorClient() as training_client:
        job_descriptor = JobDescriptor(job_type=JobType.word_count, input_size_gb=0.3)
        print(
            training_client.run_job(
                job_descriptor=job_descriptor,
                execution_configuration=HadoopJobExecutionConfig(),
                session_id=selected_id,
                episode_context=EpisodeContext(episode_num=1, episode_step=1, is_baseline=True)
            )
        )


if __name__ == '__main__':
    main()
