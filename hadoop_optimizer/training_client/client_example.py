from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_types import JobType
from hadoop_optimizer.training_client.client import HadoopOptimizerTrainingClient


def main():
    deployment_client = HadoopOptimizerTrainingClient()
    job_descriptor = JobDescriptor(job_type=JobType.word_count, input_size_gb=10)
    print(deployment_client.run_job(job_descriptor=job_descriptor, execution_configuration=HadoopJobExecutionConfig()))


if __name__ == '__main__':
    main()
