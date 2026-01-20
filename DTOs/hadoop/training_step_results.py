from dataclasses import dataclass

from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_execution_performance import JobExecutionPerformance
from DTOs.hadoop.training_metadata import TrainingMetadata


@dataclass(frozen=True)
class TrainingStepResults:
    job_descriptor: JobDescriptor
    job_config: HadoopJobExecutionConfig
    training_metadata: TrainingMetadata
    job_performance: JobExecutionPerformance
    training_id: str
