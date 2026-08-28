from typing import Optional
from pydantic import BaseModel, model_validator
from DTOs.hadoop.drl.training.training_metadata import TrainingMetadata
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_execution_performance import JobExecutionPerformance


class TrainingStepResults(BaseModel):
    training_id: str
    job_descriptor: JobDescriptor
    job_config: HadoopJobExecutionConfig
    training_metadata: TrainingMetadata
    job_performance: JobExecutionPerformance
    step_reward: Optional[float] = None
    cumulative_reward: Optional[float] = None

    model_config = {
        "frozen": True  # ensures that this class is immutable
    }

    @model_validator(mode="after")
    def validate_rewards(self):
        if (self.step_reward is None) != (self.cumulative_reward is None):
            raise ValueError(
                "Step Reward and Cumulative Reward must both be set or both be None"
            )
        return self
