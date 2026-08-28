from collections import defaultdict
from typing import DefaultDict, Optional

from DTOs.hadoop.drl.training.training_progress_context import TrainingProgressContext
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_types import JobType


class TrainingProgressTracker:
    """
    Tracks training progression counters at different granularities:
    - global_step: total number of steps across all episodes
    - job_type_step: number of steps for each JobType
    - task_variant_step: number of steps for each unique JobDescriptor (basically job type + input size)
    """
    def __init__(self):
        self.global_step = 0
        self.job_type_step: DefaultDict[JobType, int] = defaultdict(int)
        self.job_type_episode: DefaultDict[JobType, int] = defaultdict(int)
        self.task_variant_step: DefaultDict[JobDescriptor, int] = defaultdict(int)
        self.task_variant_episode: DefaultDict[JobDescriptor, int] = defaultdict(int)

        self.current_episode_number = 0

    def update_training_progress(self, current_job_descriptor: JobDescriptor, current_episode_number: int):
        """
        This function should be called on each training step
        """

        self.global_step += 1
        self.job_type_step[current_job_descriptor.job_type] += 1
        self.task_variant_step[current_job_descriptor] += 1

        if current_episode_number != self.current_episode_number: # new episode
            self.job_type_episode[current_job_descriptor.job_type] += 1
            self.task_variant_episode[current_job_descriptor] += 1

            self.current_episode_number = current_episode_number

    def get_progress_context(
            self,
            current_job_descriptor: JobDescriptor,
            *,
            is_baseline: bool
    ) -> Optional[TrainingProgressContext]:
        if is_baseline:
            return None

        return TrainingProgressContext(
            global_step=self.global_step,
            job_type_episode=self.job_type_episode[current_job_descriptor.job_type],
            job_type_step=self.job_type_step[current_job_descriptor.job_type],
            task_variant_episode=self.task_variant_episode[current_job_descriptor],
            task_variant_step=self.task_variant_step[current_job_descriptor]
        )
