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
        self.task_variant_step: DefaultDict[JobDescriptor, int] = defaultdict(int)

    def update_training_progress(self, current_job_descriptor: JobDescriptor):
        """
        This function should be called on each training step
        """
        self.global_step += 1
        self.job_type_step[current_job_descriptor.job_type] += 1
        self.task_variant_step[current_job_descriptor] += 1

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
            job_type_step=self.job_type_step[current_job_descriptor.job_type],
            task_variant_step=self.task_variant_step[current_job_descriptor]
        )
