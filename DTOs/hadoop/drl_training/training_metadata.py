from typing import Optional

from pydantic import BaseModel

from DTOs.hadoop.drl_training.episode_context import EpisodeContext
from DTOs.hadoop.drl_training.training_progress_context import TrainingProgressContext


class TrainingMetadata(BaseModel):
    episode_context: EpisodeContext
    progress_context: Optional[TrainingProgressContext]
