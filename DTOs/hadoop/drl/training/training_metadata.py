from typing import Optional

from pydantic import BaseModel

from DTOs.hadoop.drl.training.episode_context import EpisodeContext
from DTOs.hadoop.drl.training.training_progress_context import TrainingProgressContext


class TrainingMetadata(BaseModel):
    episode_context: EpisodeContext
    progress_context: Optional[TrainingProgressContext]
