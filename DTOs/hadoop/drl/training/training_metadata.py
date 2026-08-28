from typing import Optional

from pydantic import BaseModel

from DTOs.hadoop.drl.training.extended_episode_context import ExtendedEpisodeContext
from DTOs.hadoop.drl.training.training_progress_context import TrainingProgressContext


class TrainingMetadata(BaseModel):
    episode_context: ExtendedEpisodeContext
    progress_context: Optional[TrainingProgressContext]
