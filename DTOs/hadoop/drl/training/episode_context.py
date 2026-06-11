from typing import Any, Dict
from pydantic import BaseModel, Field
from DTOs.hadoop.drl.training.extended_episode_context import ExtendedEpisodeContext


class EpisodeContext(BaseModel):
    episode_num: int = Field(ge=0, description="Current episode number in training")
    episode_step: int = Field(ge=0, description="Step number within this episode")

    @classmethod
    def from_dict(cls, metadata_dict: Dict[str, Any]) -> "EpisodeContext":
        field_names = set(cls.model_fields.keys())

        relevant_fields = {
            key: val for key, val in metadata_dict.items() if key in field_names
        }

        missing_fields = field_names - set(relevant_fields.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        return cls(**relevant_fields)

    @classmethod
    def from_episode_context(cls, episode_context: ExtendedEpisodeContext) -> "EpisodeContext":
        return EpisodeContext(
            episode_num=episode_context.episode_num,
            episode_step=episode_context.episode_step
        )
