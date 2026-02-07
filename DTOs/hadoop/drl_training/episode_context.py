from typing import Dict, Any
from pydantic import BaseModel, Field


class EpisodeContext(BaseModel):
    episode_num: int = Field(ge=0, description="Current episode number in training")
    episode_step: int = Field(ge=0, description="Step number within this episode")
    is_baseline: bool

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
