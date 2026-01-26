from typing import Dict, Any

from pydantic import BaseModel, Field


class TrainingMetadata(BaseModel):
    episode_num: int = Field(ge=0)
    step_num: int = Field(ge=0)
    is_baseline: bool

    @classmethod
    def from_dict(cls, metadata_dict: Dict[str, Any]) -> "TrainingMetadata":
        field_names = set(cls.model_fields.keys())

        relevant_fields = {
            key: val for key, val in metadata_dict.items() if key in field_names
        }

        missing_fields = field_names - set(relevant_fields.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        return cls(**relevant_fields)
