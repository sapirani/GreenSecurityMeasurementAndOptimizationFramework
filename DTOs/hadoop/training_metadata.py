from pydantic import BaseModel, Field


class TrainingMetadata(BaseModel):
    episode_num: int = Field(ge=0)
    step_num: int = Field(ge=0)
    is_baseline: bool
