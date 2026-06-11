from pydantic import BaseModel, Field, model_validator


class ExtendedEpisodeContext(BaseModel):
    episode_num: int = Field(ge=0, description="Current episode number in training")
    episode_step: int = Field(ge=0, description="Step number within this episode")
    is_baseline: bool
    is_last_step: bool

    @model_validator(mode="after")
    def check_invalid_combination(self):
        if self.is_baseline and self.is_last_step:
            raise ValueError("A step cannot be both `baseline` and `last step`")
        return self
