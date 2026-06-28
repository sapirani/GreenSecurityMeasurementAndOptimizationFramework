from pydantic import BaseModel, Field


class TrainingProgressContext(BaseModel):
    global_step: int = Field(ge=0)          # counts every step across all episodes
    job_type_episode: int = Field(ge=0)     # episode count per job type
    job_type_step: int = Field(ge=0)        # total step count per job type
    task_variant_episode: int = Field(ge=0) # episode count per JobDescriptor (basically job type + input size)
    task_variant_step: int = Field(ge=0)    # total step count per JobDescriptor (basically job type + input size)
