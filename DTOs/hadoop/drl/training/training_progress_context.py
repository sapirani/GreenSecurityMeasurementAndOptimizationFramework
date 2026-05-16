from pydantic import BaseModel, Field


class TrainingProgressContext(BaseModel):
    global_step: int = Field(ge=0)          # counts every step across all episodes
    job_type_step: int = Field(ge=0)        # step count per job type
    task_variant_step: int = Field(ge=0)    # step count per JobDescriptor (basically job type + input size)
