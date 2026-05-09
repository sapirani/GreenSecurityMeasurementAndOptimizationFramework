from pydantic import BaseModel, Field

from DTOs.hadoop.job_types import JobType


class JobDescriptor(BaseModel):
    job_type: JobType = Field(
        ...,
        description="Select the type of Hadoop job to run.",
    )

    input_size_gb: float = Field(
        ...,
        gt=0,
        description="Select the input size (in gigabytes) for the Hadoop job. Must be greater than 0.",
    )

    model_config = {
        "frozen": True  # ensures that this class is immutable
    }
