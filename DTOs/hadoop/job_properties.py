from fastapi import Query
from pydantic import BaseModel, Field


class JobProperties(BaseModel):
    input_size_gb: float    # TODO: NOT ALWAYS THE INPUT SIZE IMPACTS THE PERFORMANCE (E.G., IN MONTE CARLO PI)
    cpu_bound_scale: float
    io_bound_scale: float

    def __str__(self):
        return self.__repr_str__('\n')


def get_job_properties(
    input_size_gb: float = Query(
        ...,
        gt=0,
        description="The input dataset size in GB. Must be a positive number."
    ),
    cpu_bound_scale: float = Query(
        ...,
        ge=0.0,
        le=1.0,
        description="Indicates how CPU-bound the task is on a scale from 0 to 1."
                    " (0 = not CPU-bound, 1 = fully CPU-bound)."
    ),
    io_bound_scale: float = Query(
        ...,
        ge=0.0,
        le=1.0,
        description="Indicates how I/O-bound the task is on a scale from 0 to 1."
                    " (0 = not I/O-bound, 1 = fully I/O-bound)."
    )

) -> JobProperties:
    return JobProperties(
        input_size_gb=input_size_gb,
        cpu_bound_scale=cpu_bound_scale,
        io_bound_scale=io_bound_scale,
    )