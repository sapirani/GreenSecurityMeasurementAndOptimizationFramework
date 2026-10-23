from pydantic import BaseModel, Field


class CachedResultsUtilizationPolicy(BaseModel):
    # TODO: CONSIDER MAX DIFF PERCENT PER CONFIG PARAM (e.g., to support high diff in vcores but low diff in memory)
    max_param_diff_percent: float = Field(
        default=27,
        ge=0,
        le=100,
    )
    min_required_similar_samples: int = Field(
        default=3,
        ge=0,
    )

    results_noise_scale: float = Field(
        default=0.3,
        ge=0,
    )

    similarity_temperature: float = Field(
        default=0.25,
        ge=0,
    )

    running_time_max_deviation_percent: float = Field(
        default=10,
        ge=0,
    )

    energy_max_deviation_percent: float = Field(
        default=18,
        ge=0,
    )
