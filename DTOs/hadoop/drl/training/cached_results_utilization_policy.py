from pydantic import BaseModel, Field


class CachedResultsUtilizationPolicy(BaseModel):
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
