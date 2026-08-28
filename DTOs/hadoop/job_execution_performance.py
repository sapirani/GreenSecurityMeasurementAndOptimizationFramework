from typing import Optional, Dict, Any
from DTOs.hadoop.consts import DocumentID
from pydantic import BaseModel, model_validator


class JobExecutionPerformance(BaseModel):
    running_time_sec: float     # including noise when simulated=True
    energy_use_mwh: float       # including noise when simulated=True
    simulated: bool = False
    running_time_sec_by_similar_jobs: Optional[float] = None    # excluding noise
    energy_use_mwh_by_similar_jobs: Optional[float] = None      # excluding noise
    std_running_time_sec: Optional[float] = None
    std_energy_mwh: Optional[float] = None
    selected_running_time_sec_noise: Optional[float] = None
    selected_energy_use_mwh_noise: Optional[float] = None
    similarity_weights: Optional[Dict[DocumentID, float]] = None

    model_config = {
        "frozen": True  # ensures that this class is immutable
    }

    def __str__(self) -> str:
        parts = [
            f"running_time_sec={self.running_time_sec:.2f}",
            f"energy_use_mwh={self.energy_use_mwh:.2f}",
            f"simulated={self.simulated}",
        ]

        if self.std_running_time_sec is not None:
            parts.append(f"std_running_time_sec={self.std_running_time_sec:.2f}")

        if self.std_energy_mwh is not None:
            parts.append(f"std_energy_mwh={self.std_energy_mwh:.2f}")

        if self.selected_running_time_sec_noise is not None:
            parts.append(
                f"selected_running_time_sec_noise={self.selected_running_time_sec_noise:.2f}"
            )

        if self.selected_energy_use_mwh_noise is not None:
            parts.append(
                f"selected_energy_use_mwh_noise={self.selected_energy_use_mwh_noise:.2f}"
            )

        if self.similarity_weights:
            top_weights = sorted(
                self.similarity_weights.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            weights_str = ", ".join(
                f"{doc_id}: {weight:.3f}"
                for doc_id, weight in top_weights
            )

            parts.append(f"similarity_weights={{ {weights_str} }}")

        return f"JobExecutionPerformance({', '.join(parts)})"

    @model_validator(mode="before")
    @classmethod
    def populate_similar_job_values(cls, data):
        if isinstance(data, dict) and data.get("simulated", cls.model_fields["simulated"].default):
            data = data.copy()

            data.setdefault(
                "running_time_sec_by_similar_jobs",
                data["running_time_sec"] - data["selected_running_time_sec_noise"],
            )
            data.setdefault(
                "energy_use_mwh_by_similar_jobs",
                data["energy_use_mwh"] - data["selected_energy_use_mwh_noise"],
            )

        return data

    @model_validator(mode="after")
    def validate_simulation_fields(self):
        simulation_fields = [
            self.std_running_time_sec,
            self.std_energy_mwh,
            self.similarity_weights,
            self.selected_running_time_sec_noise,
            self.selected_energy_use_mwh_noise,
        ]

        has_any_simulated_field = any(v is not None for v in simulation_fields)
        has_all_simulated_field = all(v is not None for v in simulation_fields)

        if self.simulated:
            if not has_all_simulated_field:
                raise ValueError("When simulated=True, all simulation fields must be provided")

        else:
            if has_any_simulated_field and not has_all_simulated_field:
                raise ValueError(
                    "When simulated=False, either all simulation fields should be provided or none of them"
                )

        return self
