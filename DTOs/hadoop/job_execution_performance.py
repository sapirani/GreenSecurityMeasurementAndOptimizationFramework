from typing import Optional, Dict
from DTOs.hadoop.consts import DocumentID
from pydantic import BaseModel, model_validator


class JobExecutionPerformance(BaseModel):
    running_time_sec: float
    energy_use_mwh: float
    # TODO: CHANGE TO IS_SIMULATED?
    simulated: bool = False
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
            )[:3]

            weights_str = ", ".join(
                f"{doc_id}: {weight:.3f}"
                for doc_id, weight in top_weights
            )

            if len(self.similarity_weights) > 3:
                weights_str += ", ..."

            parts.append(f"similarity_weights={{ {weights_str} }}")

        return f"JobExecutionPerformance({', '.join(parts)})"

    @model_validator(mode="after")
    def validate_simulation_fields(self):
        simulation_fields = [
            self.std_running_time_sec,
            self.std_energy_mwh,
            self.similarity_weights,
            self.selected_running_time_sec_noise,
            self.selected_energy_use_mwh_noise,
        ]

        if self.simulated:
            # must ALL be present (not None)
            if any(v is None for v in simulation_fields):
                raise ValueError(
                    "When simulated=True, all simulation fields must be provided"
                )

        else:
            # must ALL be None
            if any(v is not None for v in simulation_fields):
                raise ValueError(
                    "When simulated=False, simulation fields must be None"
                )

        return self
