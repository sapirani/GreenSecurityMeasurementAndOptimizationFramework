from typing import Optional

from gymnasium import spaces
from gymnasium.core import ObsType

from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig


class EnvironmentTruncatedException(Exception):
    """
    Raised when a truncation condition outside the scope of the MDP is satisfied.
    It means that the episode should terminate before reaching a terminal state (contrary to the definition of MDP),
     typically due to a timelimit.
    """
    def __init__(
            self,
            last_job_configuration: HadoopJobExecutionConfig,
            elapsed_steps: int,
            max_steps: int,
            truncation_description: Optional[str] = "",
    ):
        self.last_job_configuration = last_job_configuration
        self.elapsed_steps = elapsed_steps
        self.max_steps = max_steps
        self.truncation_description = truncation_description

        error_message = f"Environment truncated or step called incorrectly"
        error_message += f". {self.truncation_description}" if self.truncation_description else ""
        super().__init__(error_message)


class StateNotReadyException(Exception):
    """Raised when the minimal required telemetry has not yet been retrieved by the DRL"""
    def __init__(self):
        super().__init__("minimal required telemetry is not yet available")


class OutOfBoundObservation(Exception):
    def __init__(self, observation: ObsType, space: spaces.Space):
        super().__init__(f"Observation (i.e., state) is out of bounds. Observation: {observation}, space: {space}")
