from gymnasium import spaces
from gymnasium.core import ObsType

from hadoop_optimizer.DTOs.hadoop_job_execution_config import HadoopJobExecutionConfig


class EnvironmentTruncatedException(Exception):
    def __init__(self, last_job_configuration: HadoopJobExecutionConfig, elapsed_steps: int, max_steps: int):
        self.last_job_configuration = last_job_configuration
        self.elapsed_steps = elapsed_steps
        self.max_steps = max_steps
        super().__init__("Environment truncated or step called incorrectly")


class StateNotReadyException(Exception):
    """Raised when the minimal required telemetry has not yet been retrieved by the DRL"""
    def __init__(self):
        super().__init__("minimal required telemetry is not yet available")


class OutOfBoundObservation(Exception):
    def __init__(self, observation: ObsType, space: spaces.Space):
        super().__init__(f"Observation out of bounds. Observation: {observation}, space: {space}")
