from typing import Optional

from DTOs.hadoop.job_execution_performance import JobExecutionPerformance


class RewardCalculator:
    def __init__(self, runtime_importance_factor: float = 0.5, energy_importance_factor: float = 0.5):
        self.baseline_performance: Optional[JobExecutionPerformance] = None
        self.runtime_importance_factor = runtime_importance_factor
        self.energy_importance_factor = energy_importance_factor

        self.last_job_performance = None
        self.last_reward = -10000.0

    def update_baseline_performance(self, job_performance: JobExecutionPerformance):
        self.baseline_performance = job_performance

    def compute_reward(self, job_performance: JobExecutionPerformance) -> float:
        self.last_job_performance = job_performance
        # TODO: IMPLEMENT THE OTHER PARTS IN THE REWARD FUNCTION (support terminated: bool, truncated: bool)
        runtime_improvement_sec = self.baseline_performance.running_time_sec - job_performance.running_time_sec
        runtime_improvement_percent = (runtime_improvement_sec / self.baseline_performance.running_time_sec) * 100

        # TODO: HANDLE LOW (and even zero) ENERGY CONSUMPTION BETTER
        if self.baseline_performance.energy_use_mwh > 1:
            energy_improvement_mwh = self.baseline_performance.energy_use_mwh - job_performance.energy_use_mwh
            energy_improvement_percent = (energy_improvement_mwh / self.baseline_performance.energy_use_mwh) * 100
        else:
            energy_improvement_percent = 0

        self.last_reward = self.runtime_importance_factor * runtime_improvement_percent + \
            self.energy_importance_factor * energy_improvement_percent

        return self.last_reward
