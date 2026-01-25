import math
from typing import Optional

from DTOs.hadoop.job_execution_performance import JobExecutionPerformance

LAST_REWARD_MIN_IMPORTANCE_FACTOR = 10


class RewardCalculator:
    """
    This reward calculator operates according to the following formula:
    gain_t = τ ⋅ (E_baseline− E_t) / E_baseline   +   δ ⋅ (T_baseline − T_t) / T_baseline

    r_t^intermediate = α ⋅ tanh( β ⋅ gain_t ) − ϵ
    r_t^(optimized−config) = λ ⋅ ( e^(gain_t ) − 1)

    r_t = [ 1 − 1(t=T) ] ⋅ r_t^intermediate + 1(t=T) ⋅ r_t^(optimized−config)
    Where: λ≫ ϵ > α
    """
    def __init__(
            self,
            alpha_hyperparam: float,
            beta_hyperparam: float,
            lambda_hyperparam: float,
            epsilon_hyperparam: float,
            tau_hyperparam: float = 0.5,
            delta_hyperparam: float = 0.5,
    ):
        if not (alpha_hyperparam > 0 and beta_hyperparam > 0 and lambda_hyperparam > 0 and epsilon_hyperparam > 0 and
                tau_hyperparam > 0 and delta_hyperparam > 0):
            raise ValueError("Selected reward hyper parameters must be positive")

        if not (lambda_hyperparam > LAST_REWARD_MIN_IMPORTANCE_FACTOR * epsilon_hyperparam > alpha_hyperparam):
            raise ValueError("Selected reward hyper parameters correlation violate constraints")

        self.baseline_performance: Optional[JobExecutionPerformance] = None
        self.__alpha = alpha_hyperparam
        self.__beta = beta_hyperparam
        self.__lambda = lambda_hyperparam
        self.__epsilon = epsilon_hyperparam
        self.__tau = tau_hyperparam
        self.__delta = delta_hyperparam

    def update_baseline_performance(self, job_performance: JobExecutionPerformance):
        self.baseline_performance = job_performance

    def __compute_step_gain(self, job_performance: JobExecutionPerformance) -> float:
        runtime_improvement_sec = self.baseline_performance.running_time_sec - job_performance.running_time_sec
        runtime_gain = (runtime_improvement_sec / self.baseline_performance.running_time_sec)

        # TODO: HANDLE LOW (and even zero) ENERGY CONSUMPTION BETTER
        if self.baseline_performance.energy_use_mwh > 1:
            energy_improvement_mwh = self.baseline_performance.energy_use_mwh - job_performance.energy_use_mwh
            energy_gain = (energy_improvement_mwh / self.baseline_performance.energy_use_mwh)
        else:
            # TODO: SUPPORT A REGULAR LOGGER IN A SEPARATE INDEX IN ELASTIC
            print("Warning!!! observed too small energy consumption for job execution")
            energy_gain = 0

        return self.__delta * runtime_gain + self.__tau * energy_gain

    def compute_reward(self, job_performance: Optional[JobExecutionPerformance], is_last_step: bool) -> float:
        step_gain = self.__compute_step_gain(job_performance)
        is_last_step = int(is_last_step)

        reward_shaping = self.__alpha * math.tanh(self.__beta * step_gain)
        negative_intermediate_reward = (1 - is_last_step) * (reward_shaping - self.__epsilon)

        last_step_reward = is_last_step * self.__lambda * (math.exp(step_gain) - 1)

        return negative_intermediate_reward + last_step_reward
