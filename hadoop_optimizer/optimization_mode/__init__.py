from enum import Enum


class OptimizationMode(str, Enum):
    CONTEXTUAL_BANDIT = "Contextual Bandit"
    RL = "RL"