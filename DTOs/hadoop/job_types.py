from enum import Enum


class JobType(str, Enum):
    word_count = "word_count"
    monte_carlo_pi = "monte_carlo_pi"
    # TODO: ADD MORE SUPPORTED JOB TYPES
