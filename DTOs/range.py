from dataclasses import dataclass


@dataclass
class Range:
    low: float
    high: float

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) cannot be greater than high ({self.high})")