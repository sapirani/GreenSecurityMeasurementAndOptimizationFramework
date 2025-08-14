from dataclasses import dataclass


@dataclass
class AggregationResult:
    pass


@dataclass
class EmptyAggregationResults(AggregationResult):
    pass
