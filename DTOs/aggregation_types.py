from enum import Enum


class AggregationType(str, Enum):
    CPUIntegral = "CPU Integral"
    ProcessSystemUsageFraction = "Process-System Usage Fraction"
    ProcessEnergyModelAggregator = "Process Energy Prediction Model"
    SystemEnergyModelAggregator = "System Energy Prediction Model"
