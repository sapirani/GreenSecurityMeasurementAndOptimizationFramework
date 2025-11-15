from enum import Enum


class FeatureSelectionType(Enum):
    PROCESS_ONLY = 1
    SYSTEM_ONLY = 2
    PROCESS_AND_SYSTEM_ONLY = 3
    PROCESS_AND_HARDWARE_ONLY = 4
