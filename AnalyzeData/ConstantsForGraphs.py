class AxisInfo:
    def __init__(self, label, unit, axis):
        self.axis = axis
        self.label = label
        self.unit = unit

class Units:
    TIME = "Seconds"
    PERCENT = "% out of 100"
    CAPACITY = "mWatt/hour"
    VOLTAGE = "mVolt"
    COUNT = "#"
    MEMORY_TOTAL = "GB"
    MEMORY_PROCESS = "MB"
    IO_BYTES = "KB"


DEFAULT = "default"