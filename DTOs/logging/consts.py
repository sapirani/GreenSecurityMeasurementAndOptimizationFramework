from enum import Enum


class LoggerName:
    SYSTEM_METRICS = "system_metrics"
    PROCESS_METRICS = "process_metrics"
    APPLICATION_FLOW = "application_flow"
    METRICS_AGGREGATIONS = "metrics_aggregations"
    DRL_TRAINING = "drl_training"
    DRL_DEBUGGING = "drl_debugging"


class IndexName(str, Enum):
    SYSTEM_METRICS = "system_metrics"
    PROCESS_METRICS = "process_metrics"
    APPLICATION_FLOW = "application_flow"
    METRICS_AGGREGATIONS = "metrics_aggregations"
    DRL_TRAINING = "drl_training"
    DRL_DEBUGGING = "drl_debugging"


SCANNER_FINISHED_MESSAGE = "The scanner has finished measuring"
