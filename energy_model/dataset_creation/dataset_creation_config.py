from energy_model.configs.columns import SystemColumns, ProcessColumns
from utils.general_consts import MINUTE

DEFAULT_BATCH_INTERVAL_SECONDS = [5 * MINUTE]
MINIMAL_BATCH_DURATION = DEFAULT_BATCH_INTERVAL_SECONDS[0] * 0.5
TIMESTAMP_COLUMN_NAME = "timestamp"
IDLE_SESSION_ID_NAME = "idle"
COLUMNS_TO_AVOID_SUM = [SystemColumns.BATCH_ID_COL, ProcessColumns.PROCESS_ID_COL,
                        SystemColumns.ENERGY_USAGE_PER_SECOND_SYSTEM_COL, TIMESTAMP_COLUMN_NAME]


class AggregationName:
    SUM = "sum"
    FIRST_SAMPLE = "first"
    LAST_SAMPLE = "last"
