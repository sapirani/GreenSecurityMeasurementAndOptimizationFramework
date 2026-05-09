from enum import Enum

TERMINATE_ACTION_NAME = "terminate"
CURRENT_JOB_CONFIG_KEY = "current_job_config"
NEXT_JOB_CONFIG_KEY = "next_job_config"
JOB_PROPERTIES_KEY = "job_properties"
DEFAULT_JOB_CONFIG_KEY = "default_config"
ELAPSED_STEPS_KEY = "elapsed_steps"
MAX_STEPS_KEY = "max_steps"


class RenderMode(str, Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    RGB_ARRAY_LIST = "rgb_array_list"
    ANSI = "ansi"
