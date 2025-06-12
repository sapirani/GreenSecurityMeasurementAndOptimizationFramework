import os
from abc import ABC

from resources_measurement.linux_resources.cgroup_versions.abstract_cgroup_version import CgroupVersion
from resources_measurement.linux_resources.cgroup_versions.cgroup_v1 import CgroupV1
from resources_measurement.linux_resources.cgroup_versions.cgroup_v2 import CgroupV2
from resources_measurement.linux_resources.cgroup_versions.common_paths import SYSTEM_CGROUP_DIR_PATH

# Lists the available controllers (e.g., cpu, memory) that can be enabled in the current cgroup.
# The format of the file is a single line with space-separated controller names.
# E.g. cpu io memory
# Presence of this file indicates that the system is using cgroup v2.
CGROUP_CONTROLLERS_FILE_NAME = r"cgroup.controllers"
CGROUP_CONTROLLERS_FILE_PATH = os.path.join(SYSTEM_CGROUP_DIR_PATH, CGROUP_CONTROLLERS_FILE_NAME)


class LinuxContainerResourceReader(ABC):
    def __init__(self):
        self._version = self.__detect_cgroup_version()

    def __detect_cgroup_version(self) -> CgroupVersion:
        return CgroupV2() if os.path.exists(CGROUP_CONTROLLERS_FILE_PATH) else CgroupV1()
