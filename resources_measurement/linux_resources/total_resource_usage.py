import os
from abc import ABC

from resources_measurement.linux_resources.cgroup_versions.abstract_cgroup_version import CgroupVersion
from resources_measurement.linux_resources.cgroup_versions.cgroup_v1 import CgroupV1, CGROUP_V1_NAME, \
    CGROUP_V1_MEMORY_CONTROLLERS, CGROUP_V1_CPU_CONTROLLERS
from resources_measurement.linux_resources.cgroup_versions.cgroup_v2 import CgroupV2, CGROUP_V2_NAME, \
    CGROUP_V2_IDENTIFIER

# A cgroup is a feature that allows you to allocate, limit, and monitor system resources among user-defined groups of processes.
# Enables control over resource distribution, ensuring that no single group can monopolize the system resources.
SYSTEM_CGROUP_DIR_PATH = r"/sys/fs/cgroup/"

# Contains details on the cgroup of the container.
# The file format is the single line:
# hierarchy-ID : controllers : cgroup-path -> WHERE:
# Hierarchy ID can be 0 (for cgroup v2), or 2, 3, etc. in v1
# Controller(s)	can be cpu,cpuacct or empty string ("") for v2
# Path to cgroup can be /docker/<container-id> or /
CGROUP_IN_CONTAINER_PATH = r"/proc/self/cgroup"

# Lists the available controllers (e.g., cpu, memory) that can be enabled in the current cgroup.
# The format of the file is a single line with space-separated controller names.
# E.g. cpu io memory
# Presence of this file indicates that the system is using cgroup v2.
CGROUP_CONTROLLERS_FILE_NAME = r"cgroup.controllers"
CGROUP_CONTROLLERS_FILE_PATH = os.path.join(SYSTEM_CGROUP_DIR_PATH, CGROUP_CONTROLLERS_FILE_NAME)



class ProcCgroupFileConsts:
    NUMBER_OF_ELEMENTS = 3
    HIERARCHY_INDEX = 0
    CONTROLLERS_INDEX = 1
    CGROUP_PATH_INDEX = 2


class LinuxContainerResourceReader(ABC):
    def __init__(self):
        self._version = self.__detect_cgroup_version()

    def __detect_cgroup_version(self) -> CgroupVersion:
        cgroup_dir = self.__get_cgroup_relative_path(self._version.get_version())
        return CgroupV2(cgroup_dir=cgroup_dir) if os.path.exists(CGROUP_CONTROLLERS_FILE_PATH) else CgroupV1(
            cgroup_dir=cgroup_dir)

    def __get_cgroup_relative_path(self, version: str) -> str:
        with open(CGROUP_IN_CONTAINER_PATH, "r") as f:
            for line in f:
                proc_cgroup_parts = line.strip().split(":")
                if len(proc_cgroup_parts) != ProcCgroupFileConsts.NUMBER_OF_ELEMENTS:
                    continue
                else:
                    hierarchy = proc_cgroup_parts[ProcCgroupFileConsts.HIERARCHY_INDEX]
                    controllers = proc_cgroup_parts[ProcCgroupFileConsts.CONTROLLERS_INDEX]
                    cgroup_path = proc_cgroup_parts[ProcCgroupFileConsts.CGROUP_PATH_INDEX].lstrip("/")

                    if (version == CGROUP_V2_NAME and hierarchy == CGROUP_V2_IDENTIFIER) or \
                            (version == CGROUP_V1_NAME and
                             (controllers == CGROUP_V1_MEMORY_CONTROLLERS or
                              controllers == CGROUP_V1_CPU_CONTROLLERS)):
                        return os.path.join(SYSTEM_CGROUP_DIR_PATH, cgroup_path)

        return SYSTEM_CGROUP_DIR_PATH
