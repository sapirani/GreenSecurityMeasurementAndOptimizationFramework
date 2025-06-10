import os
from abc import ABC, abstractmethod

from resources_measurement.linux_resources.cgroup_versions.abstract_cgroup_version import detect_cgroup_version, \
    CgroupVersion
from resources_measurement.linux_resources.cgroup_versions.cgroup_v1 import CgroupV1
from resources_measurement.linux_resources.cgroup_versions.cgroup_v2 import CgroupV2
from resources_measurement.linux_resources.config import CGROUP_CONTROLLERS_FILE_PATH


class LinuxContainerResourceReader(ABC):
    def __init__(self):
        self._version = detect_cgroup_version()
        # self._resource_usage_path = self._get_resource_file_path(self._version)

    def detect_cgroup_version(self) -> CgroupVersion:
        return CgroupV2() if os.path.exists(CGROUP_CONTROLLERS_FILE_PATH) else CgroupV1()

    def extract_cgroup_relative_path(self, version: str, v1_controllers: str) -> str:
        with open(CGROUP_IN_CONTAINER_PATH, "r") as f:
            for line in f:
                proc_cgroup_parts = line.strip().split(":")
                if len(proc_cgroup_parts) != ProcCgroupFileConsts.NUMBER_OF_ELEMENTS:
                    continue
                else:
                    hierarchy = proc_cgroup_parts[ProcCgroupFileConsts.HIERARCHY_INDEX]
                    controllers = proc_cgroup_parts[ProcCgroupFileConsts.CONTROLLERS_INDEX]
                    cgroup_path = proc_cgroup_parts[ProcCgroupFileConsts.CGROUP_PATH_INDEX].lstrip("/")

                    if (version == FileKeywords.V2 and hierarchy == FileKeywords.CGROUP_V2_IDENTIFIER) or \
                            (version == FileKeywords.V1 and controllers == v1_controllers):
                        return os.path.join(SYSTEM_CGROUP_FILE_PATH, cgroup_path)

        return SYSTEM_CGROUP_FILE_PATH