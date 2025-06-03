import os

from resources_measurement.linux_resources.config import FileKeywords, CGROUP_CONTROLLERS_FILE_PATH, \
    CGROUP_IN_CONTAINER_PATH, ProcCgroupFileConsts, SYSTEM_CGROUP_FILE_PATH


def detect_cgroup_version() -> str:
    return FileKeywords.V2 if os.path.exists(CGROUP_CONTROLLERS_FILE_PATH) else FileKeywords.V1


def extract_cgroup_relative_path(version: str, v1_controllers: str) -> str:
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