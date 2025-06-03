# A cgroup is a feature that allows you to allocate, limit, and monitor system resources among user-defined groups of processes.
# Enables control over resource distribution, ensuring that no single group can monopolize the system resources.
import os

SYSTEM_CGROUP_FILE_PATH = r"/sys/fs/cgroup/"

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
CGROUP_CONTROLLERS_FILE_PATH = os.path.join(SYSTEM_CGROUP_FILE_PATH, CGROUP_CONTROLLERS_FILE_NAME)



class ProcCgroupFileConsts:
    NUMBER_OF_ELEMENTS = 3
    HIERARCHY_INDEX = 0
    CONTROLLERS_INDEX = 1
    CGROUP_PATH_INDEX = 2


class FileKeywords:
    V1 = "v1"
    V2 = "v2"
    USAGE_USEC = "usage_usec"
    MAX = "max"

    CGROUP_V1_CPU_IDENTIFIER = "cpu,cpuacct"
    CGROUP_V1_MEMORY_IDENTIFIER = "memory"
    CGROUP_V2_IDENTIFIER = "0"