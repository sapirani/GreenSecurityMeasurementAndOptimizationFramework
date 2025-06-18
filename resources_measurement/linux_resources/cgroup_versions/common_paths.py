# A cgroup is a feature that allows you to allocate, limit, and monitor system resources among user-defined groups of processes.
# Enables control over resource distribution, ensuring that no single group can monopolize the system resources.
SYSTEM_CGROUP_DIR_PATH: str = r"/sys/fs/cgroup/"

CPUSET_CPUS_FILE_NAME: str = "cpuset.cpus"