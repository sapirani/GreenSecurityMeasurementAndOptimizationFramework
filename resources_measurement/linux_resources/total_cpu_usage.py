import os
import time

SYS_FS_CGROUP_PATH = r"/sys/fs/cgroup/"

CGROUP_CONTROLLERS_FILE_NAME = r"cgroup.controllers"
CGROUP_CONTROLLERS_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CGROUP_CONTROLLERS_FILE_NAME)

CPU_STATS_FILE_NAME = r"cpu.stat"
CPU_STATS_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_STATS_FILE_NAME)

CPU_ACCT_USAGE_FILE_NAME = r"cpuacct/cpuacct.usage"
CPU_ACCT_USAGE_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_ACCT_USAGE_FILE_NAME)

CPU_MAX_FILE_NAME = r"cpu.max"
CPU_MAX_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_MAX_FILE_NAME)

CPU_CF_QUOTA_FILE_NAME = r"cpu/cpu.cfs_quota_us"
CPU_CF_QUOTA_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_CF_QUOTA_FILE_NAME)

CPU_CF_PERIOD_FILE_NAME = r"cpu/cpu.cfs_period_us"
CPU_CF_PERIOD_FILE_PATH = os.path.join(SYS_FS_CGROUP_PATH, CPU_CF_PERIOD_FILE_NAME)


class FileKeywords:
    V1 = "v1"
    V2 = "v2"
    USAGE_USEC = "usage_usec"
    MAX = "max"


def detect_cgroup_version():
    if os.path.exists(CGROUP_CONTROLLERS_FILE_PATH):
        return FileKeywords.V2
    else:
        return FileKeywords.V1


def read_cpu_usage_ns(version):
    if version == FileKeywords.V2:
        with open(CPU_STATS_FILE_PATH) as f:
            for line in f:
                if line.startswith(FileKeywords.USAGE_USEC):
                    return int(line.split()[1]) * 1000  # convert to nanoseconds
    else:
        with open(CPU_ACCT_USAGE_FILE_PATH) as f:
            return int(f.read().strip())


def read_cpu_limit(version):
    if version == FileKeywords.V2:
        try:
            with open(CPU_MAX_FILE_PATH) as f:
                quota_str, period_str = f.read().strip().split()
                if quota_str == FileKeywords.MAX:
                    return None  # no limit
                return int(quota_str), int(period_str)
        except:
            return None
    else:
        try:
            with open(CPU_CF_QUOTA_FILE_PATH) as f:
                quota = int(f.read().strip())
            with open(CPU_CF_PERIOD_FILE_PATH) as f:
                period = int(f.read().strip())
            if quota == -1:
                return None
            return quota, period
        except:
            return None


def get_num_cpus_allowed(quota_period_tuple):
    if quota_period_tuple is None:
        return os.cpu_count()
    quota, period = quota_period_tuple
    return quota / period


def get_container_cpu_usage(interval=1.0):
    version = detect_cgroup_version()
    quota_period = read_cpu_limit(version)
    allowed_cpus = get_num_cpus_allowed(quota_period)

    usage1 = read_cpu_usage_ns(version)
    time.sleep(interval)
    usage2 = read_cpu_usage_ns(version)

    cpu_delta_ns = usage2 - usage1
    total_possible_ns = interval * 1e9 * allowed_cpus

    cpu_percent = (cpu_delta_ns / total_possible_ns) * 100
    return cpu_percent
