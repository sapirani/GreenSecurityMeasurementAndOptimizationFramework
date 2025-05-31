import os
import time

def detect_cgroup_version():
    if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
        return "v2"
    else:
        return "v1"

def read_cpu_usage_ns(version):
    if version == "v2":
        with open("/sys/fs/cgroup/cpu.stat") as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1]) * 1000  # convert to nanoseconds
    else:
        with open("/sys/fs/cgroup/cpuacct/cpuacct.usage") as f:
            return int(f.read().strip())

def read_cpu_limit(version):
    if version == "v2":
        try:
            with open("/sys/fs/cgroup/cpu.max") as f:
                quota_str, period_str = f.read().strip().split()
                if quota_str == "max":
                    return None  # no limit
                return int(quota_str), int(period_str)
        except:
            return None
    else:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                quota = int(f.read().strip())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
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
