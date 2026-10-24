import os
from typing import Dict

import psutil

from DTOs.scanner_counters.system.disk_results import SystemDiskResults
from resource_usage_recorder.container_recorder.linux_resources.cgroup_versions.abstract_cgroup_version import \
    CgroupMetricReader
from utils.general_consts import KB, NANOSECONDS_IN_MILLISECONDS

CGROUP_V1_NAME = "V1"


class CgroupMetricReaderV1(CgroupMetricReader):
    __CGROUP_V1_CPU_CONTROLLERS = "cpu,cpuacct"
    __CGROUP_V1_MEMORY_CONTROLLERS = "memory"

    # Reports the current memory usage in the cgroup, in bytes.
    # Includes memory used by all descendant cgroups.
    # The format of this file is a single integer value in bytes.
    __MEMORY_USAGE_FILE_NAME_V1 = "memory/memory.usage_in_bytes"

    # Shows the memory limit set for the cgroup in bytes.
    # If set to a very large number (e.g., 9223372036854771712), it means no limit.
    # The format of this file is a single integer value in bytes.
    __MEMORY_MAX_FILE_NAME_V1 = "memory/memory.limit_in_bytes"

    # Reports the total CPU time consumed by tasks in the cgroup.
    # "acct" stands for "accounting" and refers to the mechanism for tracking resource usage.
    # Therefore, cpuacct.usage provides a report on the total CPU time used by all processes managed by the cgroup.
    # The format of the file is a single integer value representing nanoseconds.
    __CPU_ACCT_USAGE_FILE_NAME_V1 = r"cpuacct/cpuacct.usage"

    # Specifies the total available run-time within a period for tasks in the cgroup. - relevant for cgroup V1
    # A quota sets the maximum amount of CPU time that a cgroup can consume during each period.
    # The format of the file is a single integer value in microseconds (-1 indicates no limit).
    __CPU_CF_QUOTA_FILE_NAME = r"cpu/cpu.cfs_quota_us"

    # Defines the length of the period for enforcing CPU quotas. - relevant for cgroup V1
    # Within each period, the cgroup's CPU usage is limited according to its quota.
    # The format of the file is a single integer value in microseconds.
    __CPU_CF_PERIOD_FILE_NAME = r"cpu/cpu.cfs_period_us"

    # Reports the number of read and write I/O operations per device
    # for the cgroup, as tracked by the I/O throttling controller.
    # Each line contains:
    # major:minor operation value
    # Supported operations include Read, Write, Sync, and Async.
    __DISK_IO_SERVICED_FILE_NAME_V1 = "blkio.throttle.io_serviced"

    # Reports the total number of bytes read and written per device
    # for the cgroup, as tracked by the I/O throttling controller.
    # Each line contains:
    # major:minor operation value
    # Values are in bytes.
    __DISK_IO_SERVICE_BYTES_FILE_NAME_V1 = "blkio.throttle.io_service_bytes"

    # Reports the cumulative I/O service time per device and operation.
    # Each line contains:
    # major:minor operation value
    # Values are in nanoseconds and can be converted to milliseconds.
    # Availability depends on kernel configuration and I/O accounting support.
    __DISK_IO_SERVICE_TIME_FILE_NAME_V1 = "blkio.io_service_time"

    __DISK_IO_CGROUP_DIR_NAME = "blkio"

    __NO_QUOTA_LIMIT = -1

    def get_version(self) -> str:
        return CGROUP_V1_NAME

    def read_cpu_usage_ns(self) -> int:
        try:
            with open(self._cpu_usage_file_path) as f:
                return int(f.read().strip())
        except Exception as e:
            raise ValueError(f"The file {self._cpu_usage_file_path} does not exist or is not readable in Cgroup V1.")

    def _get_cpu_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__CPU_ACCT_USAGE_FILE_NAME_V1)

    def get_container_vcores(self) -> float:
        try:
            with open(os.path.join(self._base_cgroup_dir, self.__CPU_CF_QUOTA_FILE_NAME)) as f:
                quota = int(f.read().strip())
            with open(os.path.join(self._base_cgroup_dir, self.__CPU_CF_PERIOD_FILE_NAME)) as f:
                period = int(f.read().strip())
            if quota == self.__NO_QUOTA_LIMIT:
                return os.cpu_count()
            return quota / period
        except Exception as e:
            print(f"Warning: Cannot get number of containers due to error: {e}, Using cpu_count()")
            return os.cpu_count()

    def get_memory_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_USAGE_FILE_NAME_V1)

    def get_memory_limit_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_MAX_FILE_NAME_V1)

    def is_container_memory_limited(self, limit: str) -> bool:
        limit_number = int(limit)
        return limit_number >= 2 ** 60

    def get_disk_io_path(self) -> str:
        """Return the cgroups v1 disk I/O statistics path."""
        return os.path.join(self._base_cgroup_dir, self.__DISK_IO_SERVICED_FILE_NAME_V1)

    def read_stats(self, filename: str) -> Dict[str, int]:
        target = {"Read": 0, "Write": 0}
        path = os.path.join(self._base_cgroup_dir, self.__DISK_IO_CGROUP_DIR_NAME, filename)
        with open(path) as f:
            for line in f:
                parts = line.strip().split()

                # Expected format: major:minor operation value
                if len(parts) != 3:
                    continue

                operation = parts[1]
                if operation in target:
                    target[operation] += int(parts[2].strip())
        return target

    def get_total_disk_counters(self) -> SystemDiskResults:
        """
        Read cumulative disk I/O counters from cgroups v1.

        Counts and byte totals come from the blkio throttle statistics.
        Read/write times come from blkio.io_service_time and are converted
        from nanoseconds to milliseconds.

        Note: accurate per-operation timing may require additional instrumentation,
        depending on the kernel and scheduler configuration.
        Specifically, further implementation of this function may utilize eBPF,
        with the appropriate kernel capabilities and permissions (BPF and PERFMON).
        """

        time_stats = {"Read": 0, "Write": 0}

        stats = self.read_stats(self.__DISK_IO_SERVICED_FILE_NAME_V1)
        byte_stats = self.read_stats(self.__DISK_IO_SERVICE_BYTES_FILE_NAME_V1)
        try:
            time_stats = self.read_stats(self.__DISK_IO_SERVICE_TIME_FILE_NAME_V1)
            time_stats["Read"] //= NANOSECONDS_IN_MILLISECONDS
            time_stats["Write"] //= NANOSECONDS_IN_MILLISECONDS
        except FileNotFoundError:
            print(
                "Warning: blkio.io_service_time is unavailable, ensure the relevant schedulers are used "
                "(e.g., CFQ and BFQ); falling back to system-wide psutil counters."
            )
            psutil_counters = psutil.disk_io_counters()
            time_stats["Read"] = psutil_counters.read_time
            time_stats["Write"] = psutil_counters.write_time

        return SystemDiskResults(
            disk_read_count=stats["Read"],
            disk_write_count=stats["Write"],
            disk_read_kb=byte_stats["Read"] / KB,
            disk_write_kb=byte_stats["Write"] / KB,
            disk_read_time=time_stats["Read"],
            disk_write_time=time_stats["Write"],
        )