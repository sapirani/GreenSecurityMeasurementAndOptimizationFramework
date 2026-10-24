import os
from typing import Dict
from DTOs.scanner_counters.system.disk_results import SystemDiskResults
from resource_usage_recorder.container_recorder.linux_resources.cgroup_versions.abstract_cgroup_version import \
    CgroupMetricReader
from utils.general_consts import KB, MICROSECONDS_IN_MILLISECONDS

CGROUP_V2_NAME = "V2"



class CgroupMetricReaderV2(CgroupMetricReader):
    __USAGE_USEC_V2 = "usage_usec"
    __NO_MEMORY_LIMIT = "max"

    __CGROUP_V2_IDENTIFIER = "0"

    # Reports the current memory usage of the cgroup in bytes,
    # including memory used by all processes in the cgroup and its descendants.
    # The file format is a single integer representing bytes of memory currently used.
    __MEMORY_USAGE_FILE_NAME_V2 = "memory.current"

    # Sets Memory usage limits for the cgroup.
    # The format of the file is either a single integer (in bytes) or the string max.
    # <max>: Maximum memory bytes that the cgroup can use.
    # If <max> is set to max, there is no memory limit.
    __MEMORY_MAX_FILE_NAME_V2 = "memory.max"

    # Provides CPU usage statistics for the cgroup.
    # The format of the file is key-value pairs, one per line.
    # E.g.
    #   usage_usec 123456789
    #   user_usec 12345678
    #   system_usec 11111111
    # The interesting key is usage_usec, its value is the total CPU time consumed by all tasks in the cgroup, in microseconds.
    __CPU_STATS_FILE_NAME_V2 = "cpu.stat"

    # Sets CPU usage limits for the cgroup. - relevant for cgroup V2
    # The format of the file is two values separated by a space: <max> <period>
    # <max>: Maximum CPU time (in microseconds) that the cgroup can use in each period.
    # <period>: Length of each period in microseconds.
    # If <max> is set to max, there is no CPU limit.
    __CPU_MAX_FILE_NAME = r"cpu.max"

    # Provides disk I/O statistics for the cgroup. - relevant for cgroup V2
    # Each line represents a block device, identified by its major:minor device number.
    # The format is: <major>:<minor> rbytes=<value> wbytes=<value> rios=<value> wios=<value> [dbytes=<value> dios=<value>]
    # rbytes: Total bytes read.
    # wbytes: Total bytes written.
    # rios: Total number of read I/O operations.
    # wios: Total number of write I/O operations.
    # dbytes: Total bytes discarded (if supported).
    # dios: Total number of discard I/O operations (if supported).
    __DISK_IO_FILE_NAME_V2 = "io.stat"

    # Provides I/O pressure statistics for the cgroup. - relevant for cgroup V2
    # Reports time during which tasks are stalled waiting for I/O.
    # The file contains "some" and "full" lines, each with the following fields:
    # avg10: Average I/O pressure over the last 10 seconds, as a percentage.
    # avg60: Average I/O pressure over the last 60 seconds, as a percentage.
    # avg300: Average I/O pressure over the last 300 seconds, as a percentage.
    # total: Cumulative I/O stall time in microseconds.
    # some: Time during which at least some tasks were stalled on I/O.
    # full: Time during which all non-idle tasks were simultaneously stalled on I/O.
    __DISK_PRESSURE_FILE_NAME_V2 = "io.pressure"

    __NO_QUOTA_LIMIT = "max"

    def __init__(self):
        super().__init__()
        self._prev_pressure_time = self._get_disk_io_pressure_total()
        io_stats_counters = self._get_io_stats_total()
        self._prev_read_ops = io_stats_counters["rios"]
        self._prev_write_ops = io_stats_counters["wios"]
        prev_ops = self._prev_read_ops + self._prev_write_ops

        if prev_ops > 0:
            self._prev_read_time = self._prev_pressure_time * io_stats_counters["rios"] /  prev_ops
            self._prev_write_time = self._prev_pressure_time * io_stats_counters["wios"] /  prev_ops
        else:
            self._prev_read_time = 0
            self._prev_write_time = 0

    def get_version(self) -> str:
        return CGROUP_V2_NAME

    def read_cpu_usage_ns(self) -> int:
        try:
            with open(self._cpu_usage_file_path) as f:
                for line in f:
                    if line.startswith(self.__USAGE_USEC_V2):
                        return int(line.split()[1]) * 1000  # convert to nanoseconds 143512538

            return 0
        except Exception as e:
            raise ValueError(f"The file {self._cpu_usage_file_path} does not exist or is not readable in Cgroup V2.")

    def _get_cpu_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__CPU_STATS_FILE_NAME_V2)

    def get_container_vcores(self) -> float:
        try:
            with open(os.path.join(self._base_cgroup_dir, self.__CPU_MAX_FILE_NAME)) as f:
                quota_str, period_str = f.read().strip().split()
                if quota_str == self.__NO_QUOTA_LIMIT:
                    return os.cpu_count()  # no limit
                return float(quota_str) / float(period_str)
        except Exception as e:
            print(f"Warning: Cannot get number of containers due to error: {e}, Using cpu_count()")
            return os.cpu_count()

    def get_memory_usage_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_USAGE_FILE_NAME_V2)

    def get_memory_limit_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__MEMORY_MAX_FILE_NAME_V2)

    def get_disk_io_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__DISK_IO_FILE_NAME_V2)

    def get_disk_io_pressure_path(self) -> str:
        return os.path.join(self._base_cgroup_dir, self.__DISK_PRESSURE_FILE_NAME_V2)

    def is_container_memory_limited(self, limit: str) -> bool:
        return limit == self.__NO_MEMORY_LIMIT

    def _get_io_stats_total(self) -> Dict[str, int]:
        io_stats_counters = {
            "rios": 0,
            "wios": 0,
            "rbytes": 0,
            "wbytes": 0,
        }

        with open(self.get_disk_io_path()) as f:
            for line in f:
                parts = line.split()

                # Skip the device ID (major:minor).
                for field in parts[1:]:
                    key, value = field.split("=", 1)
                    if key in io_stats_counters:
                        io_stats_counters[key] += int(value)

        return io_stats_counters

    def _get_disk_io_pressure_total(self) -> float:
        with open(os.path.join(self._base_cgroup_dir, self.__DISK_PRESSURE_FILE_NAME_V2)) as f:
            for line in f:
                parts = line.split()

                if parts and parts[0] == "some":
                    for field in parts[1:]:
                        key, value = field.split("=", 1)
                        if key == "total":
                            return int(value) / MICROSECONDS_IN_MILLISECONDS

        raise RuntimeError(f"Could not find 'some' total in {self.__DISK_PRESSURE_FILE_NAME_V2}")

    def get_total_disk_counters(self) -> SystemDiskResults:
        """
        Disk read and write times are not directly available through the standard
        cgroup v2 I/O statistics.

        This function uses the cumulative I/O stall time reported by io.pressure
        as a proxy for disk read and write times.

        The "some" total measures the cumulative time during which at least one
        task in the cgroup was stalled on I/O. It does not distinguish between
        read and write operations.

        To estimate read and write times, the increase in I/O stall time between
        samples is distributed proportionally to the number of read and write
        operations observed during the same interval.

        These values are estimates and do not represent actual read/write service
        times. Accurate per-operation timing would require additional instrumentation,
        such as eBPF, with the appropriate kernel capabilities and permissions (BPF and PERFMON).
        """
        io_stats_counters = self._get_io_stats_total()

        total_pressure_time = self._get_disk_io_pressure_total()
        added_pressure_time = total_pressure_time - self._prev_pressure_time

        added_read_ops = io_stats_counters["rios"] - self._prev_read_ops
        added_write_ops = io_stats_counters["wios"] - self._prev_write_ops
        added_ops = added_read_ops + added_write_ops

        if added_ops > 0:
            added_read_time = int(added_pressure_time * added_read_ops / added_ops)
            added_write_time = int(added_pressure_time) - added_read_time
        else:
            added_read_time = 0
            added_write_time = 0

        self._prev_read_ops = io_stats_counters["rios"]
        self._prev_write_ops = io_stats_counters["wios"]
        self._prev_read_time += added_read_time
        self._prev_write_time += added_write_time
        self._prev_pressure_time = total_pressure_time

        return SystemDiskResults(
            disk_read_count=io_stats_counters["rios"],
            disk_write_count=io_stats_counters["wios"],
            disk_read_kb=io_stats_counters["rbytes"] / KB,
            disk_write_kb=io_stats_counters["wbytes"] / KB,
            disk_read_time=int(self._prev_read_time),
            disk_write_time=int(self._prev_write_time),
        )
