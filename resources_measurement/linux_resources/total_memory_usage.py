import psutil

from resources_measurement.linux_resources.total_resource_usage import LinuxContainerResourceReader


class LinuxContainerMemoryReader(LinuxContainerResourceReader):
    def __init__(self):
        super().__init__()
        self.__memory_limit = self.__get_memory_limit_bytes()
        self.__memory_usage_path = self._version.get_memory_usage_path()

    def get_memory_usage_bytes(self) -> int:
        try:
            with open(self.__memory_usage_path) as f:
                return int(f.read().strip())
        except Exception as e:
            print(f"Error reading memory usage: {e}")
            return 0

    def get_memory_usage_percent(self) -> float:
        usage = self.get_memory_usage_bytes()
        if self.__memory_limit <= 0:
            return 0.0  # TODO: check if this is the right value
        return (usage / self.__memory_limit) * 100

    def __get_host_memory_limit(self) -> int:
        try:
            return psutil.virtual_memory().total
        except Exception as e:
            print(f"Error reading host memory: {e}")
        return 1  # Avoid division by zero

    def __get_memory_limit_bytes(self) -> int:
        max_memory_file_path = self._version.get_memory_limit_path()
        try:
            with open(max_memory_file_path) as max_memory_file:
                max_val = max_memory_file.read().strip()
                if self._version.should_get_host_memory_limit():
                    return self.__get_host_memory_limit()
                return int(max_val)


        except ValueError as e:
            print(f"Unexpected format in memory limit file in path {max_memory_file_path}: {max_val}")
            return self.__get_host_memory_limit()
