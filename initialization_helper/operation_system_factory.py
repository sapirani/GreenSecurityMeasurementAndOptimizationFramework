import platform

from operating_systems.abstract_operating_system import AbstractOSFuncs
from operating_systems.os_linux import LinuxOS
from operating_systems.os_windows import WindowsOS


def running_os_factory(is_inside_container: bool) -> AbstractOSFuncs:
    if platform.system() == "Linux":
        return LinuxOS(is_inside_container=is_inside_container)
    elif platform.system() == "Windows":
        return WindowsOS()

    raise Exception("Operating system is not supported")