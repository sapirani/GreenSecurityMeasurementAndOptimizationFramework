import os
import re
import time
from typing import Union

import psutil

from os_funcs import OSFuncsInterface
from tasks.program_classes.abstract_program import ProgramInterface


class SplunkProgram(ProgramInterface):
    def get_program_name(self):
        return "Splunk Enterprise SIEM"


    def process_ignore_cond(self, p):
        return super(SplunkProgram, self).process_ignore_cond(p) or (not p.name().__contains__('splunk'))

    # def should_use_powershell(self) -> bool:
    #     return True
    # def should_find_child_id(self) -> bool:
    #     return True

    # def find_child_id(self, p, is_posix) -> Union[int, None]:  # from python 3.10 - int | None:
    #     try:
    #         children = None
    #         # time.sleep(40)
    #         p.wait()
    #         result = OSFuncsInterface.run("splunk status", self.should_use_powershell(), is_posix=is_posix)
    #         print(result)

    #         stdout = result.stdout.decode('utf-8')
    #         if is_posix:
    #             run_match = re.search(fr'(PID:\s*(\d+))', stdout)
    #             print(run_match)
    #         else:
    #             run_match = re.search(fr'(pid\s*(\d+))', stdout)

    #         # stop_match_linux = re.search(f'splunkd is not running.', stdout)
    #         # stop_match_windows = re.search(f'Splunkd: Stopped', stdout)
    #         if run_match:
    #             children = int(run_match.group(1).split()[1])
    #             print(f"pid: {children}")
    #             return children
    #         else:
    #             raise Exception("Splunk didn't started correctly!")
    #     except psutil.NoSuchProcess:
    #         return None

    # def should_use_powershell(self) -> bool:
    #     return False
