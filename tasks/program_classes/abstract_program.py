import time
from typing import Union

import psutil

from utils.general_consts import SYSTEM_IDLE_PID


class ProgramInterface:
    def __init__(self):
        self.results_path = None
        self.processes_ids = None

    def get_program_name(self) -> str:
        pass

    def get_process_name(self) -> str:
        pass

    def get_command(self) -> str:
        pass

    def on_exit(self):
        pass

    def path_adjustments(self) -> str:
        return ""

    def should_find_child_id(self):
        return False

    def should_use_powershell(self) -> bool:
        return False

    def set_results_dir(self, results_path):
        self.results_path = results_path

    def set_processes_ids(self, processes_ids):
        self.processes_ids = processes_ids

    def process_ignore_cond(self, p):
        return p.pid == SYSTEM_IDLE_PID

    def find_child_id(self, p, is_posix) -> Union[int, None]:
        try:
            children = []
            process_pid = p.pid
            for i in range(600):
                children = psutil.Process(process_pid).children()
                if len(children) != 0:
                    break
                time.sleep(0.1)

        except psutil.NoSuchProcess:
            return None

        if len(children) != 1:
            return None

        return children[0].pid
