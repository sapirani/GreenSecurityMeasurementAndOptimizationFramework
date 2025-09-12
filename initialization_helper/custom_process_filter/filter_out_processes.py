"""
This module is an example for user-defined process filtering.
You can write your own filter, which will be applied to any candidate process for measuring.
You must implement the AbstractProcessFilter in each filter you perform.

To use this functionality, create your own module (a Python file) inside this directory,
and reference that module within the program_parameters.py file.

Note: each class that implements AbstractProcessFilter in your module will automatically be applied as a filter.
If at least one of the filters return True, the process will not be measured.
For example, in this module both python and cmd.exe process will be filtered out,
and will not appear in measurement results.
"""


import psutil
from initialization_helper.custom_process_filter.abstarct_process_filter import AbstractProcessFilter


class FilterOutPythonProcesses(AbstractProcessFilter):
    def should_ignore_process(self, process: psutil.Process) -> bool:
        return "python" in process.name()


class FilterOutCMDProcesses(AbstractProcessFilter):
    def should_ignore_process(self, process: psutil.Process) -> bool:
        return process.name() == "cmd.exe"
