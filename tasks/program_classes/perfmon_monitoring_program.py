from typing import Union

from tasks.program_classes.abstract_program import ProgramInterface


class PerfmonProgram(ProgramInterface):
    def __init__(self, program_name):
        super().__init__()
        self.results_path = None
        self.program_name = program_name

    def get_program_name(self) -> str:
        return "Performance Monitor"

    def get_process_name(self) -> str:
        pass

    def should_use_powershell(self) -> bool:
        return True

    def get_command(self) -> str:
        def process_counters_variables(process_id):
            # name = process_name.replace(".exe", "")

            return f"""$proc_id{process_id}={process_id}
            $proc_path{process_id}=((Get-Counter "\\Process(*)\\ID Process").CounterSamples | ? {{$_.RawValue -eq $proc_id{process_id}}}).Path
            $proc_base_path{process_id} = ($proc_path{process_id} -replace "\\\\id process$","")
            
            $io_read_bytes{process_id} = $proc_base_path{process_id} + "\\IO Read Bytes/sec"
            $io_write_bytes{process_id} = $proc_base_path{process_id} + "\\IO Write Bytes/sec"
            $io_read_operations{process_id} = $proc_base_path{process_id} + "\\IO Read Operations/sec"
            $io_write_operations{process_id} = $proc_base_path{process_id} + "\\IO Write Operations/sec"
            """

        def process_counters():
            counters = ""
            for process_id in self.processes_ids:
                counters += f'''
                $io_read_bytes{process_id},
                $io_write_bytes{process_id},
                $io_read_operations{process_id},
                $io_write_operations{process_id},
                '''
            return counters

        processes_vars = ""

        for process_id in self.processes_ids:
            processes_vars += process_counters_variables(process_id) + "\n"

        return f'''Get-Counter 
        {processes_vars}
        $gc = {process_counters()} "\\PhysicalDisk(_Total)\\Disk Reads/sec",
        "\\PhysicalDisk(_Total)\\Disk Writes/sec",
        "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec",
        "\\PhysicalDisk(_Total)\\Disk Write Bytes/sec",
        "\\Processor(_Total)\\% Processor Time", 
        "\\Power Meter(_Total)\\Power"
        Get-Counter -counter $gc -Continuous | Export-Counter -FileFormat "CSV" -Path "C:{self.results_path}\\perfmon.csv"'''

        # return f'Get-Counter gc = "\\PhysicalDisk(_Total)\\Disk Reads/sec", "\\PhysicalDisk(_Total)\\Disk Writes/sec", "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec", "\\PhysicalDisk(_Total)\\Disk Write Bytes/sec", "\\Processor(_Total)\\% Processor Time" Get-Counter -counter $gc -Continuous | Export-Counter -FileFormat "CSV" -Path "{self.results_path}\\perfmon.csv"'

    def find_child_id(self, p, is_posix) -> Union[int, None]:  # from python 3.10 - int | None:
        return None
