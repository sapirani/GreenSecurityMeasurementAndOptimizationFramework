from tasks.program_classes.abstract_program import ProgramInterface


class NoScanProgram(ProgramInterface):
    def get_program_name(self) -> str:
        return "No Scan"
