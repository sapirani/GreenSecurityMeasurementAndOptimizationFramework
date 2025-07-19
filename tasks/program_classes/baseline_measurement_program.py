from tasks.program_classes.abstract_program import ProgramInterface


class BaselineMeasurementProgram(ProgramInterface):
    def get_program_name(self) -> str:
        return "Baseline Measurement Program"
