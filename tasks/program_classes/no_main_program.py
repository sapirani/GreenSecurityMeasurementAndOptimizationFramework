from tasks.program_classes.abstract_program import ProgramInterface


class NoMainProgram(ProgramInterface):
    def get_program_name(self) -> str:
        return "No Main Program"
