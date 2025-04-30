from tasks.program_classes.abstract_program import ProgramInterface


class PythonServer(ProgramInterface):
    def get_program_name(self):
        return "Python Server"

    def get_command(self) -> str:
        return f"python -m http.server"
