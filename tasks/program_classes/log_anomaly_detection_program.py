from tasks.program_classes.abstract_program import ProgramInterface


class LogAnomalyDetection(ProgramInterface):
    def __init__(self, model_name, action, script_relative_path, installation_dir):
        super().__init__()
        self.installation_dir = installation_dir
        self.model_name = model_name
        self.script_relative_path = script_relative_path
        self.action = action

    def get_program_name(self):
        return "LogAnomalyDetection"

    def get_command(self):
        return rf"& python -m '{self.script_relative_path}' {self.action}"
