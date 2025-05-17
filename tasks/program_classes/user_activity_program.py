import os

from tasks.program_classes.abstract_program import ProgramInterface


class UserActivityProgram(ProgramInterface):
    def get_program_name(self):
        return "User Activity"

    def get_command(self) -> str:
        return f'python {os.path.join("tasks/UserActivity", "user_activity.py")} ' \
               f'"{os.path.join(self.results_path, "tasks_times.csv")}"'
