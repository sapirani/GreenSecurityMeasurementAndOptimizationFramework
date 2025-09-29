import subprocess
import sys
import re
from pathlib import Path
from typing import Optional

from utils.general_consts import ProgramToScan

SESSION_ID_SCANNER_FLAG = "--measurement_session_id"


def run_scanner(scanner_path: str, session_id: str):
    subprocess.run([sys.executable, scanner_path, SESSION_ID_SCANNER_FLAG, session_id], check=True)


def update_parameter(current_content: str, field_name: str, field_value: str) -> str:
    """
    This method finds specific parameter in the content and replace its value with the given value.
    The regex finds the pattern that matches the following:
    field_name(possible spaces)=(possible spaces)(any possible value)end_of_line

    Input:
        current_content: content to be updated
        field_name: the name of the parameter
        field_value: the new value of the parameter
    Output:
        updated_content: content with the new value for the specific parameter
    """
    new_content = re.sub(
        fr'^{field_name}\s*=\s*.*$',
        f'{field_name} = {field_value}',
        current_content,
        flags=re.MULTILINE
    )
    return new_content


def update_multiple_parameters(program_parameters_file_path: str,
                               field_names_and_values: list[tuple[str, Optional[str]]]):
    program_parameters_file = Path(program_parameters_file_path)
    content = program_parameters_file.read_text()

    for field_name, field_value in field_names_and_values:
        if field_value is not None:
            content = update_parameter(content, field_name, field_value)

    program_parameters_file.write_text(content)


def update_main_program(program_parameters_file_path: str,
                        main_program_value: ProgramToScan):
    update_multiple_parameters(program_parameters_file_path,
                               [("main_program_to_scan", f"ProgramToScan.{main_program_value.name}")])


def update_dummy_task_values(program_parameters_file_path: str,
                             rate: Optional[float] = None, size: Optional[int] = None):
    rate_value = f"{rate}" if rate is not None else None
    size_value = f"{size}" if size is not None else None
    update_multiple_parameters(program_parameters_file_path, [
        ("dummy_task_rate", rate_value),
        ("dummy_task_unit_size", size_value)
    ])
