from consts import TimePickerInputStrategy
from user_input.GUI.gui_date_picker import GUITimePicker
from user_input.abstract_date_picker import TimePickerChosenInput
from user_input.cli_time_picker import CLITimePicker


def get_time_picker_input(time_picker_input_strategy: TimePickerInputStrategy) -> TimePickerChosenInput:
    if time_picker_input_strategy == TimePickerInputStrategy.GUI:
        time_picker = GUITimePicker()
    elif time_picker_input_strategy == TimePickerInputStrategy.CLI:
        time_picker = CLITimePicker()
    else:
        raise ValueError("Time picker input strategy is not supported!")

    return time_picker.get_input()
