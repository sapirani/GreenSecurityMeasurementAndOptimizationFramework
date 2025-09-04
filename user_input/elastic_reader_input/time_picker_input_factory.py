from consts import TimePickerInputStrategy
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput
from user_input.elastic_reader_input.cli_time_picker import CLITimePicker


def get_time_picker_input(
        time_picker_input_strategy: TimePickerInputStrategy,
        preconfigured_time_input: TimePickerChosenInput
) -> TimePickerChosenInput:
    if time_picker_input_strategy == TimePickerInputStrategy.FROM_CONFIGURATION:
        return preconfigured_time_input

    if time_picker_input_strategy == TimePickerInputStrategy.GUI:
        from user_input.elastic_reader_input.GUI.gui_date_picker import GUITimePicker
        time_picker = GUITimePicker()
    elif time_picker_input_strategy == TimePickerInputStrategy.CLI:
        time_picker = CLITimePicker()
    else:
        raise ValueError("Time picker input strategy is not supported!")

    return time_picker.get_input()
