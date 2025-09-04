import os
os.environ["KIVY_LOG_MODE"] = "PYTHON"

# must come after setting KIVY_LOG_MODE (env variable)
from user_input.elastic_reader_input.GUI.date_range_gui import ModeApp
from user_input.elastic_reader_input.abstract_date_picker import AbstractTimePicker, TimePickerChosenInput


class GUITimePicker(AbstractTimePicker):

    def _inner_get_input(self) -> TimePickerChosenInput:
        app = ModeApp()
        app.run()

        return TimePickerChosenInput(
            mode=app.selected_mode,
            start=app.selected_start_datetime,
            end=app.selected_end_datetime
        )
