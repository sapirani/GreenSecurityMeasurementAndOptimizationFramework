import os
os.environ["KIVY_LOG_MODE"] = "PYTHON"

# must come after setting KIVY_LOG_MODE (env variable)
from elastic_reader_user_input.GUI.date_range_gui import ModeApp
from elastic_reader_user_input.abstract_date_picker import AbstractTimePicker, TimePickerChosenInput


class GUITimePicker(AbstractTimePicker):

    def _inner_get_input(self) -> TimePickerChosenInput:
        app = ModeApp()
        app.run()

        return TimePickerChosenInput(
            mode=app.selected_mode,
            start=app.selected_start_datetime,
            end=app.selected_end_datetime
        )
