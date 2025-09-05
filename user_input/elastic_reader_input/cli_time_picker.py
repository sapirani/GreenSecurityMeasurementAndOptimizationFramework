from datetime import datetime, timezone

from user_input.elastic_reader_input.abstract_date_picker import AbstractTimePicker, TimePickerChosenInput, ReadingMode


class CLITimePicker(AbstractTimePicker):
    @staticmethod
    def __choose_mode() -> ReadingMode:
        valid_values = [mode for mode in ReadingMode]

        while True:
            user_input = input(f"Enter mode ({', '.join(valid_values)}): ").strip()

            for mode in ReadingMode:
                if user_input.lower() == mode.lower():
                    return ReadingMode(mode)

            print(f"Invalid mode '{user_input}'. Please choose from: {', '.join(valid_values)}.")

    @staticmethod
    def __get_time(start: bool = True) -> datetime:
        date_format = "%Y-%m-%d %H:%M:%S"  # Includes seconds

        while True:
            dt_str = input(f"Choose *{'start' if start else 'end'}* date "
                           f"(YYYY-MM-DD HH:MM:SS) [e.g., 2025-01-01 14:30:45]: ").strip()
            try:
                dt = datetime.strptime(dt_str, date_format)
                dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
                return dt
            except ValueError:
                print(f"Invalid format. Please use: YYYY-MM-DD HH:MM:SS")

    def _inner_get_input(self) -> TimePickerChosenInput:
        mode = self.__choose_mode()
        if mode == ReadingMode.REALTIME:
            start_time = datetime.now(timezone.utc)
            end_time = None
        elif mode == ReadingMode.SINCE:
            start_time = self.__get_time()
            end_time = None
        elif mode == ReadingMode.OFFLINE:
            while True:
                start_time = self.__get_time()
                end_time = self.__get_time(start=False)

                if start_time < end_time:
                    break
                else:
                    print("Start time must be before end time. Please try again.")
        else:
            raise ValueError("Invalid mode selected")

        return TimePickerChosenInput(
            start=start_time,
            end=end_time,
            mode=mode
        )
