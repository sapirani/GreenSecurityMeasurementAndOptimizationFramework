from datetime import datetime

from dependency_injector import containers, providers
from dependency_injector.providers import Provider

from consts import TimePickerInputStrategy
from hadoop_optimizer.drl_model.drl_model import DRLModel
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput, ReadingMode
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    drl_model: Provider[DRLModel] = providers.Singleton(DRLModel)
    drl_time_picker_input: Provider[TimePickerChosenInput] = providers.Factory(
        get_time_picker_input,
        TimePickerInputStrategy.FROM_CONFIGURATION,
        providers.Callable(lambda: TimePickerChosenInput(
            start=datetime.now(tz=datetime.now().astimezone().tzinfo),
            end=None,
            mode=ReadingMode.REALTIME
        ))
    )