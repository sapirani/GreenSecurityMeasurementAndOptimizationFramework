from elastic_reader.elastic_consumers.consumer_factory import get_consumers
from elastic_reader.elastic_reader_parameters import *
from DTOs.logging.consts import IndexName
from elastic_reader_service import ElasticReaderService
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input


if __name__ == '__main__':
    time_picker_input = get_time_picker_input(time_picker_input_strategy, preconfigured_time_picker_input)

    elastic_reader_service = ElasticReaderService(
        time_picker_input=time_picker_input,
        consumers=get_consumers(consumer_types, time_picker_input.mode, verbosity),
        indices_to_read_from=[IndexName.PROCESS_METRICS, IndexName.SYSTEM_METRICS]
    )

    elastic_reader_service.run_elastic_reader()

