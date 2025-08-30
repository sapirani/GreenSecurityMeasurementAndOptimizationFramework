from typing import List
from aggregation_manager import AggregationManager
from elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from elastic_consumers.consumer_factory import get_consumers
from elastic_reader import ElasticReader
from elastic_reader_parameters import *
from consts import ElasticIndex
from user_input.abstract_date_picker import TimePickerChosenInput
from user_input.input_factory import get_time_picker_input


def main(
    time_picker_input: TimePickerChosenInput,
    consumers: List[AbstractElasticConsumer],
    indices_to_read_from: List[ElasticIndex]
):
    print(time_picker_input)
    reader = ElasticReader(time_picker_input, indices_to_read_from)
    aggregation_manager = AggregationManager()

    for iteration_results in reader.read():     # TODO: SUPPORT READING AGGREGATIONS DIRECTLY FROM INDEX
        aggregation_results = None
        if aggregation_strategy == AggregationStrategy.CALCULATE:
            try:
                aggregation_results = aggregation_manager.aggregate_iteration_raw_results(iteration_results)
            except Exception as e:  # TODO: HANDLE NONE VALUES INSIDE AGGREGATORS AND DO NOT RAISE EXCEPTIONS
                print(
                    "Warning! It seems like indexing is too slow. Consider increasing MAX_INDEXING_TIME_SECONDS")
                print("The received exception:", e)

        for consumer in consumers:
            try:
                consumer.consume(iteration_results, aggregation_results)
            except Exception as e:
                print(f"Warning! consumer {consumer.__class__.__name__} raised an exception:")
                print(e)


if __name__ == '__main__':
    main(
        time_picker_input=get_time_picker_input(time_picker_input_strategy),
        consumers=get_consumers(consumer_types),
        # TODO: SUPPORT COMBINATIONS OF INDICES TO READ FROM (as a user input in the elastic_reader_parameters.py)
        indices_to_read_from=[ElasticIndex.PROCESS, ElasticIndex.SYSTEM]
    )
