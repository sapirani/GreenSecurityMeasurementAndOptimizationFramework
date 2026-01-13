import threading
import traceback
from typing import List, Iterator, Optional
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from elastic_reader.aggregation_manager import AggregationManager
from elastic_reader.elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from elastic_reader.elastic_consumers.consumer_factory import get_consumers
from elastic_reader.elastic_reader import ElasticReader
from elastic_reader.elastic_reader_parameters import *
from elastic_reader.consts import ElasticIndex
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput
from user_input.elastic_reader_input.time_picker_input_factory import get_time_picker_input


def iterate_results(
        raw_results_iterator: Iterator[IterationRawResults],  # TODO: SUPPORT READING AGGREGATIONS DIRECTLY FROM INDEX
        consumers: List[AbstractElasticConsumer],
        aggregation_manager: AggregationManager,
):
    for iteration_results in raw_results_iterator:
        aggregation_results = None
        if aggregation_strategy == AggregationStrategy.CALCULATE:
            aggregation_results = aggregation_manager.aggregate_iteration_raw_results(iteration_results)

        for consumer in consumers:
            try:
                consumer.consume(iteration_results, aggregation_results)
            except Exception:
                print(f"Warning! consumer {consumer.__class__.__name__} raised an exception:")
                traceback.print_exc()


def trigger_post_processing(consumers: List[AbstractElasticConsumer]):
    print("Calling consumers' post processing")
    for consumer in consumers:
        try:
            consumer.post_processing()
        except Exception:
            print(f"Warning! consumer {consumer.__class__.__name__} raised an exception:")
            traceback.print_exc()


def run_elastic_reader(
    time_picker_input: TimePickerChosenInput,
    consumers: List[AbstractElasticConsumer],
    indices_to_read_from: List[ElasticIndex],
    *,
    should_terminate_event: Optional[threading.Event] = None
):
    print(time_picker_input)
    reader = ElasticReader(time_picker_input, indices_to_read_from, should_terminate_event=should_terminate_event)
    aggregation_manager = AggregationManager()

    try:
        iterate_results(reader.read(), consumers, aggregation_manager)
    except KeyboardInterrupt:
        print("A keyboard interrupt was detected, finalizing...")
        print("Note: last iteration results might be incomplete due to interruption")
        iterate_results(reader.identify_non_graceful_termination(force=True), consumers, aggregation_manager)

    trigger_post_processing(consumers)


if __name__ == '__main__':
    time_picker_input = get_time_picker_input(time_picker_input_strategy, preconfigured_time_picker_input)
    run_elastic_reader(
        time_picker_input=time_picker_input,
        consumers=get_consumers(consumer_types, time_picker_input.mode, verbosity),
        # TODO: SUPPORT COMBINATIONS OF INDICES TO READ FROM (as a user input in the elastic_reader_parameters.py)
        indices_to_read_from=[ElasticIndex.PROCESS, ElasticIndex.SYSTEM]
    )
