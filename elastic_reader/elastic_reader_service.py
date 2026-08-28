import threading
import traceback
from datetime import datetime
from typing import Optional, List, Iterator
from DTOs.logging.consts import IndexName
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from elastic_reader.aggregation_manager import AggregationManager
from elastic_reader.consts import AggregationStrategy
from elastic_reader.elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from elastic_reader.elastic_reader import ElasticReader
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput, ReadingMode


class ElasticReaderService:
    def __init__(
            self,
            consumers: List[AbstractElasticConsumer],
            aggregation_strategy: AggregationStrategy = AggregationStrategy.CALCULATE,
            time_picker_input: Optional[TimePickerChosenInput] = None,
            indices_to_read_from: Optional[List[IndexName]] = None,
            aggregation_manager: Optional[AggregationManager] = None,
            should_terminate_event: Optional[threading.Event] = None
    ):
        self.consumers = consumers
        self.aggregation_strategy = aggregation_strategy
        self.should_terminate_event = should_terminate_event or threading.Event()

        self.aggregation_manager = aggregation_manager or AggregationManager()

        self.time_picker_input = time_picker_input or TimePickerChosenInput(
                start=datetime.now(tz=datetime.now().astimezone().tzinfo),
                end=None,
                mode=ReadingMode.REALTIME
            )

        self.indices_to_read_from = indices_to_read_from or [IndexName.PROCESS_METRICS, IndexName.SYSTEM_METRICS]

        self.elastic_reader_thread = None

    def __enter__(self):
        self.start_in_background()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    async def __aenter__(self):
        self.start_in_background()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def is_running(self):
        return bool(self.elastic_reader_thread and self.elastic_reader_thread.is_alive())

    def start_in_background(self):
        if self.is_running():
            return

        self.should_terminate_event.clear()

        print("Starting Elastic Reader")
        self.elastic_reader_thread = threading.Thread(target=self._run_elastic_reader, daemon=True)
        self.elastic_reader_thread.start()

    def stop(self):
        print("terminating elastic reader")
        self.should_terminate_event.set()
        if self.elastic_reader_thread:
            self.elastic_reader_thread.join(timeout=30)
            if self.elastic_reader_thread.is_alive():
                print("WARNING: elastic reader did not shut down cleanly")

        self.elastic_reader_thread = None

    def _iterate_results(
            self,
            raw_results_iterator: Iterator[IterationRawResults],
            # TODO: SUPPORT READING AGGREGATIONS DIRECTLY FROM INDEX
    ):
        for iteration_results in raw_results_iterator:
            aggregation_results = None
            if self.aggregation_strategy == AggregationStrategy.CALCULATE:
                aggregation_results = self.aggregation_manager.aggregate_iteration_raw_results(iteration_results)

            for consumer in self.consumers:
                try:
                    consumer.consume(iteration_results, aggregation_results)
                except Exception:
                    print(f"Warning! consumer {consumer.__class__.__name__} raised an exception:")
                    traceback.print_exc()

    def _trigger_post_processing(self):
        print("Calling consumers' post processing")
        for consumer in self.consumers:
            try:
                consumer.post_processing()
            except Exception:
                print(f"Warning! consumer {consumer.__class__.__name__} raised an exception:")
                traceback.print_exc()

    def run_elastic_reader(self):
        if self.is_running():
            raise RuntimeError("Cannot run ElasticReaderService both in background and foreground")

        self._run_elastic_reader()

    def _run_elastic_reader(self):
        print(self.time_picker_input)
        reader = ElasticReader(
            self.time_picker_input,
            self.indices_to_read_from,
            should_terminate_event=self.should_terminate_event
        )

        try:
            self._iterate_results(reader.read())
        except KeyboardInterrupt:
            print("A keyboard interrupt was detected, finalizing...")
            print("Note: last iteration results might be incomplete due to interruption")
            self._iterate_results(reader.identify_non_graceful_termination(force=True))

        self._trigger_post_processing()
