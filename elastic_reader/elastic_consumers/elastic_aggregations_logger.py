from dataclasses import asdict
from logging import StreamHandler
from typing import Optional, Dict, Any

from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from application_logging.formatters.pretty_extra_formatter import PrettyExtrasFormatter
from application_logging.handlers.elastic_bulk_handler import get_elastic_bulk_handler
from application_logging.logging_utils import get_measurement_logger
from elastic_reader.consts import Verbosity
from elastic_reader.elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from elastic_reader.elastic_reader_parameters import ES_USER, ES_PASS, ES_URL, custom_pipeline_name
from user_input.elastic_reader_input.abstract_date_picker import ReadingMode
from utils.general_consts import LoggerName, IndexName


class ElasticAggregationsLogger(AbstractElasticConsumer):
    def __init__(
            self,
            *,
            reading_mode: ReadingMode,
            verbosity_level: Verbosity = Verbosity.NONE,
            queued_iterations_num_max: int = 15     # used as an optimization for in offline mode
    ):
        self.logger = get_measurement_logger(
            logger_name=LoggerName.METRICS_AGGREGATIONS,
            logger_handler=get_elastic_bulk_handler(
                ES_USER,
                ES_PASS, ES_URL,
                IndexName.METRICS_AGGREGATIONS,
                pipeline_name=custom_pipeline_name
            ),
        )

        self.reading_mode = reading_mode

        if verbosity_level == Verbosity.VERBOSE:
            handler = StreamHandler()
            handler.setFormatter(PrettyExtrasFormatter())
            self.logger.addHandler(handler)

        self.queued_iterations_num = 0
        self.queued_iterations_num_max = queued_iterations_num_max

    def _flatten_dict(self, nested_dict: Dict[str, Any]) -> Dict[str, Any]:
        flat = {}
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                # Recursively flatten nested dicts
                flat.update(self._flatten_dict(value))
            else:
                flat[key] = value
        return flat

    # TODO: IF CALCAULTIONS ARE SELECTED - ENSURE THAT NO AGGREGATIONS ARE FOUND IN THE INDEX DURING THE ENTIRE REQUESTED TIMERANGE (RAISE AN EXCEPTION AND CRASH)
    def consume(
            self,
            iteration_raw_results: IterationRawResults,
            iteration_aggregation_results: Optional[IterationAggregatedResults],
    ):
        if not iteration_aggregation_results:
            return

        for process_identity, process_results in iteration_aggregation_results.processes_results.items():
            self.logger.info(
                "Process Aggregation Results",
                extra=
                {
                    **self._flatten_dict(asdict(iteration_aggregation_results.iteration_metadata)),
                    **asdict(process_identity),
                    **asdict(process_results.process_metadata),
                    **{
                        result_name: result_val
                        for aggregation_result in process_results.aggregation_results.values()
                        for result_name, result_val in asdict(aggregation_result).items()
                    }
                }
            )
        self.logger.info(
            "System Aggregation Results",
            extra=
            {
                **self._flatten_dict(asdict(iteration_aggregation_results.iteration_metadata)),
                **{key: value for aggregation_result in iteration_aggregation_results.system_aggregated_results.values()
                   for key, value in asdict(aggregation_result).items()}
            }
        )

        self.queued_iterations_num += 1

        # Significant performance optimization
        if self.reading_mode == ReadingMode.OFFLINE and self.queued_iterations_num != self.queued_iterations_num_max:
            return

        for handler in self.logger.handlers:
            handler.flush()
        self.queued_iterations_num = 0

    def post_processing(self):
        for handler in self.logger.handlers:
            handler.flush()
