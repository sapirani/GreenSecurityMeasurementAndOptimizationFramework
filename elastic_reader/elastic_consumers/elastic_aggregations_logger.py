from dataclasses import asdict
from logging import StreamHandler
from typing import Optional

from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from application_logging.formatters.pretty_extra_formatter import PrettyExtrasFormatter
from application_logging.handlers.elastic_handler import get_elastic_logging_handler
from application_logging.logging_utils import get_measurement_logger
from consts import Verbosity
from elastic_reader_parameters import ES_URL, ES_USER, ES_PASS
from elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from utils.general_consts import LoggerName, IndexName


class ElasticAggregationsLogger(AbstractElasticConsumer):
    logger = get_measurement_logger(
        logger_name=LoggerName.METRICS_AGGREGATIONS,
        logger_handler=get_elastic_logging_handler(ES_USER, ES_PASS, ES_URL, IndexName.METRICS_AGGREGATIONS),
    )

    def __init__(self, verbosity_level: Verbosity):
        if verbosity_level == Verbosity.VERBOSE:
            handler = StreamHandler()
            handler.setFormatter(PrettyExtrasFormatter())
            self.logger.addHandler(handler)

    # TODO: IF CALCAULTIONS ARE SELECTED - ENSURE THAT NO AGGREGATIONS ARE FOUND IN THE INDEX DURING THE ENTIRE REQUESTED TIMERANGE (RAISE AN EXCEPTION AND CRASH)
    def consume(
            self,
            iteration_raw_results: IterationRawResults,
            iteration_aggregation_results: Optional[IterationAggregatedResults],
    ):
        if not iteration_aggregation_results:
            return

        # TODO: IMPROVE LOGGING SPEED BY USING from elasticsearch.helpers import bulk
        for process_identity, process_results in iteration_aggregation_results.processes_results.items():
            self.logger.info(
                "Process Aggregation Results",
                extra=
                {
                    **asdict(iteration_aggregation_results.iteration_metadata),
                    **asdict(process_identity),
                    **asdict(process_results.process_metadata),
                    **{
                        result_name: result_val
                        for aggregation_result in process_results.aggregation_results for result_name, result_val in
                        asdict(aggregation_result).items()
                    }
                }
            )
        self.logger.info(
            "System Aggregation Results",
            extra=
            {
                **asdict(iteration_aggregation_results.iteration_metadata),
                **{key: value for aggregation_result in iteration_aggregation_results.system_aggregated_results
                   for key, value in asdict(aggregation_result).items()}
            }
        )
