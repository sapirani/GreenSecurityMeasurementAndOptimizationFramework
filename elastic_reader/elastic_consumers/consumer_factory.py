from typing import List

from elastic_reader.consts import ElasticConsumerType, Verbosity
from elastic_reader.elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from elastic_reader.elastic_consumers.csv_consumer import CSVConsumer
from elastic_reader.elastic_consumers.drl_consumer import DRLConsumer
from elastic_reader.elastic_consumers.elastic_aggregations_logger import ElasticAggregationsLogger
from user_input.elastic_reader_input.abstract_date_picker import ReadingMode


def get_consumers(
        consumer_types: List[ElasticConsumerType],
        reading_mode: ReadingMode,
        verbosity_level: Verbosity
) -> List[AbstractElasticConsumer]:
    consumers = []

    if ElasticConsumerType.AGGREGATIONS_LOGGER in consumer_types:
        consumers.append(ElasticAggregationsLogger(reading_mode=reading_mode, verbosity_level=verbosity_level))
    if ElasticConsumerType.CSV in consumer_types:   # TODO: SUPPORT csv consumer
        consumers.append(CSVConsumer())
    if ElasticConsumerType.DRL in consumer_types:   # TODO: SUPPORT drl consumer
        consumers.append(DRLConsumer())

    return consumers
