from typing import List
from consts import ElasticConsumerType
from elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from elastic_consumers.csv_consumer import CSVConsumer
from elastic_consumers.drl_consumer import DRLConsumer
from elastic_consumers.elastic_aggregations_logger import ElasticAggregationsLogger


def get_consumers(consumer_types: List[ElasticConsumerType]) -> List[AbstractElasticConsumer]:
    consumers = []

    if ElasticConsumerType.AGGREGATIONS_LOGGER in consumer_types:
        consumers.append(ElasticAggregationsLogger())
    if ElasticConsumerType.CSV in consumer_types:   # TODO: SUPPORT csv consumer
        consumers.append(CSVConsumer())
    if ElasticConsumerType.DRL in consumer_types:   # TODO: SUPPORT drl consumer
        consumers.append(DRLConsumer())

    return consumers
