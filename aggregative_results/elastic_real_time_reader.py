import time
from datetime import datetime

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

# --- Config ---
from aggregative_results.aggregation_manager import AggregationManager
from aggregative_results.raw_results_dtos import IterationMetadata, IterationRawResults
from aggregative_results.raw_results_dtos.system_raw_results import SystemRawResults

ES_URL = "http://127.0.0.1:9200"
ES_USER = "elastic"
ES_PASS = "SVR4mUZl"
INDEX = "system_metrics"
POLL_INTERVAL = 2  # seconds


def parse_time(ts_str):
    return pd.to_datetime(ts_str)


if __name__ == '__main__':
    # --- Connect to Elasticsearch ---
    client = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASS), verify_certs=False)
    if not client.ping():
        raise RuntimeError("Failed to connect to Elasticsearch")

    print("Connected. Watching for new CPU measurements...\n")

    aggregation_manager = AggregationManager()
    measurement_start_date = None
    last_timestamp = datetime.utcnow().isoformat()

    # TODO: GROUP RESULTS BY HOSTNAME and session_id
    while True:
        time.sleep(POLL_INTERVAL)

        try:
            s = Search(using=client, index=INDEX).filter(
                'range', timestamp={'gt': last_timestamp}
            ).sort('timestamp')

            response = s.execute()
            hits = response.hits

            for hit in hits:
                raw_data = hit.to_dict()

                # TODO: LOG RAW RESULTS TO THE CONSOLE ONLY

                iteration_metadata = IterationMetadata.from_dict(raw_data)
                parsed_system_results = SystemRawResults.from_dict(raw_data)

                # TODO: ENSURE ITERATION IS FINISHED
                iteration_raw_results = IterationRawResults(
                    metadata=iteration_metadata,
                    system_raw_results=parsed_system_results,
                    processes_raw_results=[]
                )

                aggregation_manager.feed_full_iteration_raw_data(iteration_raw_results)

            print("iteration finished")
        except KeyboardInterrupt:
            print("Stopped.")
            break
