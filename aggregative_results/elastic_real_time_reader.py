import time
from collections import defaultdict
from datetime import datetime
from typing import Tuple, DefaultDict

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

# --- Config ---
from aggregative_results.aggregation_manager import AggregationManager
from aggregative_results.raw_results_dtos import Metadata, IterationRawResults, ProcessRawResults
from aggregative_results.raw_results_dtos.system_raw_results import SystemRawResults

ES_URL = "http://127.0.0.1:9200"
ES_USER = "elastic"
ES_PASS = "SVR4mUZl"
INDEX_SYSTEM = "system_metrics"
INDEX_PROCESS = "process_metrics"
# TODO: CHECK WHAT IS GOING WRONG WHEN THIS INTERVAL IS INCREASED
POLL_INTERVAL = 2  # seconds


if __name__ == '__main__':
    # --- Connect to Elasticsearch ---
    client = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASS), verify_certs=False)
    if not client.ping():
        raise RuntimeError("Failed to connect to Elasticsearch")

    print("Connected. Watching for new CPU measurements...\n")

    aggregation_manager = AggregationManager()
    measurement_start_date = None
    iteration_metadata = None
    default_fetching_timestamp = datetime.utcnow()  # TODO: RENAME?
    last_iteration_timestamps: DefaultDict[Tuple[str, str], datetime] = defaultdict(lambda: default_fetching_timestamp)

    while True:
        time.sleep(POLL_INTERVAL)

        try:
            s = Search(using=client, index=INDEX_SYSTEM).filter(
                'range',
                timestamp=
                {
                    'gt': max(
                        last_iteration_timestamps.values(),
                        default=default_fetching_timestamp
                    ).isoformat()
                }
            ).sort('timestamp')

            hits = s.scan()     # <--- Fetches all documents lazily1

            # Assuming there is one document in system metrics in each iteration per (hostname, session_id) pair
            for hit in hits:
                raw_data = hit.to_dict()

                metadata = Metadata.from_dict(raw_data)
                parsed_system_results = SystemRawResults.from_dict(raw_data)

                # --- Fetch corresponding process_metrics ---
                process_search = Search(using=client, index=INDEX_PROCESS).filter(
                    'range',
                    timestamp={
                        'gt': last_iteration_timestamps[(metadata.hostname, metadata.session_id)].isoformat(),
                        'lte': metadata.timestamp.isoformat()
                    }
                ).sort('timestamp')

                process_response = process_search.scan()  # <--- Fetches all documents lazily

                process_results = [
                    ProcessRawResults.from_dict(hit.to_dict())
                    for hit in process_response
                ]

                iteration_raw_results = IterationRawResults(
                    metadata=metadata,
                    system_raw_results=parsed_system_results,
                    processes_raw_results=process_results
                )

                aggregation_manager.feed_full_iteration_raw_data(iteration_raw_results)
                last_iteration_timestamps[(metadata.hostname, metadata.session_id)] = metadata.timestamp

        except KeyboardInterrupt:
            print("Stopped.")
            break
