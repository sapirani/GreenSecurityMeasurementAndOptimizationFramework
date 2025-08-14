import time
from collections import defaultdict
from datetime import datetime
from typing import Tuple, DefaultDict

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

# --- Config ---
from aggregative_results import ES_URL, ES_USER, ES_PASS, INDEX_SYSTEM, INDEX_PROCESS, PULL_INTERVAL_SECONDS
from aggregative_results.aggregation_manager import AggregationManager
from aggregative_results.dtos import ProcessRawResults
from aggregative_results.dtos.raw_results_dtos import IterationMetadata, SystemRawResults, IterationRawResults

if __name__ == '__main__':
    # --- Connect to Elasticsearch ---
    client = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASS), verify_certs=False)
    if not client.ping():
        raise RuntimeError("Failed to connect to Elasticsearch")

    print("Connected. Watching for new measurements...\n")

    aggregation_manager = AggregationManager()
    measurement_start_date = None
    default_fetching_timestamp = datetime.utcnow()  # TODO: RENAME?
    last_iteration_timestamps: DefaultDict[Tuple[str, str], datetime] = defaultdict(lambda: default_fetching_timestamp)

    while True:
        time.sleep(PULL_INTERVAL_SECONDS)

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
            ).sort('timestamp').extra(size=1000)

            response = s.execute()

            # Assuming there is one document in system metrics in each iteration per (hostname, session_id) pair
            for hit in response.hits:
                raw_data = hit.to_dict()

                iteration_metadata = IterationMetadata.from_dict(raw_data)
                parsed_system_results = SystemRawResults.from_dict(raw_data)

                # --- Fetch corresponding process_metrics ---
                process_search = Search(using=client, index=INDEX_PROCESS).filter(
                    'range',
                    timestamp={
                        'gt': last_iteration_timestamps[(iteration_metadata.hostname, iteration_metadata.session_id)].isoformat(),
                        'lte': iteration_metadata.timestamp.isoformat()
                    }
                ).sort('timestamp').extra(size=10000)

                process_response = process_search.execute()

                process_results = [
                    ProcessRawResults.from_dict(hit.to_dict())
                    for hit in process_response
                ]

                iteration_raw_results = IterationRawResults(
                    metadata=iteration_metadata,
                    system_raw_results=parsed_system_results,
                    processes_raw_results=process_results
                )

                aggregation_manager.feed_full_iteration_raw_data(iteration_raw_results)
                last_iteration_timestamps[(iteration_metadata.hostname, iteration_metadata.session_id)] = iteration_metadata.timestamp

        except KeyboardInterrupt:
            print("Stopped.")
            break
