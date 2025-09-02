import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, DefaultDict

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import time

from aggregative_results import ES_URL, ES_USER, ES_PASS, INDEX_SYSTEM, INDEX_PROCESS, PULL_INTERVAL_SECONDS, \
    PULL_PAGE_SIZE, MAX_INDEXING_TIME_SECONDS
from aggregative_results.DTOs.raw_results_dtos.iteration_info import IterationMetadata, IterationRawResults
from aggregative_results.DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from aggregative_results.DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from aggregative_results.aggregation_manager import AggregationManager

# ------------------------------
# Configuration
# ------------------------------
# os.environ["KIVY_LOG_MODE"] = "PYTHON"
# from aggregative_results.GUI.date_range_gui import ModeApp  # must come after setting KIVY_LOG_MODE (env variable)

es = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASS), verify_certs=False)

# ------------------------------
# User input
# ------------------------------
# app = ModeApp()
# app.run()

mode = "realtime"
local_timezone = datetime.now().astimezone().tzinfo
start_datetime = datetime.now(local_timezone)
# start_datetime = datetime(2025,8,22,10,59,0)
# end_datetime = datetime.now(local_timezone)
end_datetime = None

print("-------------- Chosen Configuration --------------")
print("Selected mode:", mode)
print("Selected start time:", start_datetime.astimezone(local_timezone))
print("Selected end time:", end_datetime.astimezone(local_timezone) if end_datetime else None)
print("--------------------------------------------------")

# ------------------------------
# Pipeline
# ------------------------------
indices_to_fetch_from = [INDEX_PROCESS, INDEX_SYSTEM]
aggregation_manager = AggregationManager()
last_sort = None
ongoing_iteration_metadata = None
current_index = INDEX_SYSTEM


@dataclass
class IterationIndicesResults:
    system_result: Optional[SystemRawResults] = None
    process_results: List[ProcessRawResults] = field(default_factory=list)

    def add_result(self, index: str, raw_results: Dict[str, Any]):
        if index == INDEX_SYSTEM:
            self.__set_system_result(SystemRawResults.from_dict(raw_results))
        elif index == INDEX_PROCESS:
            self.__add_process_result(ProcessRawResults.from_dict(raw_results))
        else:
            raise ValueError("Received unexpected index")

    def __set_system_result(self, result: SystemRawResults):
        self.system_result = result

    def __add_process_result(self, result: ProcessRawResults):
        self.process_results.append(result)

    def get_system_result(self) -> SystemRawResults:
        return self.system_result

    def get_processes_results(self) -> List[ProcessRawResults]:
        return self.process_results


results_by_session_host: DefaultDict[IterationMetadata, IterationIndicesResults] = defaultdict(lambda: IterationIndicesResults())
previous_metadata_set = set()


# TODO: LAST ITERATION IS MISSING. THINK ABOUT WHAT TO DO WITH IT.
while True:
    s = Search(using=es, index=','.join(indices_to_fetch_from))

    # ensure all documents are indexed before querying
    max_time = datetime.now(timezone.utc) - timedelta(seconds=MAX_INDEXING_TIME_SECONDS)
    query_time_limit = min(max_time, end_datetime) if end_datetime else max_time

    # Timerange filter
    s = s.query("range", timestamp={"gte": start_datetime, "lte": query_time_limit})

    # Sorting
    s = s.sort(
        "timestamp",
        {"session_id.keyword": {"order": "asc"}},
        {"hostname.keyword": {"order": "asc"}},
        {"_index": {"order": "asc"}},
        {"pid": {"order": "asc", "missing": "_last", "unmapped_type": "long"}},
        {"process_name.keyword": {"order": "asc", "missing": "_last", "unmapped_type": "keyword"}},
        "_doc"
    )

    if last_sort:
        s = s.extra(search_after=last_sort)

    response = s[:PULL_PAGE_SIZE].execute()
    hits = response.hits

    if not hits:
        for iteration_metadata, iteration_results in results_by_session_host.copy().items():
            # TODO: EXTRACT TIMEDELTA SECONDS TO A CONST
            if datetime.now(timezone.utc) - iteration_metadata.timestamp > timedelta(minutes=2):  # assuming it is the last iteration

                # TODO: UNIFY THESE LINES WITH THE REGULAR CASE:
                iteration_raw_results = IterationRawResults(
                    metadata=iteration_metadata,
                    system_raw_results=results_by_session_host[iteration_metadata].get_system_result(),
                    # todo: support optional
                    processes_raw_results=results_by_session_host[iteration_metadata].get_processes_results()
                )
                try:
                    aggregation_manager.aggregate_iteration_raw_results(iteration_raw_results)
                except Exception as e:
                    print("Warning! It seems like indexing is too slow. Consider increasing MAX_INDEXING_TIME_SECONDS")
                    print("The received exception:", e)

                results_by_session_host.pop(iteration_metadata)

                ongoing_iteration_metadata = None

        if mode == "realtime" or mode == "since":  # TODO: ADD ENUMS
            time.sleep(PULL_INTERVAL_SECONDS)  # wait a bit for new docs
            continue
        else:
            break  # offline finished

    # Process documents
    for examined_doc in hits:
        raw_data = examined_doc.to_dict()

        current_doc_iteration_metadata = IterationMetadata.from_dict(raw_data)

        if not ongoing_iteration_metadata:     # first iteration
            ongoing_iteration_metadata = current_doc_iteration_metadata
        elif current_doc_iteration_metadata != ongoing_iteration_metadata:   # reached to a new iteration

            if current_doc_iteration_metadata in previous_metadata_set:
                print("Warning! received an old metadata")

            # TODO: YIELD ITERATION RESULTS (MAKE THIS FUNCTION A GENERATOR) AND CALCULATE AGGREGATIONS SOMEWHERE ELSE
            iteration_raw_results = IterationRawResults(
                metadata=ongoing_iteration_metadata,
                system_raw_results=results_by_session_host[ongoing_iteration_metadata].get_system_result(), # todo: support optional
                processes_raw_results=results_by_session_host[ongoing_iteration_metadata].get_processes_results()
            )

            try:
                aggregation_manager.aggregate_iteration_raw_results(iteration_raw_results)
            except Exception as e:
                print("Warning! It seems like indexing is too slow. Consider increasing MAX_INDEXING_TIME_SECONDS")
                print("The received exception:", e)

            results_by_session_host.pop(ongoing_iteration_metadata)     # Delete previous iteration data

            # Instantiate the new iteration
            previous_metadata_set.add(current_doc_iteration_metadata)
            ongoing_iteration_metadata = current_doc_iteration_metadata

        # insert the new document to the iteration dict either way
        results_by_session_host[ongoing_iteration_metadata].add_result(examined_doc.meta.index, raw_data)

    last_sort = hits[-1].meta.sort

# TODO: ADD SOME POP UP EXPLAINING THAT WE ARE WAITING TO DETECT THE LAST ITERATION AND ASK IF HE IS SURE THAT HE WANTS TO QUIT NOW
