from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import DefaultDict, Iterator, Optional

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import time

from consts import ElasticIndex, FINAL_ITERATION_TIMEOUT_SECONDS, PULL_INTERVAL_SECONDS, \
    MAX_INDEXING_TIME_SECONDS, PULL_PAGE_SIZE, ES_URL, ES_USER, ES_PASS
from DTOs.raw_results_dtos.iteration_indices_results import IterationIndicesResults
from DTOs.raw_results_dtos.iteration_info import IterationMetadata, IterationRawResults
from user_input.abstract_date_picker import ReadingMode, TimePickerChosenInput


class ElasticReader:
    def __init__(self, time_picker_input: TimePickerChosenInput, indices: list[ElasticIndex]):
        self.time_picker_input = time_picker_input
        self.indices = indices
        self.es = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASS), verify_certs=False)

        self.__ongoing_iteration_metadata: Optional[IterationMetadata] = None
        self.__previous_metadata_set = set()
        self.__results_by_session_host: DefaultDict[IterationMetadata, IterationIndicesResults] = defaultdict(
            lambda: IterationIndicesResults())

    # TODO: ADD TYPING
    def __get_next_hits(self, last_sort):
        s = Search(using=self.es, index=','.join(self.indices))

        # ensure all documents are indexed before querying
        max_time = datetime.now(timezone.utc) - timedelta(seconds=MAX_INDEXING_TIME_SECONDS)
        query_time_limit = min(max_time, self.time_picker_input.end) if self.time_picker_input.end else max_time

        # Timerange filter
        s = s.query("range", timestamp={"gte": self.time_picker_input.start, "lte": query_time_limit})

        # Sorting - uniqueness is mandatory!!!
        s = s.sort(
            "timestamp",
            {"session_id.keyword": {"order": "asc"}},
            {"hostname.keyword": {"order": "asc"}},
            {"_index": {"order": "asc"}},
            {"pid": {"order": "asc", "missing": "_last", "unmapped_type": "long"}},
            {"process_name.keyword": {"order": "asc", "missing": "_last", "unmapped_type": "keyword"}},
            "_doc"
        )

        if last_sort:  # retrieve results that come after the last retrieved result
            s = s.extra(search_after=last_sort)

        response = s[:PULL_PAGE_SIZE].execute()
        return response.hits

    def __identify_last_iterations(self):
        # TODO: ADD DOCSTRING FOR EXPLANATIONS
        for iteration_metadata, iteration_results in self.__results_by_session_host.copy().items():
            if datetime.now(timezone.utc) - iteration_metadata.timestamp > timedelta(
                    seconds=FINAL_ITERATION_TIMEOUT_SECONDS):  # assuming it is the last iteration

                yield IterationRawResults(
                    metadata=iteration_metadata,
                    system_raw_results=self.__results_by_session_host[iteration_metadata].get_system_result(),
                    # todo: support optional
                    processes_raw_results=self.__results_by_session_host[iteration_metadata].get_processes_results()
                )

                # Delete previous iteration data
                self.__results_by_session_host.pop(iteration_metadata)
                self.__ongoing_iteration_metadata = None

    def read(self) -> Iterator[IterationRawResults]:
        last_sort = None

        while True:
            hits = self.__get_next_hits(last_sort)

            if not hits:
                # TODO: EXPLAIN WHY IT IS ALSO RELEVANT HERE FOR THE OFFLINE
                yield from self.__identify_last_iterations()

                if self.time_picker_input.mode == ReadingMode.REALTIME or self.time_picker_input.mode == ReadingMode.SINCE:
                    time.sleep(PULL_INTERVAL_SECONDS)  # wait a bit for new docs
                    continue
                else:
                    return  # offline finished

            # Process documents
            for examined_doc in hits:
                raw_data = examined_doc.to_dict()
                current_doc_iteration_metadata = IterationMetadata.from_dict(raw_data)

                if not self.__ongoing_iteration_metadata:  # first iteration
                    self.__ongoing_iteration_metadata = current_doc_iteration_metadata
                elif current_doc_iteration_metadata != self.__ongoing_iteration_metadata:  # reached to a new iteration

                    if current_doc_iteration_metadata in self.__previous_metadata_set:
                        print("Warning! received an old metadata")

                    yield IterationRawResults(
                        metadata=self.__ongoing_iteration_metadata,
                        system_raw_results=self.__results_by_session_host[self.__ongoing_iteration_metadata].get_system_result(),
                        # todo: support optional
                        processes_raw_results=self.__results_by_session_host[
                            self.__ongoing_iteration_metadata].get_processes_results()
                    )

                    # Delete previous iteration data
                    self.__results_by_session_host.pop(self.__ongoing_iteration_metadata)

                    # Instantiate the new iteration
                    self.__ongoing_iteration_metadata = current_doc_iteration_metadata
                    self.__previous_metadata_set.add(current_doc_iteration_metadata)

                # insert the new document to the iteration dict either way
                self.__results_by_session_host[self.__ongoing_iteration_metadata].add_result(examined_doc.meta.index, raw_data)

            last_sort = hits[-1].meta.sort

        # TODO: ADD SOME POP UP EXPLAINING THAT WE ARE WAITING TO DETECT THE LAST ITERATION AND ASK IF HE IS SURE THAT HE WANTS TO QUIT NOW
