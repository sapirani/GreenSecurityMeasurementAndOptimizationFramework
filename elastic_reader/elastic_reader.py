import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import DefaultDict, Iterator, Optional, List, Any

from elasticsearch import Elasticsearch
from elasticsearch.dsl.response import Hit
from elasticsearch.dsl import Search
import time
from DTOs.logging.consts import IndexName, SCANNER_FINISHED_MESSAGE
from elastic_reader.consts import MAX_INDEXING_TIME_SECONDS, PULL_PAGE_SIZE, \
    NON_GRACEFUL_TERMINATION_DETECTION_WINDOW_SECONDS, PULL_INTERVAL_SECONDS, IndexRetrievalOrder
from DTOs.raw_results_dtos.iteration_results import IterationResults
from DTOs.raw_results_dtos.iteration_info import IterationMetadata, IterationRawResults
from elastic_reader.elastic_reader_parameters import ES_USER, ES_PASS, ES_URL
from user_input.elastic_reader_input.abstract_date_picker import ReadingMode, TimePickerChosenInput

# TODO: INJECT URL AND CREDENTIALS INSTEAD OF IMPORTING DIRECTLY FROM THE PARAMETERS FILE
# TODO: CONSIDER MAKING THIS CLASS MORE GENERIC TO SUPPORT OTHER INDICES FOR REAL, WHICH RETURNS
#  IterationRawResults THAT IS COMPRISED ONLT OF PROCESS, SYSTEM AND METADATA.
# TODO: MAYBE INJECT THE ENTIRE Elasticsearch CLIENT
class ElasticReader:
    def __init__(
            self,
            time_picker_input: TimePickerChosenInput,
            indices: list[IndexName],
            *,
            should_terminate_event: Optional[threading.Event] = None
    ):
        self.time_picker_input = time_picker_input
        self.indices = indices
        if IndexName.APPLICATION_FLOW not in self.indices:
            self.indices.append(IndexName.APPLICATION_FLOW)

        self.should_terminate_event = should_terminate_event
        self.es = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASS), verify_certs=False)

        self.__ongoing_iteration_metadata: Optional[IterationMetadata] = None
        self.__previous_metadata_set = set()
        self.__results_by_session_host: DefaultDict[IterationMetadata, IterationResults] = defaultdict(
            lambda: IterationResults())

    def __get_next_hits(self, last_sort: Optional[List[Any]]) -> List[Hit]:
        """
        :param last_sort: latest result we had, as retrieved by the last hit.meta.sort.
            The values in the list correspond to the fields that the query was sorted according to.
        :return: available results from Elasticsearch, that where ingested into Elasticsearch after the last documented
            represented as last_sort. If none exist, return an empty list

        Important! sorting must be unique. I.e., we cannot have 2 or more documents receiving the same "sort score".
            To "enhance" uniqueness, '_seq_no' is used. After each operation in the shard
            (a subset of documents in the index), such as document insertion, update, or deletion, this number is
             incremented and attached to the document. A single document is attached to a single shard.
             In our case, where telemetry documents are not supposed to be modified and are used as read-only, this
             number is fixed and unique per document in the shard. If this assumption breaks - the correctness of this
             function also breaks (we may retrieve duplicates of the same document).
             In general, it is not guaranteed that '_seq_no' is globally unique across shards (since 2 documents reside
             in 2 different shards may obtain the same value), so it must be ensured
             that the combination with the other order fields that the sorting relies on is unique.
             I.e, if other order fields are identical between 2 documents, and they reside in 2 different shards, and
             somehow the exact same number of operations occurred in each of the shards - we may miss or see the same
             document more than once.
             For most appropriate fields, such as _id, a special option in Elasticsearch should be enabled.
        """
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
            { # index ordering
                "_script": {
                    "type": "number",
                    "order": "asc",
                    "script": {
                        "lang": "painless",
                        "source": f"""
                          if (doc['_index'].value == '{IndexName.PROCESS_METRICS}') return {IndexRetrievalOrder.FIRST};
                          if (doc['_index'].value == '{IndexName.SYSTEM_METRICS}') return {IndexRetrievalOrder.SECOND};
                          if (doc['_index'].value == '{IndexName.APPLICATION_FLOW}') return {IndexRetrievalOrder.THIRD};
                          return {IndexRetrievalOrder.LAST};
                        """
                    }
                }
            },
            {"pid": {"order": "asc", "missing": "_last", "unmapped_type": "long"}},
            {"process_name.keyword": {"order": "asc", "missing": "_last", "unmapped_type": "keyword"}},
            # Previous fields should be unique; "_seq_no" provides uniqueness if they aren't,
            # though collisions are still possible in extreme cases.
            "_seq_no"
        )

        if last_sort:  # retrieve results that come after the last retrieved result
            s = s.extra(search_after=last_sort)

        response = s[:PULL_PAGE_SIZE].execute()
        return response.hits

    def identify_non_graceful_termination(self, *, force: bool = False) -> Iterator[IterationRawResults]:
        """
        Regular iterations (where each usually measures all processes and system telemetry)
        are identified by receiving a newer document.
        I.e., since documents are uniquely sorted with timestamp being the primary sorting key,
        retrieving a document with a newer timestamp means that the iteration is over,
        and we are starting a new iteration.
        As for the last iteration, we are looking for a specific log message that says that scanning has terminated.

        This function assumes that if enough time passed (defined in the consts.py file)
        since the timestamp of the last fetched document, the iteration is done (i.e., some non-graceful termination
        happened in the measurement telemetry code), and would yield the iteration results.

        :param force: enable immediate identification as the last iterations (per session-hostname pair), and return
        iteration results accordingly
        """
        for iteration_metadata, iteration_results in self.__results_by_session_host.copy().items():
            if datetime.now(timezone.utc) - iteration_metadata.timestamp > timedelta(
                    seconds=NON_GRACEFUL_TERMINATION_DETECTION_WINDOW_SECONDS) or force:  # assuming it is the last iteration

                yield from self.__yield_iteration_per_session_host(iteration_metadata, is_last_iteration=True)

    def __yield_iteration_per_session_host(
            self,
            iteration_metadata: IterationMetadata,
            *,
            is_last_iteration: bool,
            next_iteration_metadata: Optional[IterationMetadata] = None
    ):
        if not self.__results_by_session_host[iteration_metadata].get_system_result():
            print(
                "Warning! received empty system results\n"
                f"datetime.now()={datetime.now()}\n"
                f"metadata={iteration_metadata}\n"
                f"next_iteration_metadata={next_iteration_metadata}\n"
                f"is_last_iteration={is_last_iteration}\n"
                f"processes_raw_results={self.__results_by_session_host[iteration_metadata].get_processes_results()}"
            )

        yield IterationRawResults(
            metadata=iteration_metadata,
            system_raw_results=self.__results_by_session_host[iteration_metadata].get_system_result(),
            processes_raw_results=self.__results_by_session_host[iteration_metadata].get_processes_results(),
            is_last_iteration=is_last_iteration,
        )

        # Delete previous iteration data
        self.__results_by_session_host.pop(iteration_metadata)
        self.__ongoing_iteration_metadata = None

    @staticmethod
    def __is_scanner_terminated(raw_data):
        return raw_data.get('message') == SCANNER_FINISHED_MESSAGE

    def read(self) -> Iterator[IterationRawResults]:
        """
        Yield the iteration results right when they are ready.
        Iteration results refers to all telemetry related to the same scanner iteration of measurements.
        E.g., all processes telemetry and system telemetry are gathered into a single iteration results DTO.
        Assumption: the scanner is outputting a termination message right when its last iteration is detected.
        Important: this message should have the exact same timestamp as the telemetry results of the last iteration
        """
        last_sort = None

        while True:
            if self.should_terminate_event is not None and self.should_terminate_event.is_set():
                return

            hits = self.__get_next_hits(last_sort)

            if not hits:
                yield from self.identify_non_graceful_termination()

                if self.time_picker_input.mode == ReadingMode.REALTIME or self.time_picker_input.mode == ReadingMode.SINCE:
                    time.sleep(PULL_INTERVAL_SECONDS)  # wait a bit for new docs
                    continue
                else:
                    return  # offline finished

            # Process the documents
            for examined_doc in hits:
                raw_data = examined_doc.to_dict()
                current_doc_iteration_metadata = IterationMetadata.from_dict(raw_data)

                if examined_doc.meta.index == IndexName.APPLICATION_FLOW:
                    if self.__is_scanner_terminated(raw_data):
                        # yield the last iteration results right away (when we receive the termination message)
                        yield from self.__yield_iteration_per_session_host(
                            self.__ongoing_iteration_metadata,
                            is_last_iteration=True
                        )
                    continue
                elif not self.__ongoing_iteration_metadata:  # first iteration
                    self.__ongoing_iteration_metadata = current_doc_iteration_metadata
                elif current_doc_iteration_metadata != self.__ongoing_iteration_metadata:  # reached to a new iteration

                    if current_doc_iteration_metadata in self.__previous_metadata_set:
                        print("Warning! received an old metadata")

                    # yield the previous iteration results as we reach to a new iteration
                    yield from self.__yield_iteration_per_session_host(
                        self.__ongoing_iteration_metadata,
                        is_last_iteration=False,
                        next_iteration_metadata=current_doc_iteration_metadata,
                    )

                    # Instantiate the new iteration
                    self.__ongoing_iteration_metadata = current_doc_iteration_metadata
                    # TODO: REMOVE FROM PREVIOUS METADATA SET SOMEWHERE
                    self.__previous_metadata_set.add(current_doc_iteration_metadata)

                # insert the new document to the iteration dict either way
                self.__results_by_session_host[self.__ongoing_iteration_metadata].add_result(
                    examined_doc.meta.index,
                    raw_data
                )

            last_sort = hits[-1].meta.sort
