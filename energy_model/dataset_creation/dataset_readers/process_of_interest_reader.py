from datetime import datetime

from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.aggregators_features.energy_model_features.full_energy_model_features import ExtendedEnergyModelFeatures
from DTOs.process_info import ProcessIdentity
from DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from DTOs.session_host_info import SessionHostIdentity
from energy_model.dataset_creation.dataset_readers.dataset_reader import DatasetReader


class ProcessOfInterestReader(DatasetReader):
    def _extract_iteration_samples(self, system_raw_results: SystemRawResults,
                                   processes_raw_results: list[ProcessRawResults],
                                   timestamp: datetime, session_host_identity: SessionHostIdentity) -> \
            list[ExtendedEnergyModelFeatures]:
        iteration_samples = []
        for process_result in processes_raw_results:
            if not process_result.process_of_interest:
                continue

            sample_raw_results = ProcessSystemRawResults(system_raw_results=system_raw_results,
                                                         process_raw_results=process_result)
            process_id = ProcessIdentity.from_raw_results(process_result)
            process_feature_extractor = self._processes_features_extractor_mapping[process_id]
            sample_features = process_feature_extractor.extract_extended_energy_model_features(
                raw_results=sample_raw_results, timestamp=timestamp, session_host_identity=session_host_identity)

            if isinstance(sample_features, EmptyFeatures):
                continue

            iteration_samples.append(sample_features)

        return iteration_samples
