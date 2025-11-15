import logging
from typing import Union

from DTOs.aggregated_results_dtos.empty_aggregation_results import EmptyAggregationResults
from DTOs.aggregated_results_dtos.energy_model_result import EnergyModelResult
from DTOs.aggregators_features.empty_features import EmptyFeatures
from DTOs.aggregators_features.energy_model_features.full_energy_model_features import EnergyModelFeatures
from DTOs.raw_results_dtos.iteration_info import IterationMetadata
from DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from elastic_reader.aggregators.abstract_aggregator import AbstractAggregator
from measurements_model_pipeline.energy_model import EnergyModel
from energy_model.energy_model_utils.energy_model_convertor import EnergyModelConvertor
from energy_model.energy_model_utils.energy_model_feature_extractor import EnergyModelFeatureExtractor
from energy_model.energy_model_utils.resource_energy_calculator import ResourceEnergyCalculator
from energy_model.energy_model_utils.sample_resources_energy import SampleResourcesEnergy

logger = logging.getLogger(__name__)

class EnergyModelAggregator(AbstractAggregator):
    def __init__(self):
        self.__model = EnergyModel.get_instance()
        self.__model.initialize_model()
        self.__resource_energy_calculator = ResourceEnergyCalculator()
        self.__energy_model_feature_extractor = EnergyModelFeatureExtractor()
        # todo: maybe support here the "DatasetProcessor" in order to change categorical columns, etc. Use when hardware columns are part of the train of the model

    def extract_features(self, raw_results: ProcessSystemRawResults,
                         iteration_metadata: IterationMetadata) -> Union[EnergyModelFeatures, EmptyFeatures]:

        return self.__energy_model_feature_extractor.extract_energy_model_features(raw_results,
                                                                                   iteration_metadata.timestamp)

    def process_sample(self, sample: Union[EnergyModelFeatures, EmptyFeatures]) -> Union[
        EnergyModelResult, EmptyAggregationResults]:
        try:
            if isinstance(sample, EmptyFeatures):
                return EmptyAggregationResults()

            sample_df = EnergyModelConvertor.convert_features_to_pandas(sample)
            energy_prediction = self.__model.predict(sample_df)[0]
            if energy_prediction < 0:
                energy_prediction = 0
                logger.warning(
                    f"Received a negative value for energy prediction. Returning 0 mwh. The sample: {sample_df}")

            energy_per_resource = self.__calculate_energy_per_resource(sample, energy_prediction)
            return EnergyModelResult(energy_mwh=energy_prediction,
                                     cpu_energy_consumption=energy_per_resource.cpu_energy_consumption,
                                     ram_energy_consumption=energy_per_resource.ram_energy_consumption,
                                     disk_io_read_energy_consumption=energy_per_resource.disk_io_read_energy_consumption,
                                     disk_io_write_energy_consumption=energy_per_resource.disk_io_write_energy_consumption,
                                     network_io_received_energy_consumption=energy_per_resource.network_io_received_energy_consumption,
                                     network_io_sent_energy_consumption=energy_per_resource.network_io_sent_energy_consumption)
        except Exception as e:
            logger.warning("Error occurred when using the energy model. Returning empty aggregation results. \nThe error is: {}".format(
                    e))
            return EmptyAggregationResults()

    def __calculate_energy_per_resource(self, sample: EnergyModelFeatures,
                                        energy_prediction: float) -> SampleResourcesEnergy:
        return self.__resource_energy_calculator.calculate_relative_energy_consumption(sample.process_features,
                                                                                       energy_prediction)
