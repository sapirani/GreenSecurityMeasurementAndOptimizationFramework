import logging

from DTOs.aggregators_features.energy_model_features.full_energy_model_features import EnergyModelFeatures
from aggregators.energy_model_aggregators.abstract_energy_model_aggregator import EnergyModelAggregator
from energy_model.energy_model_utils.sample_resources_energy import SampleResourcesEnergy
from energy_model.models.aggregations_energy_model import ModelType

logger = logging.getLogger(__name__)


class ProcessEnergyModelAggregator(EnergyModelAggregator):
    def __init__(self):
        super().__init__(ModelType.ProcessBased)

    def _calculate_energy_per_resource(self, sample: EnergyModelFeatures,
                                       energy_prediction: float) -> SampleResourcesEnergy:
        return self._resource_energy_calculator.calculate_relative_energy_consumption(sample.process_features,
                                                                                      energy_prediction)
