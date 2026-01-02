from energy_model.dataset_creation.dataset_creation_config import DatasetReaderType, DatasetCreatorType, \
    TargetCalculatorType
from energy_model.dataset_creation.dataset_creators.aggregated_dataset_creator import AggregatedDatasetCreator
from energy_model.dataset_creation.dataset_creators.basic_dataset_creator import BasicDatasetCreator
from energy_model.dataset_creation.dataset_creators.dataset_creator import DatasetCreator
from energy_model.dataset_creation.dataset_creators.energy_aggregated_dataset_creator import \
    EnergyAggregatedDatasetCreator
from energy_model.dataset_creation.dataset_creators.processes_ratio_dataset_creator import ProcessesRatioDatasetCreator
from energy_model.dataset_creation.dataset_readers.all_processes_reader import AllProcessesReader
from energy_model.dataset_creation.dataset_readers.dataset_reader import DatasetReader
from energy_model.dataset_creation.dataset_readers.process_of_interest_reader import ProcessOfInterestReader
from energy_model.dataset_creation.target_calculators.battery_drain_target_calculator import \
    BatteryDrainTargetCalculator
from energy_model.dataset_creation.target_calculators.idle_based_target_calculator import IdleBasedTargetCalculator
from energy_model.dataset_creation.target_calculators.system_based_target_calculator import SystemBasedTargetCalculator
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator


class DatasetCreatorFactory:
    @staticmethod
    def dataset_creator_factory(dataset_reader_choice: DatasetReaderType, dataset_creator_choice: DatasetCreatorType,
                                target_calculator_choice: TargetCalculatorType, should_filter_batches: bool,
                                batch_time_intervals: list[int] = None) -> DatasetCreator:
        if (target_calculator_choice == TargetCalculatorType.IdleBased and dataset_creator_choice != DatasetCreatorType.WithProcessRatio) or (
            target_calculator_choice == TargetCalculatorType.BatteryDrainBased and dataset_creator_choice != DatasetCreatorType.WithEnergyAggregation):
            raise ValueError(
                f"Target Calculator Type {target_calculator_choice} does not match the Dataset Creator Type {dataset_creator_choice}!")

        target_calculator = DatasetCreatorFactory.get_target_calculator(target_calculator_choice)
        dataset_reader = DatasetCreatorFactory.get_dataset_reader(dataset_reader_choice)

        if dataset_creator_choice == DatasetCreatorType.WithProcessRatio:
            return ProcessesRatioDatasetCreator(target_calculator, dataset_reader, batch_time_intervals, should_filter_batches)
        elif dataset_creator_choice == DatasetCreatorType.WithEnergyAggregation:
            return EnergyAggregatedDatasetCreator(dataset_reader, batch_time_intervals, should_filter_batches)
        elif dataset_creator_choice == DatasetCreatorType.WithAggregation:
            return AggregatedDatasetCreator(target_calculator, dataset_reader, batch_time_intervals, should_filter_batches)
        elif dataset_creator_choice == DatasetCreatorType.Basic:
            return BasicDatasetCreator(target_calculator, dataset_reader, batch_time_intervals, should_filter_batches)
        else:
            raise ValueError(f"Dataset Creator Type {dataset_creator_choice} is not supported!")

    @staticmethod
    def get_target_calculator(target_calculator_choice: TargetCalculatorType) -> TargetCalculator:
        if target_calculator_choice == TargetCalculatorType.IdleBased:
            return IdleBasedTargetCalculator()
        elif target_calculator_choice == TargetCalculatorType.BatteryDrainBased:
            return BatteryDrainTargetCalculator()
        elif target_calculator_choice == TargetCalculatorType.SystemBased:
            return SystemBasedTargetCalculator()
        else:
            raise ValueError(f"Target Calculator Type {target_calculator_choice} is not supported!")

    @staticmethod
    def get_dataset_reader(dataset_reader_choice: DatasetReaderType) -> DatasetReader:
        if dataset_reader_choice == DatasetReaderType.ProcessOfInterest:
            return ProcessOfInterestReader()
        elif dataset_reader_choice == DatasetReaderType.AllProcesses:
            return AllProcessesReader()
        else:
            raise ValueError(f"Dataset Reader Type {dataset_reader_choice} is not supported!")
