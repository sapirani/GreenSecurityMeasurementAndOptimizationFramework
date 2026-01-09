from energy_model.dataset_creation.dataset_creation_config import RawTelemetryReaderType, DatasetCreatorType, \
    TargetCalculatorType
from energy_model.dataset_creation.dataset_creators.aggregated_dataset_creator import AggregatedDatasetCreator
from energy_model.dataset_creation.dataset_creators.energy_per_second_dataset_creator import EnergyPerSecondDatasetCreator
from energy_model.dataset_creation.dataset_creators.dataset_creator import DatasetCreator
from energy_model.dataset_creation.dataset_creators.processes_ratio_dataset_creator import ProcessesRatioDatasetCreator
from energy_model.dataset_creation.raw_telemetry_readers.all_processes_telemetry_reader import AllProcessesTelemetryReader
from energy_model.dataset_creation.raw_telemetry_readers.raw_telemetry_reader import RawTelemetryReader
from energy_model.dataset_creation.raw_telemetry_readers.process_of_interest_telemetry_reader import ProcessOfInterestTelemetryReader
from energy_model.dataset_creation.target_calculators.battery_drain_target_calculator import \
    BatteryDrainTargetCalculator
from energy_model.dataset_creation.target_calculators.idle_based_target_calculator import IdleBasedTargetCalculator
from energy_model.dataset_creation.target_calculators.system_based_target_calculator import SystemBasedTargetCalculator
from energy_model.dataset_creation.target_calculators.target_calculator import TargetCalculator

ERROR_MSG = "Target Calculator Type {target_calculator_choice} does not match the Dataset Creator Type {dataset_creator_choice}!"

class DatasetCreatorFactory:
    @staticmethod
    def dataset_creator_factory(dataset_reader_choice: RawTelemetryReaderType, dataset_creator_choice: DatasetCreatorType,
                                target_calculator_choice: TargetCalculatorType, should_filter_batches: bool,
                                batch_time_intervals: list[int] = None) -> DatasetCreator:

        # Check if the combination of target calculator and dataset creator is valid
        # The possible combinations are:
        # * target calculator = Idle based <=> dataset creator = With Process Ratio
        # * target calculator = Battery Drain based <=> dataset creator = With Energy Aggregations
        # * target calculator = System based => dataset creator = EnergyPerSecond or With Aggregations
        # These are the only possible combinations since we need to check whether different combinations work without errors.
        # TODO: check other combinations and remove this if command
        if (target_calculator_choice != TargetCalculatorType.IdleBased and dataset_creator_choice == DatasetCreatorType.WithProcessRatio) or \
           (target_calculator_choice != TargetCalculatorType.BatteryDrainBased and dataset_creator_choice == DatasetCreatorType.WithEnergyAggregation) or \
           (target_calculator_choice == TargetCalculatorType.IdleBased and dataset_creator_choice != DatasetCreatorType.WithProcessRatio) or \
           (target_calculator_choice == TargetCalculatorType.BatteryDrainBased and dataset_creator_choice != DatasetCreatorType.WithEnergyAggregation):
            raise ValueError(
                f"Target Calculator Type {target_calculator_choice} does not match the Dataset Creator Type {dataset_creator_choice}!")

        target_calculator = DatasetCreatorFactory.target_calculator_factory(target_calculator_choice)
        dataset_reader = DatasetCreatorFactory.telemetry_reader_factory(dataset_reader_choice)

        if dataset_creator_choice == DatasetCreatorType.WithProcessRatio:
            return ProcessesRatioDatasetCreator(target_calculator, dataset_reader, batch_time_intervals, should_filter_batches)
        elif dataset_creator_choice == DatasetCreatorType.WithEnergyAggregation:
            return AggregatedDatasetCreator(target_calculator, dataset_reader, batch_time_intervals, should_filter_batches)
        elif dataset_creator_choice == DatasetCreatorType.WithAggregation:
            return AggregatedDatasetCreator(target_calculator, dataset_reader, batch_time_intervals, should_filter_batches)
        elif dataset_creator_choice == DatasetCreatorType.Basic:
            return EnergyPerSecondDatasetCreator(target_calculator, dataset_reader, batch_time_intervals, should_filter_batches)
        else:
            raise ValueError(f"Dataset Creator Type {dataset_creator_choice} is not supported!")

    @staticmethod
    def target_calculator_factory(target_calculator_choice: TargetCalculatorType) -> TargetCalculator:
        if target_calculator_choice == TargetCalculatorType.IdleBased:
            return IdleBasedTargetCalculator()
        elif target_calculator_choice == TargetCalculatorType.BatteryDrainBased:
            return BatteryDrainTargetCalculator()
        elif target_calculator_choice == TargetCalculatorType.SystemBased:
            return SystemBasedTargetCalculator()
        else:
            raise ValueError(f"Target Calculator Type {target_calculator_choice} is not supported!")

    @staticmethod
    def telemetry_reader_factory(dataset_reader_choice: RawTelemetryReaderType) -> RawTelemetryReader:
        if dataset_reader_choice == RawTelemetryReaderType.ProcessOfInterest:
            return ProcessOfInterestTelemetryReader()
        elif dataset_reader_choice == RawTelemetryReaderType.AllProcesses:
            return AllProcessesTelemetryReader()
        else:
            raise ValueError(f"Dataset Reader Type {dataset_reader_choice} is not supported!")
