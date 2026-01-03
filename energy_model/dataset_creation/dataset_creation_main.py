from enum import Enum

from energy_model.dataset_creation.dataset_creation_config import DEFAULT_DATASET_READER, DEFAULT_DATASET_CREATOR, \
    DEFAULT_TARGET_CALCULATOR, DEFAULT_BATCH_INTERVAL_SECONDS, DatasetReaderType, TargetCalculatorType, \
    DatasetCreatorType
from energy_model.dataset_creation.dataset_creator_factory import DatasetCreatorFactory
from energy_model.energy_model_parameters import FULL_DATASET_BEFORE_PROCESSING_DIR

class PipelineType(Enum):
    Default = 1
    Costume = 2
    Exit = 3

def print_main_hello_msg() -> PipelineType:
    print("Choose from the following options:")
    print("1. Use default (production) configuration.")
    print("2. Use a custom (dev) configuration.")
    print("3. Exit the program.")
    return PipelineType(int(input("Please enter your choice: ")))

def print_dataset_reader_type() -> DatasetReaderType:
    print("Choose DatasetReader from the following options:")
    print("1. Read only processes of interest")
    print("2. Read all processes available")
    return DatasetReaderType(int(input("Please enter your choice: ")))

def print_target_calculator_type() -> TargetCalculatorType:
    print("Choose TargetCalculator from the following options:")
    print("1. SystemBased - Calculate the system's energy (by energy per second for batch)")
    print("2. IdleBased - Calculate the difference between system energy currently and the system's energy in idle state.")
    print("3. BatteryDrainBased - Calculate the system's energy by the battery drain in the batch.")
    return TargetCalculatorType(int(input("Please enter your choice: ")))

def print_dataset_creator_type() -> DatasetCreatorType:
    print("Choose DatasetCreator from the following options:")
    print("1. Basic - Calculate energy per second for each sample before calculating target.")
    print("2. Aggregate - Each batch contains a single sample per process where the telemetry is summed.")
    print("3. Energy based Aggregate - Same aggregations as option 2, but the energy usage of the system equals to the battery drain.")
    print("4. Processes Ratio - If a batch contains more than one process, calculate the impact of this process on the energy usage of the entire batch.")
    return DatasetCreatorType(int(input("Please enter your choice: ")))


if __name__ == '__main__':
    print("Welcome to Dataset Creator!")
    while (pipeline_choice := print_main_hello_msg()) != PipelineType.Exit:
        try:
            if pipeline_choice == PipelineType.Default:
                dataset_creator = DatasetCreatorFactory.dataset_creator_factory(
                    DEFAULT_DATASET_READER,
                    DEFAULT_DATASET_CREATOR,
                    DEFAULT_TARGET_CALCULATOR,
                    DEFAULT_BATCH_INTERVAL_SECONDS
                )

                print(f"Using default dataset creator with: dataset_reader = {DEFAULT_DATASET_READER}, target_calculator = {DEFAULT_TARGET_CALCULATOR}, batch_size = {DEFAULT_BATCH_INTERVAL_SECONDS}, dataset_creator = {DEFAULT_DATASET_CREATOR}")

            elif pipeline_choice == PipelineType.Costume:
                dataset_reader_choice = print_dataset_reader_type()
                target_calculator_choice = print_target_calculator_type()
                dataset_creator_choice = print_dataset_creator_type()
                print("Do you wish to use only batches with a single process? (y\\n) ")
                should_filter_batches_choice = input()
                should_filter_batches = True if should_filter_batches_choice == 'y' else False

                print("Enter batch intervals:")
                batch_intervals = [int(interval) for interval in input("Enter space-separated integers: ").split()]
                dataset_creator = DatasetCreatorFactory.dataset_creator_factory(dataset_reader_choice,
                                                                                dataset_creator_choice,
                                                                                target_calculator_choice,
                                                                                should_filter_batches,
                                                                                batch_intervals)
                print(
                    f"Using dataset creator with: dataset_reader = {dataset_reader_choice}, target_calculator = {target_calculator_choice}, batch_size = {batch_intervals}, dataset_creator = {dataset_creator_choice}, should_filter_batches = {should_filter_batches}")
            else:
                raise ValueError(f"Unsupported Pipeline type {pipeline_choice}!\n")
        except Exception as e:
            print(e)
            print("Try Again!\n\n")
            continue

        full_dataset = dataset_creator.create_dataset()
        dataset_path = dataset_creator.get_dataset_file_name(FULL_DATASET_BEFORE_PROCESSING_DIR)
        full_dataset.to_csv(dataset_path, index=False)
        print(f"Saved dataset into: {dataset_path}")