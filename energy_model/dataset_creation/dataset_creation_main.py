from energy_model.dataset_creation.dataset_creation_config import DEFAULT_DATASET_READER, DEFAULT_DATASET_CREATOR, \
    DEFAULT_TARGET_CALCULATOR, DEFAULT_BATCH_INTERVAL_SECONDS, DatasetReaderType, TargetCalculatorType, \
    DatasetCreatorType
from energy_model.dataset_creation.dataset_creator_factory import DatasetCreatorFactory
from energy_model.energy_model_parameters import FULL_DATASET_BEFORE_PROCESSING_PATH
if __name__ == '__main__':
    print("Welcome to Dataset Creator!")
    print("Choose from the following options:")
    print("1. Use default (production) configuration.")
    print("2. Use a custom (dev) configuration.")
    implementation_choice = int(input("Please enter your choice: "))
    if implementation_choice == 1:
        dataset_creator = DatasetCreatorFactory.dataset_creator_factory(
            DEFAULT_DATASET_READER,
            DEFAULT_DATASET_CREATOR,
            DEFAULT_TARGET_CALCULATOR,
            DEFAULT_BATCH_INTERVAL_SECONDS
        )
    elif implementation_choice == 2:
        print("Configure the Dataset Creator yourself!")
        print("Choose DatasetReader from the following options:")
        print("1. Read only processes of interest")
        print("2. Read all processes available")
        dataset_reader_choice = DatasetReaderType(int(input("Please enter your choice: ")))

        print("Choose TargetCalculator from the following options:")
        print("1. SystemBased - Calculate the system's energy (by energy per second for batch)")
        print("2. IdleBased - Calculate the difference between system energy currently and the system's energy in idle state.")
        print("3. BatteryDrainBased - Calculate the system's energy by the battery drain in the batch.")
        target_calculator_choice = TargetCalculatorType(int(input("Please enter your choice: ")))

        print("Choose DatasetCreator from the following options:")
        print("1. Basic - Calculate energy per second for each sample before calculating target.")
        print("2. Aggregate - Each batch contains a single sample per process where the telemetry is summed.")
        print("3. Energy based Aggregate - Same aggregations as option 2, with 'first' and 'last' aggregations on the battery drain.")
        print("4. Processes Ratio - If a batch contains more than one process, calculate the impact of this process on the energy usage of the entire batch.")
        dataset_creator_choice = DatasetCreatorType(int(input("Please enter your choice: ")))

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
    else:
        raise ValueError(f"Unsupported implementation type {implementation_choice}!")

    full_dataset = dataset_creator.create_dataset()
    full_dataset.to_csv(FULL_DATASET_BEFORE_PROCESSING_PATH)
