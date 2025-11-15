from energy_model.dataset_creation.process_based_dataset_creator import ProcessBasedDatasetCreator
from energy_model.dataset_creation.system_based_dataset_creator import SystemBasedDatasetCreator
from energy_model.configs.paths_config import FULL_DATASET_BEFORE_PROCESSING_PATH

if __name__ == '__main__':
    print("Welcome to Dataset Creator!")
    print("Choose from the following options:")
    print("1. Create process based dataset (energy column for process based on idle energy)")
    print("2. Create system based dataset (energy column for system is the battery drain between samples)")
    dataset_creator_choice = input("Please enter your input: ")
    if dataset_creator_choice == "1":
        dataset_creator = ProcessBasedDatasetCreator()
    elif dataset_creator_choice == "2":
        dataset_creator = SystemBasedDatasetCreator()
    else:
        raise ValueError(f"You entered an invalid option {dataset_creator_choice}, Bye.")

    full_dataset = dataset_creator.create_dataset()
    full_dataset.to_csv(FULL_DATASET_BEFORE_PROCESSING_PATH)