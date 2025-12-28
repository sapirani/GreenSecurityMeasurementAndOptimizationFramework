from energy_model.dataset_creation.dataset_creators.energy_extended_dataset_creator import EnergyExtendedDatasetCreator
from energy_model.dataset_creation.dataset_creators.process_based_dataset_creators.process_basic_dataset_creator import \
    ProcessBasicDatasetCreator
from energy_model.dataset_creation.dataset_creators.system_based_dataset_creators.system_basic_dataset_creator import \
    SystemBasicDatasetCreator
from energy_model.dataset_creation.dataset_creators.system_based_dataset_creators.system_extended_dataset_creator import \
    SystemExtendedDatasetCreator
from energy_model.energy_model_parameters import FULL_DATASET_BEFORE_PROCESSING_PATH

if __name__ == '__main__':
    print("Welcome to Dataset Creator!")
    print("Choose from the following options:")
    print("1. Create process based basic dataset (energy column for process based on idle energy)")
    print(
        "2. Create system based basic dataset (energy column for system is the battery drain between samples, without summing the process samples)")
    print(
        "3. Create system based extended dataset (energy column for system is the battery drain between samples, summing the process samples)")
    print(
        "4. Create system based energy-extended dataset (energy column for system is the battery drain between samples only (without batch-time), summing the process samples)")

    dataset_creator_choice = input("Please enter your input: ")
    if dataset_creator_choice == "1":
        dataset_creator = ProcessBasicDatasetCreator()
    elif dataset_creator_choice == "2":
        dataset_creator = SystemBasicDatasetCreator()
    elif dataset_creator_choice == "3":
        dataset_creator = SystemExtendedDatasetCreator()
    elif dataset_creator_choice == "4":
        dataset_creator = EnergyExtendedDatasetCreator()
    else:
        raise ValueError(f"You entered an invalid option {dataset_creator_choice}, Bye.")

    full_dataset = dataset_creator.create_dataset()
    full_dataset.to_csv(FULL_DATASET_BEFORE_PROCESSING_PATH)
