from measurements_model.dataset_creation.dataset_creator import DatasetCreator

if __name__ == "__main__":
    idle_path = r"C:\Users\Administrator\Desktop\green security\tmp - idle\Measurement 1"
    measurements_path = r"C:\Users\Administrator\Desktop\green security\tmp"
    creator = DatasetCreator(idle_dir_path=idle_path, measurements_dir_path=measurements_path)
    measurements = creator.create_dataset()
    print(measurements)