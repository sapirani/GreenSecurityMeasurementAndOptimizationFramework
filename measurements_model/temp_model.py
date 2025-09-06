from measurements_model.dataset_creation.dataset_creator import DatasetCreator

IDLE_DIR_PATH = r"dataframes/cpu_consumer_20_usage_10_min.csv"
# todo: remove when moving permanently to the new approach.
if __name__ == "__main__":
    creator = DatasetCreator(
        idle_session_path=IDLE_DIR_PATH
    )

    df = creator.create_dataset()
    print(df.shape)



    # model = EnergyModel()
    # model.initialize_model()
    # model.predict(pd.DataFrame([[1, 2, 3], [4, 5, 6]]))