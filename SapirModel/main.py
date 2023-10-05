import pandas as pd

from SapirModel.CreateDataset import read_directories
from SapirModel.MeasurementConstants import *
from SapirModel.MeasurementModel import MeasurementModel


def initialize_dataset(is_train=True):
    cols = DATASET_COLUMNS + [ProcessColumns.ENERGY_USAGE_PROCESS_COL] if is_train else DATASET_COLUMNS
    return pd.DataFrame(columns=cols)


def fit(train_df, model):
    ## TODO: maybe change the values of target - for now its system_energy - idle_energy, maybe the target should be system_energy?
    model.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1:])


def predict(test_df, model):
    model.predict(test_df)


def main():
    print("======== Creating Train Dataset ========")
    train_df = initialize_dataset()
    train_df = read_directories(train_df, TRAIN_MEASUREMENTS_DIR_PATH, is_train=True)
    model = MeasurementModel()
    print("======== Training the Model ========")
    fit(train_df, model)
    print("======== Creating Test Dataset ========")
    test_df = initialize_dataset(False)
    test_df = read_directories(test_df, TRAIN_MEASUREMENTS_DIR_PATH, is_train=False)
    print("======== Predicting Energy using the model ========")
    predict(test_df, model)


if __name__ == '__main__':
    main()
