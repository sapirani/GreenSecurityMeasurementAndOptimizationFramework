# Different options for features in the dataset
from abc import abstractmethod

import pandas as pd

from SapirModel.MeasurementConstants import TRAIN_SET_PATH, TEST_SET_PATH, SystemColumns, IDLEColumns


class FeatureSelectorInterface:
    def __init__(self):
        self.train_df = pd.read_csv(TRAIN_SET_PATH)
        self.test_df = pd.read_csv(TEST_SET_PATH)

    def create_train_set(self):
        train_set_after_feature_removal = self.remove_features(self.train_df)
        """
        Remove unnecessary features from train set,
        There are several options for that
        return: train set with the relevant features
        """
        #self.train_df = train_set_after_feature_removal
        return train_set_after_feature_removal

    def create_test_set(self):
        test_set_after_feature_removal = self.remove_features(self.test_df)
        """
        Remove unnecessary features from test set,
        There are several options for that
        return: test set with the relevant features
        """
        #self.test_df = test_set_after_feature_removal
        return test_set_after_feature_removal

    @abstractmethod
    def remove_features(self, df):
        """
        Should remove the not relevant features from the dataset.
        Args:
            df: The dataset with all existing features

        Returns: Dataset after feature selection
        """
        pass


# TODO: should remove duration column?

class AllFeaturesNoEnergy(FeatureSelectorInterface):  # no subtraction in system column + idle features
    def remove_features(self, df):
        return df.drop([SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL, IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL], axis=1)


class ProcessAndFullSystem(FeatureSelectorInterface):  # no subtraction in system column
    def remove_features(self, df):
        return df.drop([SystemColumns.ENERGY_TOTAL_USAGE_SYSTEM_COL, IDLEColumns.ENERGY_TOTAL_USAGE_IDLE_COL,
                        IDLEColumns.CPU_IDLE_COL, IDLEColumns.MEMORY_IDLE_COL, IDLEColumns.PAGE_FAULT_IDLE_COL,
                        IDLEColumns.DISK_READ_BYTES_IDLE_COL, IDLEColumns.DISK_READ_COUNT_IDLE_COL,
                        IDLEColumns.DISK_READ_TIME, IDLEColumns.DISK_WRITE_TIME, IDLEColumns.DISK_WRITE_BYTES_IDLE_COL,
                        IDLEColumns.DISK_WRITE_COUNT_IDLE_COL, IDLEColumns.DURATION_COL, IDLEColumns.COMPARED_TO_IDLE],
                       axis=1)
