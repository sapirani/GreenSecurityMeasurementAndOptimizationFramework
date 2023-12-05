# Different options for features in the dataset
import os
from abc import abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

from SapirModel.MeasurementConstants import TRAIN_SET_PATH, TEST_SET_PATH, SystemColumns, IDLEColumns, \
    DATASETS_DIRECTORY, ProcessColumns, TRAIN_SET_AFTER_PROCESSING_PATH, TEST_SET_AFTER_PROCESSING_PATH
from Scanner.general_consts import HardwareColumns


# Interface for choosing the train and test sets
class TrainTestSplitterInterface:
    def __init__(self, feature_selector):
        self.full_df = pd.read_csv(TRAIN_SET_PATH)
        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        self.feature_selector = feature_selector
        self.train_test_load()

    def create_dataset_x_y(self, df, path_after_processing):
        if os.path.isfile(path_after_processing):
            existing_df = pd.read_csv(path_after_processing)
            return self.feature_selector.get_x_y_df_by_col(existing_df, ProcessColumns.ENERGY_USAGE_PROCESS_COL)
        else:
            df = self.feature_selector.remove_features(df)
            df = self.feature_selector.preprocess_dataset(df)

            df.to_csv(path_after_processing, index=False)
            return self.feature_selector.get_x_y_df_by_col(df, ProcessColumns.ENERGY_USAGE_PROCESS_COL)

    def create_train_set(self):
        """
                Remove unnecessary features from train set,
                There are several options for that
                return: train set with the relevant features
                """

        """if os.path.isfile(TRAIN_SET_AFTER_PROCESSING_PATH):
            self.train_x, self.train_y = self.feature_selector.get_x_y_from_file(TRAIN_SET_AFTER_PROCESSING_PATH)

        else:
            self.train_x = self.feature_selector.remove_features(self.train_x)
            self.train_x = self.feature_selector.preprocess_dataset(self.train_x)

            full_df_after_pre_process = self.train_x
            full_df_after_pre_process[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = self.train_y
            full_df_after_pre_process.to_csv(TRAIN_SET_AFTER_PROCESSING_PATH)"""

        return self.create_dataset_x_y(self.train_set, TRAIN_SET_AFTER_PROCESSING_PATH)

    def create_test_set(self):
        """
                Remove unnecessary features from test set,
                There are several options for that
                return: test set with the relevant features
                """
        """if os.path.isfile(TEST_SET_AFTER_PROCESSING_PATH):
            self.test_x, self.test_y = self.feature_selector.get_x_y_from_file(TEST_SET_AFTER_PROCESSING_PATH)

        else:
            self.test_x = self.feature_selector.remove_features(self.test_x)
            self.test_x = self.feature_selector.preprocess_dataset(self.test_x)

            full_df_after_pre_process = self.test_x
            full_df_after_pre_process[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = self.test_y
            full_df_after_pre_process.to_csv(TEST_SET_AFTER_PROCESSING_PATH)"""

        return self.create_dataset_x_y(self.test_set, TEST_SET_AFTER_PROCESSING_PATH)

    @abstractmethod
    def train_test_load(self):
        pass


class RegularTrainTestSplit(TrainTestSplitterInterface):
    def train_test_load(self):
        full_x, full_y = self.feature_selector.get_x_y_df(self.full_df)
        train_x, test_x, train_y, test_y = train_test_split(full_x, full_y, test_size=0.2)
        self.train_set = self.feature_selector.concat_x_y(train_x, train_y)
        self.test_set = self.feature_selector.concat_x_y(test_x, test_y)


class CyberTestSplit(TrainTestSplitterInterface):
    def train_test_load(self):
        self.train_set = self.full_df
        self.test_set = pd.read_csv(TEST_SET_PATH)


# *** Interface for choosing features
class FeatureSelectorInterface:

    def preprocess_dataset(self, df):
        df = pd.get_dummies(df)
        return df

    def get_x_y_df(self, df):
        return df.iloc[:, :-1], df.iloc[:, -1:]


    def get_x_y_df_by_col(self, df, col):
        return df.loc[:, df.columns != col], df[col]

    def concat_x_y(self, x, y):
        full_df = x
        full_df[ProcessColumns.ENERGY_USAGE_PROCESS_COL] = y
        return full_df

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
                        IDLEColumns.DISK_WRITE_COUNT_IDLE_COL, IDLEColumns.DURATION_COL],
                       axis=1)



class WithoutSystem(ProcessAndFullSystem):
    def remove_features(self, df):
        df = super().remove_features(df)
        return df.drop([SystemColumns.CPU_SYSTEM_COL, SystemColumns.MEMORY_SYSTEM_COL,
                        SystemColumns.DISK_READ_BYTES_SYSTEM_COL, SystemColumns.DISK_READ_COUNT_SYSTEM_COL,
                        SystemColumns.DISK_WRITE_BYTES_SYSTEM_COL, SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL,
                        SystemColumns.DISK_READ_TIME, SystemColumns.DISK_WRITE_TIME, SystemColumns.PAGE_FAULT_SYSTEM_COL],
                       axis=1)

class WithoutHardware(ProcessAndFullSystem):
    def remove_features(self, df):
        df = super().remove_features(df)
        return df.drop([HardwareColumns.PC_TYPE, HardwareColumns.PC_MANUFACTURER, HardwareColumns.SYSTEM_FAMILY, HardwareColumns.MACHINE_TYPE,
                   HardwareColumns.DEVICE_NAME, HardwareColumns.OPERATING_SYSTEM, HardwareColumns.OPERATING_SYSTEM_RELEASE, HardwareColumns.OPERATING_SYSTEM_VERSION,
                   HardwareColumns.PROCESSOR_NAME, HardwareColumns.PROCESSOR_PHYSICAL_CORES, HardwareColumns.PROCESSOR_TOTAL_CORES, HardwareColumns.PROCESSOR_MAX_FREQ,
                   HardwareColumns.PROCESSOR_MIN_FREQ, HardwareColumns.TOTAL_RAM,
                   HardwareColumns.PHYSICAL_DISK_NAME, HardwareColumns.PHYSICAL_DISK_MANUFACTURER, HardwareColumns.PHYSICAL_DISK_MODEL,
                   HardwareColumns.PHYSICAL_DISK_MEDIA_TYPE, HardwareColumns.LOGICAL_DISK_NAME, HardwareColumns.LOGICAL_DISK_MANUFACTURER,
                   HardwareColumns.LOGICAL_DISK_MODEL, HardwareColumns.LOGICAL_DISK_DISK_TYPE, HardwareColumns.LOGICAL_DISK_PARTITION_STYLE,
                   HardwareColumns.LOGICAL_DISK_NUMBER_OF_PARTITIONS, HardwareColumns.PHYSICAL_SECTOR_SIZE, HardwareColumns.LOGICAL_SECTOR_SIZE,
                   HardwareColumns.BUS_TYPE, HardwareColumns.FILESYSTEM, HardwareColumns.BATTERY_DESIGN_CAPACITY, HardwareColumns.FULLY_CHARGED_BATTERY_CAPACITY],
                       axis=1)