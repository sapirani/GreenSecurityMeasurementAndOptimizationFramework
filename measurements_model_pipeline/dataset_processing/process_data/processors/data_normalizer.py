import pandas as pd
from sklearn import preprocessing

from measurements_model_pipeline.dataset_processing.process_data.processors.data_processor import Processor


class DataNormalizer(Processor):
    def __init__(self, columns_to_process: list[str]):
        self.__columns_to_process = columns_to_process

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        return pd.DataFrame(x_scaled)
