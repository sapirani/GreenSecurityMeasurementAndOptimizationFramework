import pandas as pd

from measurements_model.dataset_processing.process_data.processors.data_processor import Processor


class CategoricalVariableProcessor(Processor):
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert categorical columns to columns of binary values -> e.g. city(a, b) turns to a -> t,f,f; b -> f,t,f, etc.
        df_with_categorical_columns = pd.get_dummies(df)
        return df_with_categorical_columns