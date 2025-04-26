from abc import abstractmethod, ABC

import pandas as pd


class FeatureSelector(ABC):
    @abstractmethod
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Should remove the not relevant features from the dataset.
        Args:
            df: The dataset with all existing features

        Returns: Dataset after feature selection
        """
        pass
