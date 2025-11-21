import pandas as pd
from sklearn.model_selection import KFold

from energy_model.configs.defaults_configs import DEFAULT_CV_SPLITS_N
from energy_model.dataset_processing.scalers.data_scaler import DataScaler
from energy_model.evaluation.model_evaluator import ModelEvaluator
from energy_model.models.model import Model
from energy_model.pipelines.pipeline_utils import extract_x_y


class ModelPipelineExecutor:
    def __init__(self, target_column: str):
        self.__target_column = target_column
        self.__model_evaluator = ModelEvaluator()

    def build_train_test_cv(self, df: pd.DataFrame, n_splits: int = DEFAULT_CV_SPLITS_N) \
            -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Return all KFold splits instead of a single split.
        """
        X, y = extract_x_y(df, self.__target_column)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []

        for train_idx, test_idx in kf.split(df):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            splits.append((X_train, X_test, y_train, y_test))

        return splits

    def build_scaler(self, X: pd.DataFrame) -> DataScaler:
        scaler = DataScaler()
        scaler.fit(X)
        return scaler

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series, scaler: DataScaler) -> Model:
        X_train_scaled = scaler.transform(X_train)
        model = Model()
        model.fit(X_train_scaled, y_train)
        return model

    def evaluate_model(self, model: Model, X_test: pd.DataFrame, y_test: pd.Series, scaler: DataScaler) \
            -> dict[str, float]:
        X_test_scaled = scaler.transform(X_test)
        y_pred = pd.Series(model.predict(X_test_scaled)).reset_index(drop=True)

        negative_predictions_mask = y_pred.lt(0)
        if negative_predictions_mask.any():
            negative_predictions = y_pred[negative_predictions_mask]
            print("Negative predictions found:")
            print(f"  Values : {negative_predictions.tolist()}")
            print(f"  Indices: {negative_predictions.index.tolist()}")

        results = self.__model_evaluator.evaluate(y_test, y_pred)
        self.__model_evaluator.print_results(results)
        return results
