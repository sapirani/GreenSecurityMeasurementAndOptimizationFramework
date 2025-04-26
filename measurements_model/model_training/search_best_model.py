import math

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from measurements_model.model_training.models_config import MODELS_WITHOUT_PARAMETERS

GRID_SEARCH_TEST_RESULTS_PATH = fr""
TEST_REAL_LABEL_COLUMN = "Actual"

class ModelSelector:

    def __init__(self, num_of_splits: int = 5, num_of_top_models: int = 5):
        self.__initial_model = ('classifier', LogisticRegression())
        self.__num_of_splits = num_of_splits + 1
        self.__num_of_top_models = num_of_top_models

    def __print_grid_search_top_models(self, best_models, score_method: str):
        print(f"~~~~ Best {self.__num_of_top_models} models based on {score_method} score ~~~~")
        for index, (score, model) in enumerate(best_models):
            print("*************** model number", index + 1, "***************")
            print(f"model {score_method} on train:", -1 * score)
            print("model params:", model)
            print()

    def __cv_splitter(self, prepared_labeled_df):
        for i in range(1, self.__num_of_splits):
            train_indices = [index for index in range(len(prepared_labeled_df)) if
                             index % (self.__num_of_splits - 1) != i]
            validation_indices = [index for index in range(len(prepared_labeled_df)) if
                                  index % (self.__num_of_splits - 1) == i]
            yield train_indices, validation_indices

    def __save_grid_search_results(self, final_results: pd.DataFrame, score_method: str):
        res_list = []
        for index, row in final_results.iterrows():
            score = row[f"mean_test_score"]

            if math.isnan(score):
                continue

            res_list.append((score, row["params"]))

        best_five_by_score = list(sorted(res_list, key=lambda x: x[0], reverse=True))[:self.__num_of_top_models]
        final_results.to_csv(f"final_results_{score_method}_metric_{datetime.now().strftime('%d_%m_%Y %H_%M')}.csv",
                             index=False)
        return best_five_by_score

    @classmethod
    def __print_scores_per_metric(cls, scores: dict[str, float]):
        print("*** Model's Accuracy on test-set ***")
        for metric, score in scores.items():
            print(f"{metric} value: {score}")

    def __calculate_and_print_scores(self, y: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        scores_per_metric = {}
        PER = (abs(y - y_pred) / y) * 100
        scores_per_metric["PER"] = PER
        scores_per_metric["Average PER"] = np.mean(PER)

        MSE = mean_squared_error(y_pred, y)
        scores_per_metric["MSE"] = MSE
        scores_per_metric["RMSE"] = math.sqrt(MSE)

        scores_per_metric["MAE"] = mean_absolute_error(y_pred, y)
        self.__print_scores_per_metric(scores_per_metric)
        return scores_per_metric

    def __select_top_models(self, x_train: pd.DataFrame, y_train: pd.Series, score_method: str):
        full_train_set = pd.concat([x_train, y_train], axis=1)
        # train = train.sort_values(by=[ProcessColumns.ENERGY_USAGE_PROCESS_COL])

        pipe = Pipeline([self.__initial_model])
        models = MODELS_WITHOUT_PARAMETERS[:3]
        grid = GridSearchCV(pipe, models, verbose=3, refit=True, cv=self.__cv_splitter(full_train_set),
                            scoring=score_method, n_jobs=-1)

        grid.fit(x_train, y_train)
        final_results = pd.DataFrame(grid.cv_results_)
        best_five_models = self.__save_grid_search_results(final_results, score_method)

        return best_five_models, grid.best_estimator_

    def __predict_and_print_best_estimator(self, best_estimator, x_test: pd.DataFrame, y_test: pd.Series,
                                           score_method: str):
        print(f"Best estimator from grid based {score_method} :")
        print(best_estimator)
        y_pred_test = best_estimator.predict(x_test)
        self.__calculate_and_print_scores(y_test, y_pred_test)
        return y_pred_test

    def choose_best_model(self, scoring_methods: list[str], x_train: pd.DataFrame, x_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series) -> dict[str, any]:
        best_estimator_per_metric = {}
        df_prediction_results = pd.DataFrame()
        df_prediction_results[TEST_REAL_LABEL_COLUMN] = y_test
        for scoring_method in scoring_methods:
            print(f"*** Select best {self.__num_of_top_models} regressors + their params: ")
            best_five_by_scorer, best_estimator_by_scorer = self.__select_top_models(x_train=x_train, y_train=y_train,
                                                                                     score_method=scoring_method)
            self.__print_grid_search_top_models(best_five_by_scorer, scoring_method)
            y_prediction_for_test = self.__predict_and_print_best_estimator(best_estimator=best_estimator_by_scorer,
                                                                            x_test=x_test, y_test=y_test,
                                                                            score_method=scoring_method)
            not_negative_scoring_method = scoring_method.replace("neg", "")
            df_prediction_results[f"Prediction_{not_negative_scoring_method}"] = y_prediction_for_test
            best_estimator_per_metric[f"Prediction_{not_negative_scoring_method}"] = best_estimator_by_scorer

        df_prediction_results.to_csv(GRID_SEARCH_TEST_RESULTS_PATH)
        return best_estimator_per_metric