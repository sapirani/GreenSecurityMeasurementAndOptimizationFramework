import math
from pathlib import Path

import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from measurements_model_pipeline.dataset_parameters import GRID_SEARCH_TEST_RESULTS_PATH, RESULTS_TOP_MODELS_PATH
from measurements_model_pipeline.utils import calculate_and_print_scores

TEST_REAL_LABEL_COLUMN = "Actual"
CLASSIFIER_KEYWORD = "classifier"
TEST_SCORE_FIELD_IN_GRID = "mean_test_score"
PARAMS_FIELD_IN_GRID = "params"


class ModelSelector:

    def __init__(self, models_to_experiment: list[dict], num_of_splits: int = 5, num_of_top_models: int = 5):
        self.__initial_model = (CLASSIFIER_KEYWORD, LogisticRegression())
        self.__num_of_splits = num_of_splits
        self.__num_of_top_models = num_of_top_models
        self.__models_to_experiment = models_to_experiment

    def __print_grid_search_top_models(self, best_models, score_method: str):
        print(f"~~~~~~~~~~~~~~~ Best {self.__num_of_top_models} models based on {score_method} score ~~~~~~~~~~~~~~~")
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
            score = row[TEST_SCORE_FIELD_IN_GRID]

            if math.isnan(score):
                continue

            res_list.append((score, row[PARAMS_FIELD_IN_GRID]))

        best_five_by_score = list(sorted(res_list, key=lambda x: x[0], reverse=True))[:self.__num_of_top_models]

        results_for_metric_path = Path(
            f"{RESULTS_TOP_MODELS_PATH}\\final_results_{score_method}_metric_{datetime.now().strftime('%d_%m_%Y %H_%M')}.csv")
        results_for_metric_path.parent.mkdir(parents=True, exist_ok=True)
        final_results.to_csv(results_for_metric_path, index=False)
        return best_five_by_score

    def __select_top_models(self, x_train: pd.DataFrame, y_train: pd.Series, score_method: str):
        pipe = Pipeline([self.__initial_model])
        kf_cv = KFold(n_splits=self.__num_of_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(pipe, self.__models_to_experiment, verbose=3, refit=True, cv=kf_cv, scoring=score_method,
                            n_jobs=1)

        grid.fit(x_train, y_train)
        final_results = pd.DataFrame(grid.cv_results_)
        best_five_models = self.__save_grid_search_results(final_results, score_method)

        return best_five_models, grid.best_estimator_

    def __predict_and_print_best_estimator(self, best_estimator, x_test: pd.DataFrame, y_test: pd.Series,
                                           score_method: str):
        print(f"Best estimator from grid based {score_method} :")
        print(best_estimator)
        y_pred_test = best_estimator.predict(x_test)
        y_pred_test = pd.Series(y_pred_test).reset_index(drop=True)
        calculate_and_print_scores(y_test, y_pred_test)
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
