import math
from datetime import datetime
from math import sqrt

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from SapirModel.DatasetFeatureSelection import ProcessAndFullSystem, RegularTrainTestSplit, CyberTestSplit, \
    WithoutHardware
from SapirModel.MeasurementConstants import TRAIN_SET_AFTER_PROCESSING_PATH, TEST_SET_AFTER_PROCESSING_PATH, \
    ProcessColumns, SystemColumns


import numpy as np
import optuna as optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import QuantileRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lars
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LarsCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.svm import LinearSVR
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.cross_decomposition import PLSCanonical
from sklearn.linear_model import RidgeCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold

from SapirModel.DatasetFeatureSelection import ProcessAndFullSystem, RegularTrainTestSplit

BEST_MODELS = 5

# TODO: hyperparameters
MLPRegressorModel = {"classifier": [MLPRegressor()],
                     "classifier__hidden_layer_sizes": [1, 20, 10, 15, 35, 50],
                     "classifier__activation": ["identity", "logistic", "tanh", "relu"],
                     "classifier__solver": ["lbfgs", "sgd", "adam"],
                     "classifier__alpha": [0.00005, 0.0005, 0.005, 0.00010, 0.001]}

"""all_regressors_algs = [LogisticRegressionModel, AdaBoostRegressorModel, ARDRegressionModel, BaggingRegressorModel,
                       BayesianRidgeModel, DecisionTreeRegressorModel, GradientBoostingRegressorModel,
                       MLPRegressorModel, LinearRegressionModel, LinearSVRModel, LassoLarsModel, LassoLarsCVModel,
                       LassoLarsICModel, LassoModel, LassoCVModel, LarsModel, LarsCVModel, ExtraTreeRegressorModel,
                       ElasticNetModel, ElasticNetCVModel, DummyRegressorModel, KNeighborsRegressorModel,
                       HuberRegressorModel, HistGradientBoostingRegressorModel, ExtraTreesRegressorModel,
                       SGDRegressorModel, RandomForestRegressorModel, RidgeCVModel, RidgeModel, RANSACRegressorModel,
                       PLSRegressionModel, PLSCanonicalModel, TheilSenRegressorModel, TransformedTargetRegressorModel,
                       TweedieRegressorModel, OrthogonalMatchingPursuitCVModel, OrthogonalMatchingPursuitModel,
                       PassiveAggressiveRegressorModel, LogisticRegressionModel, AdaBoostRegressorModel]"""

all_neural_networks_and_params = [MLPRegressorModel]  # TODO: add more regressors



def print_scores(y, y_pred):
    print("*** Model's Accuracy ***")


    MSE = mean_squared_error(y_pred, y)
    print(f"MSE value: {MSE}")
    print(f"RMSE value: {sqrt(MSE)}")

    MAE = mean_absolute_error(y_pred, y)
    print(f"MAE value: {MAE}")


def get_x_y_df_by_col(df):
    return df.loc[:, df.columns != ProcessColumns.ENERGY_USAGE_PROCESS_COL], df[ProcessColumns.ENERGY_USAGE_PROCESS_COL]


def save_grid_search_results(final_results, scorer):
    res_list = []
    for index, row in final_results.iterrows():
        score = row[f"mean_test_score"]

        if math.isnan(score):
            continue

        res_list.append((score, row["params"]))

    best_five_by_score = list(sorted(res_list, key=lambda x: x[0], reverse=True))[:5]
    final_results.to_csv(f"final_results_{scorer}_metric_{datetime.now().strftime('%d_%m_%Y %H_%M')}.csv", index=False)
    return best_five_by_score

def print_grid_search_top_models(best_models, scorer):
    print(f"~~~~ Best {5} models based on {scorer} score ~~~~")
    for index, (score, model) in enumerate(best_models):
        print("*************** model number", index + 1, "***************")
        print(f"model {scorer} on train:", -1 * score)
        print("model params:", model)
        print()


def normalize_data(df):
    # CPU_PROCESS_COL = "cpu_usage_process"
    #     MEMORY_PROCESS_COL = "memory_usage_process"
    #     DISK_READ_BYTES_PROCESS_COL = "disk_read_bytes_usage_process"
    #     DISK_READ_COUNT_PROCESS_COL = "disk_read_count_usage_process"
    #     DISK_WRITE_BYTES_PROCESS_COL = "disk_write_bytes_usage_process"
    #     DISK_WRITE_COUNT_PROCESS_COL = "disk_write_count_usage_process"
    #     PAGE_FAULTS_PROCESS_COL = "number_of_page_faults_process"
    normalized_df = df
    normalized_df[ProcessColumns.CPU_PROCESS_COL] = (df[ProcessColumns.CPU_PROCESS_COL] - df[ProcessColumns.CPU_PROCESS_COL].min()) \
                                                    / (df[ProcessColumns.CPU_PROCESS_COL].max() - df[ProcessColumns.CPU_PROCESS_COL].min())

    normalized_df[ProcessColumns.MEMORY_PROCESS_COL] = (df[ProcessColumns.MEMORY_PROCESS_COL] - df[ProcessColumns.MEMORY_PROCESS_COL].min()) \
                                                    / (df[ProcessColumns.MEMORY_PROCESS_COL].max() - df[ProcessColumns.MEMORY_PROCESS_COL].min())

    normalized_df[ProcessColumns.DISK_READ_BYTES_PROCESS_COL] = (df[ProcessColumns.DISK_READ_BYTES_PROCESS_COL] - df[ProcessColumns.DISK_READ_BYTES_PROCESS_COL].min()) \
                                                    / (df[ProcessColumns.DISK_READ_BYTES_PROCESS_COL].max() - df[ProcessColumns.DISK_READ_BYTES_PROCESS_COL].min())

    normalized_df[ProcessColumns.DISK_READ_COUNT_PROCESS_COL] = (df[ProcessColumns.DISK_READ_COUNT_PROCESS_COL] - df[ProcessColumns.DISK_READ_COUNT_PROCESS_COL].min()) \
                                                    / (df[ProcessColumns.DISK_READ_COUNT_PROCESS_COL].max() - df[ProcessColumns.DISK_READ_COUNT_PROCESS_COL].min())

    normalized_df[ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL] = (df[ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL] - df[ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL].min()) \
                                                                / (df[ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL].max() - df[ProcessColumns.DISK_WRITE_BYTES_PROCESS_COL].min())

    normalized_df[ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL] = (df[ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL] - df[ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL].min()) \
                                                                / (df[ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL].max() - df[ProcessColumns.DISK_WRITE_COUNT_PROCESS_COL].min())

    normalized_df[ProcessColumns.PAGE_FAULTS_PROCESS_COL] = (df[ProcessColumns.PAGE_FAULTS_PROCESS_COL] - df[ProcessColumns.PAGE_FAULTS_PROCESS_COL].min()) \
                                                                 / (df[ProcessColumns.PAGE_FAULTS_PROCESS_COL].max() - df[ProcessColumns.PAGE_FAULTS_PROCESS_COL].min())

    normalized_df[SystemColumns.CPU_SYSTEM_COL] = (df[SystemColumns.CPU_SYSTEM_COL] - df[SystemColumns.CPU_SYSTEM_COL].min()) \
                                                    / (df[SystemColumns.CPU_SYSTEM_COL].max() - df[SystemColumns.CPU_SYSTEM_COL].min())

    normalized_df[SystemColumns.MEMORY_SYSTEM_COL] = (df[SystemColumns.MEMORY_SYSTEM_COL] - df[SystemColumns.MEMORY_SYSTEM_COL].min()) \
                                                       / (df[SystemColumns.MEMORY_SYSTEM_COL].max() - df[SystemColumns.MEMORY_SYSTEM_COL].min())

    normalized_df[SystemColumns.DISK_READ_BYTES_SYSTEM_COL] = (df[SystemColumns.DISK_READ_BYTES_SYSTEM_COL] - df[SystemColumns.DISK_READ_BYTES_SYSTEM_COL].min()) \
                                                                / (df[SystemColumns.DISK_READ_BYTES_SYSTEM_COL].max() - df[SystemColumns.DISK_READ_BYTES_SYSTEM_COL].min())

    normalized_df[SystemColumns.DISK_READ_COUNT_SYSTEM_COL] = (df[SystemColumns.DISK_READ_COUNT_SYSTEM_COL] - df[SystemColumns.DISK_READ_COUNT_SYSTEM_COL].min()) \
                                                                / (df[SystemColumns.DISK_READ_COUNT_SYSTEM_COL].max() - df[SystemColumns.DISK_READ_COUNT_SYSTEM_COL].min())

    normalized_df[SystemColumns.DISK_READ_BYTES_SYSTEM_COL] = (df[SystemColumns.DISK_READ_BYTES_SYSTEM_COL] - df[SystemColumns.DISK_READ_BYTES_SYSTEM_COL].min()) \
                                                                 / (df[SystemColumns.DISK_READ_BYTES_SYSTEM_COL].max() - df[SystemColumns.DISK_READ_BYTES_SYSTEM_COL].min())

    normalized_df[SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL] = (df[SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL] - df[SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL].min()) \
                                                                 / (df[SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL].max() - df[SystemColumns.DISK_WRITE_COUNT_SYSTEM_COL].min())

    normalized_df[SystemColumns.DISK_READ_TIME] = (df[SystemColumns.DISK_READ_TIME] - df[SystemColumns.DISK_READ_TIME].min()) \
                                                / (df[SystemColumns.DISK_READ_TIME].max() - df[SystemColumns.DISK_READ_TIME].min())

    normalized_df[SystemColumns.DISK_WRITE_TIME] = (df[SystemColumns.DISK_WRITE_TIME] - df[SystemColumns.DISK_WRITE_TIME].min()) \
                                                / (df[SystemColumns.DISK_WRITE_TIME].max() - df[SystemColumns.DISK_WRITE_TIME].min())

    normalized_df[SystemColumns.PAGE_FAULT_SYSTEM_COL] = (df[SystemColumns.PAGE_FAULT_SYSTEM_COL] - df[SystemColumns.PAGE_FAULT_SYSTEM_COL].min()) \
                                                / (df[SystemColumns.PAGE_FAULT_SYSTEM_COL].max() - df[SystemColumns.PAGE_FAULT_SYSTEM_COL].min())

    return normalized_df


def main():
    feature_selector = WithoutHardware()
    splitter = RegularTrainTestSplit(feature_selector)
    x_train, y_train = splitter.create_train_set()
    x_test, y_test = splitter.create_test_set()

    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)


    pipe = Pipeline([('classifier', MLPRegressor())])
    grid_mae = GridSearchCV(pipe, all_neural_networks_and_params, verbose=3, refit=True, cv=452, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_mae.fit(x_train, y_train)

    pipe = Pipeline([('classifier', MLPRegressor())])
    grid_rmse = GridSearchCV(pipe, all_neural_networks_and_params, verbose=3, refit=True, cv=452, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_rmse.fit(x_train, y_train)

    estimator_mae = grid_mae.best_estimator_
    estimator_rmse = grid_rmse.best_estimator_

    print("Best estimator MAE: ")
    print(estimator_mae)
    top_5_mae = save_grid_search_results(pd.DataFrame(grid_mae.cv_results_), "MAE")
    print_grid_search_top_models(top_5_mae, 'MAE')
    y_pred_mae = estimator_mae.predict(x_test)
    print_scores(y_test, y_pred_mae)

    print("Best estimator RMSE: ")
    print(estimator_rmse)
    top_5_rmse = save_grid_search_results(pd.DataFrame(grid_rmse.cv_results_), "RMSE")
    print_grid_search_top_models(top_5_rmse, 'RMSE')
    y_pred_mae = estimator_rmse.predict(x_test)
    print_scores(y_test, y_pred_mae)


main()
