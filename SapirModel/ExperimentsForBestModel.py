import math
from datetime import datetime
from math import sqrt

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

from SapirModel.DatasetFeatureSelection import ProcessAndFullSystem, RegularTrainTestSplit, WithoutHardware, \
    WithoutSystem
from SapirModel.MeasurementConstants import ProcessColumns

BEST_MODELS = 5

# TODO: hyperparameters
LogisticRegressionModel = {"classifier": [LogisticRegression()],
                           "classifier__class_weight": ["balanced", None],
                           "classifier__solver": ["liblinear", "lbfgs"],
                           "classifier__C": [0.01, 0.1, 0.2, 0.3, 0.5],
                           "classifier__tol": [0.0001, 0.001, 0.000001, 0.1]}
AdaBoostRegressorModel = {"classifier": [AdaBoostRegressor()],
                          "classifier__learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                          "classifier__n_estimators": [10, 50, 100, 500]}
ARDRegressionModel = {"classifier": [ARDRegression()],
                      "classifier__compute_score": [True, False],
                      "classifier__threshold_lambda": [10000, 5000, 7500, 2500],
                      "classifier__alpha_1": [0.000001, 0.00001, 0.0001]}
BaggingRegressorModel = {"classifier": [BaggingRegressor()],
                         'classifier__bootstrap': [True, False],
                         'classifier__bootstrap_features': [False],
                         'classifier__max_features': [1, 2, 3],
                         'classifier__n_estimators': [10, 50, 100, 1000],
                         "classifier__max_samples": [0.05, 0.1, 0.5]}
BayesianRidgeModel = {"classifier": [BayesianRidge()],
                      "classifier__alpha_init": [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9],
                      "classifier__lambda_init": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-9]}
DecisionTreeRegressorModel = {"classifier": [DecisionTreeRegressor()],
                              "classifier__criterion": ['squared_error', 'absolute_error'],
                              "classifier__max_depth": np.arange(1, 21).tolist()[0::2],
                              "classifier__min_samples_split": np.arange(2, 11).tolist()[0::2],
                              "classifier__max_leaf_nodes": np.arange(3, 26).tolist()[0::2]}
GradientBoostingRegressorModel = {"classifier": [GradientBoostingRegressor()],
                                  'classifier__max_depth': [80, 90, 100, 110],
                                  'classifier__max_features': [2, 3],
                                  'classifier__min_samples_leaf': [3, 4, 5],
                                  'classifier__min_samples_split': [8, 10, 12],
                                  'classifier__n_estimators': [100, 200, 300, 1000]}
MLPRegressorModel = {"classifier": [MLPRegressor()],
                     "classifier__hidden_layer_sizes": [1, 50],
                     "classifier__activation": ["identity", "logistic", "tanh", "relu"],
                     "classifier__solver": ["lbfgs", "sgd", "adam"],
                     "classifier__alpha": [0.00005, 0.0005]}
LinearRegressionModel = {"classifier": [LinearRegression()],
                         "classifier__fit_intercept": [True, False]}
LinearSVRModel = {"classifier": [LinearSVR()],
                  'classifier__epsilon': [0.],
                  'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
LassoLarsICModel = {"classifier": [LassoLarsIC()]}
LassoLarsCVModel = {"classifier": [LassoLarsCV()]}
LassoLarsModel = {"classifier": [LassoLars()],
                  "classifier__alpha": np.geomspace(1e-8, 1e-6, 10)}
LassoCVModel = {"classifier": [LassoCV()]}
LassoModel = {"classifier": [Lasso()],
              "classifier__alpha": np.arange(0.0001, 0.01, 0.0005)}
LarsCVModel = {"classifier": [LarsCV()]}
LarsModel = {"classifier": [Lars()]}
ExtraTreeRegressorModel = {"classifier": [ExtraTreeRegressor()],
                           # 'classifier__n_estimators': [10, 50, 100],
                           'classifier__criterion': ['squared_error', 'absolute_error'],
                           'classifier__max_depth': [2, 8, 16, 32, 50],
                           'classifier__min_samples_split': [2, 4, 6],
                           'classifier__min_samples_leaf': [1, 2],
                           # 'oob_score': [True, False],
                           'classifier__max_features': ['auto', 'sqrt', 'log2']}
ElasticNetModel = {"classifier": [ElasticNet()],
                   "classifier__max_iter": [1, 5, 10, 1000],
                   "classifier__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                   "classifier__l1_ratio": np.arange(0.0, 1.0, 0.1)}
ElasticNetCVModel = {"classifier": [ElasticNetCV()],
                     'classifier__alphas': [[0.1], [1], [10], [0.01]],
                     'classifier__l1_ratio': np.arange(0.40, 1.00, 0.10),
                     'classifier__tol': [0.0001, 0.001]}
DummyRegressorModel = {"classifier": [DummyRegressor()]}
KNeighborsRegressorModel = {"classifier": [KNeighborsRegressor()],
                            'classifier__n_neighbors': [3, 7, 15, 25, 30]}
HuberRegressorModel = {"classifier": [HuberRegressor()],
                       "classifier__max_iter": [100, 200, 500, 1000],
                       "classifier__alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
                       "classifier__epsilon": [1.15, 1.25, 1.35, 1.5]}
HistGradientBoostingRegressorModel = {"classifier": [HistGradientBoostingRegressor()],
                                      'classifier__max_depth': range(5, 16, 2),
                                      'classifier__min_samples_leaf': range(10, 100, 10)}
ExtraTreesRegressorModel = {"classifier": [ExtraTreesRegressor()],
                            "classifier__max_depth": [3, 5, 7, 12],
                            "classifier__min_samples_leaf": [3, 5, 7],
                            "classifier__min_weight_fraction_leaf": [0.1, 0.2, 0.5],
                            "classifier__max_features": ["auto", "log2", "sqrt"],
                            "classifier__max_leaf_nodes": [10, 30, 60, 80, 90]}
SGDRegressorModel = {"classifier": [SGDRegressor()],
                     'classifier__max_iter': [100000, 1000000],
                     'classifier__tol': [1e-10, 1e-3],
                     'classifier__eta0': [0.001, 0.01]}
RandomForestRegressorModel = {"classifier": [RandomForestRegressor()],
                              'classifier__n_estimators': [50, 100, 200, 500, 1000],
                              'classifier__max_features': ['auto', 'sqrt'],
                              'classifier__max_depth': [5, 7, 15, 30, 70],
                              'classifier__min_samples_split': [2, 5, 10],
                              'classifier__min_samples_leaf': [1, 2, 4]}
RidgeCVModel = {"classifier": [RidgeCV()]}
RidgeModel = {"classifier": [Ridge()],
              'classifier__alpha': [1, 0.1, 0.01, 0.0001],
              "classifier__fit_intercept": [True, False],
              "classifier__solver": ['svd', 'lsqr', 'saga']}
RANSACRegressorModel = {"classifier": [RANSACRegressor()]}
PLSRegressionModel = {"classifier": [PLSRegression()],
                      'classifier__n_components': [1, 2, 5, 9, 15, 20]}
PassiveAggressiveRegressorModel = {"classifier": [PassiveAggressiveRegressor()],
                                   "classifier__C": [0.0001, 0.01, 0.1, 1.0, 10],
                                   "classifier__epsilon": [0.0001, 0.001, 0.01, 0.1]}
TweedieRegressorModel = {"classifier": [TweedieRegressor()]}
TransformedTargetRegressorModel = {"classifier": [TransformedTargetRegressor()]}
TheilSenRegressorModel = {"classifier": [TheilSenRegressor()]}

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

algs_no_params = [{"classifier": [TheilSenRegressor()]}, {"classifier": [TransformedTargetRegressor()]},
                  {"classifier": [TweedieRegressor()]}, {"classifier": [PassiveAggressiveRegressor()]},
                  {"classifier": [PLSRegression()]}, {"classifier": [RANSACRegressor()]},
                  {"classifier": [RandomForestRegressor()]}, {"classifier": [SGDRegressor()]},
                  {"classifier": [ExtraTreesRegressor()]}, {"classifier": [HistGradientBoostingRegressor()]},
                  {"classifier": [KNeighborsRegressor()]}, {"classifier": [ElasticNet()]},
                  {"classifier": [ExtraTreeRegressor()]}, {"classifier": [LinearRegression()]},
                  {"classifier": [MLPRegressor()]}, {"classifier": [GradientBoostingRegressor()]},
                  {"classifier": [DecisionTreeRegressor()]}, {"classifier": [BaggingRegressor()]},
                  {"classifier": [ARDRegression()]}, {"classifier": [AdaBoostRegressor()]}]

all_regressors_algs_and_params = [LogisticRegressionModel, AdaBoostRegressorModel, ARDRegressionModel,
                                  BaggingRegressorModel,
                                  BayesianRidgeModel, DecisionTreeRegressorModel, GradientBoostingRegressorModel,
                                  MLPRegressorModel, LinearRegressionModel, LinearSVRModel, LassoLarsModel,
                                  LassoLarsCVModel,
                                  LassoLarsICModel, LassoModel, LassoCVModel, LarsModel, LarsCVModel,
                                  ExtraTreeRegressorModel,
                                  ElasticNetModel, ElasticNetCVModel, DummyRegressorModel, KNeighborsRegressorModel,
                                  HuberRegressorModel, HistGradientBoostingRegressorModel, ExtraTreesRegressorModel,
                                  SGDRegressorModel, RandomForestRegressorModel, RidgeCVModel, RidgeModel,
                                  RANSACRegressorModel,
                                  PLSRegressionModel, TheilSenRegressorModel, TransformedTargetRegressorModel,
                                  TweedieRegressorModel, PassiveAggressiveRegressorModel, LogisticRegressionModel,
                                  AdaBoostRegressorModel]  # TODO: add more regressors

feature_selector = ProcessAndFullSystem()
splitter = RegularTrainTestSplit(feature_selector)
x_train, y_train = splitter.create_train_set()
x_test, y_test = splitter.create_test_set()

"""def print_final_results_on_test(top_models):
    # {'classifier': LogisticRegression(), 'classifier__C': 0.01, 'classifier__class_weight': None, 'classifier__solver': 'lbfgs', 'classifier__tol': 0.1}
    for (train_cv_score, model_params) in top_models:
        print(model_params)
        print(type(model_params))
        regressor = model_params['classifier']
        print(type(regressor))
        del model_params['classifier']
        params = {key[12:]: val for key,val in model_params.items()}
        model = regressor(**params)
        model.fit(x_train, y_train)
        y_pred_test = model.predict(x_test)
        print_scores(y_test, y_pred_test)"""

"""def compare_models(x_train, y_train, x_validation, y_validation, use_pca=True):
    models_and_scores = []

    for dict in all_regressors_algs:
        alg_model = dict["classifier"]
        alg_name = type(alg_model).__name__
        pipe = make_pipeline(PCA(), alg_model()) if use_pca else alg_model()
        pipe.fit(x_train, y_train)
        y_predictions = pipe.predict(x_validation)
        accuracy_score = mean_absolute_error(y_validation, y_predictions)
        print("The mean absolute error ({0} PCA) of {1} is: {2}".format("with" if use_pca else "without", alg_name,
                                                                        accuracy_score))

        models_and_scores.append((alg_name, accuracy_score))

    return models_and_scores"""

"""def print_and_save_grid_search_results(final_results):
    res_list = []
    for index, row in final_results.iterrows():
        rmse = row[f"mean_test_neg_root_mean_squared_error"]
        mae = row[f"mean_test_neg_mean_absolute_error"]

        if math.isnan(rmse) or math.isnan(mae):
            continue

        res_list.append((mae, rmse, row["params"]))


    all_models_sorted_by_mae = list(sorted(res_list, key=lambda x: x[0], reverse=True))
    all_models_sorted_by_rmse = list(sorted(res_list, key=lambda x: x[1], reverse=True))

    best_five_by_mae = all_models_sorted_by_mae[:BEST_MODELS]
    best_five_by_rmse = all_models_sorted_by_rmse[:BEST_MODELS]

    print(f"~~~~ Best {BEST_MODELS} models based on MAE score ~~~~")
    for index, (neg_mae_score, neg_rmse_score, model) in enumerate(best_five_by_mae):
        print("*************** model number", index + 1, "***************")
        print("model MAE:", -1*neg_mae_score)
        print("model RMSE:", -1*neg_rmse_score)
        print("model params:", model)
        print()

    print(f"~~~~ Best {BEST_MODELS} models based on RMSE score ~~~~")
    for index, (neg_mae_score, neg_rmse_score, model) in enumerate(best_five_by_rmse):
        print("*************** model number", index + 1, "***************")
        print("model MAE:", -1 * neg_mae_score)
        print("model RMSE:", -1 * neg_rmse_score)
        print("model params:", model)
        print()

    final_results.to_csv(f"final_results_{datetime.now().strftime('%d_%m_%Y %H_%M')}.csv", index=False)
    return best_five_by_mae, best_five_by_rmse"""


def print_scores(y, y_pred):
    print("*** Model's Accuracy on test-set ***")
    PER = (abs(y - y_pred) / y) * 100
    print(f"PER value: {PER}")
    print(f"Average Per: {np.mean(PER)}")

    MSE = mean_squared_error(y_pred, y)
    print(f"MSE value: {MSE}")
    print(f"RMSE value: {sqrt(MSE)}")

    MAE = mean_absolute_error(y_pred, y)
    print(f"MAE value: {MAE}")


def cv_splitter(prepared_labeled_df):
    for i in range(1, 6):
        train_indices = [index for index in range(len(prepared_labeled_df)) if index % 5 != i]
        validation_indices = [index for index in range(len(prepared_labeled_df)) if index % 5 == i]
        yield train_indices, validation_indices



def save_grid_search_results(final_results, scorer):
    res_list = []
    for index, row in final_results.iterrows():
        score = row[f"mean_test_score"]

        if math.isnan(score):
            continue

        res_list.append((score, row["params"]))

    best_five_by_score = list(sorted(res_list, key=lambda x: x[0], reverse=True))[:BEST_MODELS]
    final_results.to_csv(f"final_results_{scorer}_metric_{datetime.now().strftime('%d_%m_%Y %H_%M')}.csv", index=False)
    return best_five_by_score


def select_top_models_with_grid(scorer):
    global x_train, y_train
    train = feature_selector.concat_x_y(x_train, y_train)
    train = train.sort_values(by=[ProcessColumns.ENERGY_USAGE_PROCESS_COL])
    pipe = Pipeline([('classifier', LogisticRegression())])
    grid = GridSearchCV(pipe, algs_no_params, verbose=3, refit=True, cv=cv_splitter(train), scoring=scorer, n_jobs=-1)
    x_train, y_train = feature_selector.get_x_y_df_by_col(train, ProcessColumns.ENERGY_USAGE_PROCESS_COL)
    grid.fit(x_train, y_train.values.ravel())

    final_results = pd.DataFrame(grid.cv_results_)
    best_five_models = save_grid_search_results(final_results, scorer)

    return best_five_models, grid.best_estimator_


def print_grid_search_top_models(best_models, scorer):
    print(f"~~~~ Best {BEST_MODELS} models based on {scorer} score ~~~~")
    for index, (score, model) in enumerate(best_models):
        print("*************** model number", index + 1, "***************")
        print(f"model {scorer} on train:", -1 * score)
        print("model params:", model)
        print()


def main():

    df_prediction_results = pd.DataFrame()
    df_prediction_results["Actual"] = y_test
    print(f"*** Select best {BEST_MODELS} regressors + their params: ")
    best_five_by_mae, best_mae_estimator = select_top_models_with_grid('neg_mean_absolute_error')
    best_five_by_rmse, best_rmse_estimator = select_top_models_with_grid('neg_root_mean_squared_error')
    print("*** Final results for train set from grid search: ***")
    print_grid_search_top_models(best_five_by_mae, 'neg_mean_absolute_error')
    print_grid_search_top_models(best_five_by_rmse, 'neg_root_mean_squared_error')

    print("*** Best estimator based on grid search on train set: ***")
    print(f"Best estimator from grid based MAE :")
    print(best_mae_estimator)
    y_pred_test = best_mae_estimator.predict(x_test)
    df_prediction_results["Prediction_Mae"] = y_pred_test
    print_scores(y_test, y_pred_test)
    print(f"Best estimator from grid based RMSE :")
    print(best_rmse_estimator)
    y_pred_test = best_rmse_estimator.predict(x_test)
    df_prediction_results["Prediction_Rmse"] = y_pred_test
    print_scores(y_test, y_pred_test)
    df_prediction_results.to_csv("TestResults.csv")


if __name__ == '__main__':
    main()
