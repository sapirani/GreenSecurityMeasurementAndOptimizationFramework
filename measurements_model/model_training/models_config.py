import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
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
from sklearn.linear_model import RidgeCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import SGDRegressor

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

ALL_REGRESSORS_MODELS = [LogisticRegressionModel, AdaBoostRegressorModel, ARDRegressionModel, BaggingRegressorModel,
                         BayesianRidgeModel, DecisionTreeRegressorModel, GradientBoostingRegressorModel,
                         MLPRegressorModel, LinearRegressionModel, LinearSVRModel, LassoLarsModel, LassoLarsCVModel,
                         LassoLarsICModel, LassoModel, LassoCVModel, LarsModel, LarsCVModel, ExtraTreeRegressorModel,
                         ElasticNetModel, ElasticNetCVModel, DummyRegressorModel, KNeighborsRegressorModel,
                         HuberRegressorModel, HistGradientBoostingRegressorModel, ExtraTreesRegressorModel,
                         SGDRegressorModel, RandomForestRegressorModel, RidgeCVModel, RidgeModel, RANSACRegressorModel,
                         PLSRegressionModel, TheilSenRegressorModel, TransformedTargetRegressorModel,
                         TweedieRegressorModel, PassiveAggressiveRegressorModel, LogisticRegressionModel,
                         AdaBoostRegressorModel]

MODELS_WITHOUT_PARAMETERS = [{"classifier": [TheilSenRegressor()]}, {"classifier": [TransformedTargetRegressor()]},
                             {"classifier": [TweedieRegressor()]}, {"classifier": [PassiveAggressiveRegressor()]},
                             {"classifier": [PLSRegression()]}, {"classifier": [RANSACRegressor()]},
                             {"classifier": [RandomForestRegressor()]}, {"classifier": [SGDRegressor()]},
                             {"classifier": [ExtraTreesRegressor()]}, {"classifier": [HistGradientBoostingRegressor()]},
                             {"classifier": [KNeighborsRegressor()]}, {"classifier": [ElasticNet()]},
                             {"classifier": [ExtraTreeRegressor()]}, {"classifier": [LinearRegression()]},
                             {"classifier": [MLPRegressor()]}, {"classifier": [GradientBoostingRegressor()]},
                             {"classifier": [DecisionTreeRegressor()]}, {"classifier": [BaggingRegressor()]},
                             {"classifier": [ARDRegression()]}, {"classifier": [AdaBoostRegressor()]}]

REGRESSION_MODELS_WITH_PARAMETERS = [LogisticRegressionModel, AdaBoostRegressorModel, ARDRegressionModel,
                                     BaggingRegressorModel, BayesianRidgeModel, DecisionTreeRegressorModel,
                                     GradientBoostingRegressorModel, MLPRegressorModel, LinearRegressionModel,
                                     LinearSVRModel, LassoLarsModel, LassoLarsCVModel, LassoLarsICModel, LassoModel,
                                     LassoCVModel, LarsModel, LarsCVModel, ExtraTreeRegressorModel, ElasticNetModel,
                                     ElasticNetCVModel, DummyRegressorModel, KNeighborsRegressorModel,
                                     HuberRegressorModel, HistGradientBoostingRegressorModel, ExtraTreesRegressorModel,
                                     SGDRegressorModel, RandomForestRegressorModel, RidgeCVModel, RidgeModel,
                                     RANSACRegressorModel, PLSRegressionModel, TheilSenRegressorModel,
                                     TransformedTargetRegressorModel, TweedieRegressorModel,
                                     PassiveAggressiveRegressorModel, LogisticRegressionModel, AdaBoostRegressorModel]
# TODO: add more regressors
