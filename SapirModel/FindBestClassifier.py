import math
from datetime import datetime
from math import sqrt

import numpy as np
import optuna as optuna
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier

from SapirModel.DatasetFeatureSelection import ProcessAndFullSystem, RegularTrainTestSplit, WithoutHardware, \
    WithoutSystem

BEST_MODELS = 5

# TODO: hyperparameters

ExtraTreeClassifierModel = {"classifier": [ExtraTreeClassifier()]}
GradientBoostingClassifierModel = {"classifier": [GradientBoostingClassifier()],
                                   'classifier__n_estimators': [16, 32, 20], 'classifier__learning_rate': [0.8, 0.5, 1.0]}
AdaBoostClassifierModel = {"classifier": [AdaBoostClassifier()],
                         "classifier__learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],  "classifier__n_estimators": [1, 2, 10, 50, 100]}
HistGradientBoostingClassifierModel = {"classifier": [HistGradientBoostingClassifier()]}
PassiveAggressiveClassifierModel = {"classifier": [PassiveAggressiveClassifier()]}
RidgeClassifierCVModel = {"classifier": [RidgeClassifierCV()]}
MLPClassifierModel = {"classifier": [MLPClassifier()]}
LinearDiscriminantAnalysisModel = {"classifier": [LinearDiscriminantAnalysis()]}
LabelSpreadingModel = {"classifier": [LabelSpreading()]}
CalibratedClassifierCVModel = {"classifier": [CalibratedClassifierCV()]}
GaussianNBModel = {"classifier": [GaussianNB()]}
LabelPropagationModel = {"classifier": [LabelPropagation()]}
BernoulliNBModel = {"classifier": [BernoulliNB()]}
ExtraTreesClassifierModel = {"classifier": [ExtraTreesClassifier()],
"classifier__random_state": [0, 1, 2, 3, 4], "classifier__n_estimators": [320, 360, 400], "classifier__max_depth": [25, 34, 45]}
SGDClassifierModel = {"classifier": [SGDClassifier()]}
RidgeClassifierModel = {"classifier": [RidgeClassifier()]}
LGBMClassifierModel = {"classifier": [LGBMClassifier()]}
XGBClassifierModel = {"classifier": [XGBClassifier()]}
BaggingClassifierModel = {"classifier": [BaggingClassifier()]}
RandomForestClassifierModel = {"classifier": [RandomForestClassifier()],
                               'classifier__n_estimators': [16, 32, 50]}
DecisionTreeClassifierModel = {"classifier": [DecisionTreeClassifier()]}
KNeighborsClassifierModel = {"classifier": [KNeighborsClassifier()]}
MultinomialNBModel = {"classifier": [MultinomialNB()]}
GaussianNBModel = {"classifier": [GaussianNB()]}
SVCModel = {"classifier": [SVC()],
     'classifier__kernel': ['linear', 'rbf'], 'classifier__C': [1, 10], 'classifier__gamma': [0.001, 0.0001]}
QuadraticDiscriminantAnalysisModel = {"classifier": [QuadraticDiscriminantAnalysis()]}
PerceptronModel = {"classifier": [Perceptron()]}
NearestCentroidModel = {"classifier": [NearestCentroid()]}
LogisticRegressionCVModel = {"classifier": [LogisticRegressionCV()]}
LinearSVCModel = {"classifier": [LinearSVC()]}
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

all_regressors_algs_and_params = [ExtraTreeClassifierModel, GradientBoostingClassifierModel, AdaBoostClassifierModel,
                                  HistGradientBoostingClassifierModel, PassiveAggressiveClassifierModel, RidgeClassifierCVModel,
                                  MLPClassifierModel, LinearDiscriminantAnalysisModel, LabelSpreadingModel,
                                  CalibratedClassifierCVModel, GaussianNBModel, LabelPropagationModel, BernoulliNBModel,
                                  ExtraTreesClassifierModel, SGDClassifierModel, RidgeClassifierModel,
                                  XGBClassifierModel, BaggingClassifierModel, RandomForestClassifierModel,
                                  DecisionTreeClassifierModel, KNeighborsClassifierModel, MultinomialNBModel,
                                  GaussianNBModel, QuadraticDiscriminantAnalysisModel, PerceptronModel,
                                  NearestCentroidModel, LogisticRegressionCVModel, LinearSVCModel]  # TODO: add more regressors

algorithms = {'AdaBoostClassifier' : AdaBoostClassifier,
 'BaggingClassifier' : BaggingClassifier,
 'BernoulliNB' : BernoulliNB,
 'CalibratedClassifierCV' : CalibratedClassifierCV,
 'DecisionTreeClassifier' : DecisionTreeClassifier,
 'ExtraTreeClassifier' : ExtraTreeClassifier,
 'ExtraTreesClassifier' : ExtraTreesClassifier,
 'GaussianNB' : GaussianNB,
 'GradientBoostingClassifier' : GradientBoostingClassifier,
 'HistGradientBoostingClassifier' : HistGradientBoostingClassifier,
 'KNeighborsClassifier' : KNeighborsClassifier,
 'LabelPropagation' : LabelPropagation,
 'LabelSpreading' : LabelSpreading,
 'LinearDiscriminantAnalysis' : LinearDiscriminantAnalysis,
 'LinearSVC' : LinearSVC,
 'LogisticRegression' : LogisticRegression,
 'LogisticRegressionCV' : LogisticRegressionCV,
 'MLPClassifier' : MLPClassifier,
 'NearestCentroid':  NearestCentroid,
 'PassiveAggressiveClassifier' :PassiveAggressiveClassifier,
 'Perceptron' : Perceptron,
 'QuadraticDiscriminantAnalysis' : QuadraticDiscriminantAnalysis,
 'RandomForestClassifier' : RandomForestClassifier,
 'RidgeClassifier' : RidgeClassifier,
 'RidgeClassifierCV' : RidgeClassifierCV,
 'SGDClassifier' : SGDClassifier,
 'SVC' : SVC}

feature_selector = WithoutHardware()
splitter = RegularTrainTestSplit(feature_selector)
x_train, y_train = splitter.create_train_set()
x_test, y_test = splitter.create_test_set()



def print_scores(y, y_pred):
    print("*** Model's Accuracy on test-set ***")
    score = metrics.accuracy_score(y, y_pred)
    print(f"Accuracy value: {score}")


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



def select_top_models_with_grid():
    pipe = Pipeline([('classifier', LogisticRegression())])
    grid = GridSearchCV(pipe, all_regressors_algs_and_params, verbose=3, refit=True, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(x_train, y_train)

    final_results = pd.DataFrame(grid.cv_results_)
    best_five_models = save_grid_search_results(final_results, "accuracy")

    return best_five_models, grid.best_estimator_

def discretisize_target():
    global y_train, y_test
    y_train = pd.cut(y_train, bins=6, labels=range(6))
    y_test = pd.cut(y_test, bins=6, labels=range(6))



def print_grid_search_top_models(best_models, scorer):
    print(f"~~~~ Best {BEST_MODELS} models based on {scorer} score ~~~~")
    for index, (score, model) in enumerate(best_models):
        print("*************** model number", index + 1, "***************")
        print(f"model {scorer} on train:", -1 * score)
        print("model params:", model)
        print()


def main():
    """
                                  XGBClassifierModel, BaggingClassifierModel, RandomForestClassifierModel,
                                  DecisionTreeClassifierModel, KNeighborsClassifierModel, MultinomialNBModel,
                                  GaussianNBModel, SVCModel, QuadraticDiscriminantAnalysisModel, PerceptronModel,
                                  NearestCentroidModel, LogisticRegressionCVModel, LinearSVCModel"""
    discretisize_target()
    """m = LGBMClassifier()
    m.fit(x_train, y_train)
    m.predict(x_test)"""



    print(f"*** Select best {BEST_MODELS} classifiers: ")
    best_five_by_acc, best_acc_estimator = select_top_models_with_grid()
    print("*** Final results for train set from grid search: ***")
    print_grid_search_top_models(best_five_by_acc, 'accuracy')

    print("*** Best estimator based on grid search on test set: ***")
    print(f"Best estimator from grid based MAE :")
    print(best_acc_estimator)
    y_pred_test = best_acc_estimator.predict(x_test)
    print_scores(y_test, y_pred_test)


if __name__ == '__main__':
    main()
