from datetime import datetime

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

from SapirModel.DatasetFeatureSelection import ProcessAndFullSystem

BEST_MODELS = 5

algorithms = {'AdaBoostRegressor': AdaBoostRegressor,
              'ARDRegression': ARDRegression,
              'BaggingRegressor': BaggingRegressor,
              'BayesianRidge': BayesianRidge,
              'DecisionTreeRegressor': DecisionTreeRegressor,
              'DummyRegressor': DummyRegressor,
              'ElasticNet': ElasticNet,
              'ElasticNetCV': ElasticNetCV,
              'ExtraTreeRegressor': ExtraTreeRegressor,
              'ExtraTreesRegressor': ExtraTreesRegressor,
              'GradientBoostingRegressor': GradientBoostingRegressor,
              'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
              'HuberRegressor': HuberRegressor,
              'KNeighborsRegressor': KNeighborsRegressor,
              'Lars': Lars,
              'LarsCV': LarsCV,
              'Lasso': Lasso,
              'LassoCV': LassoCV,
              'LassoLars': LassoLars,
              'LassoLarsCV': LassoLarsCV,
              'LassoLarsIC': LassoLarsIC,
              'LinearSVR': LinearSVR,
              'LinearRegression': LinearRegression,
              'MLPRegressor': MLPRegressor,
              'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit,
              'OrthogonalMatchingPursuitCV': OrthogonalMatchingPursuitCV,
              'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
              'PLSCanonical': PLSCanonical,
              'PLSRegression': PLSRegression,
              'RandomForestRegressor': RandomForestRegressor,
              'RANSACRegressor': RANSACRegressor,
              'Ridge': Ridge,
              'RidgeCV': RidgeCV,
              'SGDRegressor': SGDRegressor,
              'TheilSenRegressor': TheilSenRegressor,
              'TransformedTargetRegressor': TransformedTargetRegressor,
              'TweedieRegressor': TweedieRegressor}

# TODO: add more models and hyperparameters
params_LogisticRegression = {"classifier": [LogisticRegression()],
                             "classifier__class_weight": ["balanced", None],
                             "classifier__solver": ["liblinear", "lbfgs"],
                             "classifier__C": [0.01, 0.1, 0.2, 0.3, 0.5],
                             "classifier__tol": [0.0001, 0.001, 0.000001, 0.1]}

params = [params_LogisticRegression] # TODO: add more params

feature_selector = ProcessAndFullSystem()
train_set = feature_selector.create_train_set()
test_set = feature_selector.create_test_set()
x_train = train_set.iloc[:,:-1]
y_train = train_set.iloc[:,-1:]
x_test = test_set.iloc[:,:-1]
y_test = test_set.iloc[:,-1:]


def compare_models(x_train, y_train, x_validation, y_validation, use_pca=True):
    models_and_scores = []

    for alg_name, alg in algorithms.items():
        pipe = make_pipeline(PCA(), alg()) if use_pca else alg()
        pipe.fit(x_train, y_train)
        y_predictions = pipe.predict(x_validation)
        accuracy_score = mean_absolute_error(y_validation, y_predictions)
        print("The mean absolute error ({0} PCA) of {1} is: {2}".format("with" if use_pca else "without", alg_name,
                                                                        accuracy_score))

        models_and_scores.append((alg_name, accuracy_score))

    return models_and_scores

def print_and_save_grid_search_results(final_results):
    res_list = []
    for index, row in final_results.iterrows():
        """best_allowed_index = np.argmax(np.array([row[f"mean_test_{i}_allowed"] for i in ALLOWED_MALICIOUS_PREDICTIONS]))
        best_allowed_for_model = ALLOWED_MALICIOUS_PREDICTIONS[best_allowed_index]
        model_score = row[f"mean_test"]"""
        res_list.append((row[f"mean_test"], row["params"]))

    print()
    print("------ Best 5 ------")
    print()

    best_five = list(sorted(res_list, key=lambda x: x[0], reverse=True)[:5])


    for index, (score, model) in enumerate(best_five):
        print("*************** model number", index + 1, "***************")
        print("model score:", score)
        print("model params:", model)
        print()

    final_results.to_csv(f"final_results_{datetime.now().strftime('%d_%m_%Y %H_%M')}.csv", index=False)


def print_scores(y, y_pred):
    print("*** Model's Accuracy ***")

    APE = 100 * (abs(y - y_pred) / y)
    print(f"APE value: {APE}")

    MSE = mean_squared_error(y_pred, y)
    print(f"MSE value: {MSE}")


def tune_hyperparams_grid_search():
    pipe = Pipeline([('classifier', MultinomialNB())])
    grid = GridSearchCV(pipe, params, verbose=3, refit=True, cv=5, error_score=mean_squared_error, n_jobs=-1)
    grid.fit(x_train, y_train)

    final_results = pd.DataFrame(grid.cv_results_)
    print_and_save_grid_search_results(final_results)

    grid.best_estimator_.score(x_test, y_test)

    print("Best Parameters: ")
    print(grid.best_params_)
    grid_predictions = grid.predict(x_test)

    # print regression report
    print(print_scores(y_test, grid_predictions))


'''def objective(trial, x_train, y_train, x_validation, y_validation):
    regressor_name = trial.suggest_categorical("regressor",
                                               ["ExtraTreesRegressor", "RandomForestRegressor", "BaggingRegressor", ])
    print(regressor_name)

    if regressor_name == "ExtraTreesRegressor":
        min_samples_leaf = trial.suggest_int('ExtraTreesRegressor_min_samples_leaf', 1, 10)
        min_samples_split = trial.suggest_int('ExtraTreesRegressor_min_samples_split', 2, 10)
        n_estimators = trial.suggest_int('ExtraTreesRegressor_n_estimators', 100, 500)

        regressor = ExtraTreesRegressor(min_samples_leaf=min_samples_leaf,
                                        min_samples_split=min_samples_split,
                                        n_estimators=n_estimators)

    elif regressor_name == "RandomForestRegressor":
        min_samples_leaf = trial.suggest_int('RandomForestRegressor_min_samples_leaf', 1, 10)
        min_samples_split = trial.suggest_int('RandomForestRegressor_min_samples_split', 2, 10)
        n_estimators = trial.suggest_int('RandomForestRegressor_n_estimators', 100, 500)

        regressor = RandomForestRegressor(min_samples_leaf=min_samples_leaf,
                                          min_samples_split=min_samples_split,
                                          n_estimators=n_estimators)

    elif regressor_name == "BaggingRegressor":
        bootstrap_features = trial.suggest_categorical('BaggingRegressor_bootstrap_features', [True, False])
        n_estimators = trial.suggest_int('BaggingRegressor_n_estimators', 10, 100)

        regressor = BaggingRegressor(bootstrap_features=bootstrap_features,
                                     n_estimators=n_estimators)

    regressor.fit(x_train, y_train)
    y_predictions = regressor.predict(x_validation)
    return mean_absolute_error(y_validation, y_predictions)'''


def get_top_models(models_and_scores, ending_string):
    print(f"Best {BEST_MODELS} models {ending_string}: ")
    return sorted(models_and_scores, key=lambda list_item: list_item[1])[:BEST_MODELS]


def main():
    print("all models using pca")
    print()
    models_and_scores_with_pca = compare_models(use_pca=True)

    print("*** List of models and scores with PCA (all models) ***")
    print(models_and_scores_with_pca)

    print("all models use all features")
    print()
    models_and_scores_without_pca = compare_models(use_pca=False)

    print("*** List of models and scores without PCA (all models) ***")
    print(models_and_scores_without_pca)

    top_five_with_pca = get_top_models(models_and_scores_with_pca, "with pca")
    #print(f"Best {BEST_MODELS} models with pca: ")
    #top_five_with_pca = sorted(models_and_scores_with_pca, key=lambda list_item: list_item[1])[:BEST_MODELS]
    print(f"*** List of models and scores with PCA ({BEST_MODELS} models) ***")
    print(top_five_with_pca)

    #print(f"Best {BEST_MODELS} models without pca: ")
    #top_five_without_pca = sorted(models_and_scores_without_pca, key=lambda list_item: list_item[1])[:BEST_MODELS]
    top_five_without_pca = get_top_models(models_and_scores_without_pca, "without pca")

    print(f"*** List of models and scores without PCA ({BEST_MODELS} models) ***")
    print(top_five_without_pca)

    print("Get best of models with/without pca: ")
    best_models_considering_pca = {}
    for i in top_five_with_pca:
        best_models_considering_pca[i[0]] = i + ("With PCA",)

    for i in top_five_without_pca:
        if i[0] in best_models_considering_pca:
            best_models_considering_pca[i[0]] = best_models_considering_pca[i[0]] if best_models_considering_pca[i[0]][
                                                                                         1] < i[1] else i + (
                "Without PCA",)
        else:
            best_models_considering_pca[i[0]] = i + ("Without PCA",)

    top_five_models = get_top_models(best_models_considering_pca.values(), "")
    #top_five_models = sorted(best_models_considering_pca.values(), key=lambda list_item: list_item[1])[
    #                  :BEST_MODELS]
    print(f"*** List of top models and scores ({BEST_MODELS} models) ***")
    print(top_five_models)

    #tune_hyperparams_grid_search()

    print("~~~~ Tune Hyper Parameters: ~~~~ ")
    '''study = optuna.create_study(study_name="best regressor", direction="minimize")
    study.optimize(objective, n_trials=50)
    print('best_trial\n', study.best_trial)
    print('best_params\n', study.best_params)
    print('best_value\n', study.best_value)'''


if __name__ == '__main__':
    main()
