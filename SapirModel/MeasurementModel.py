import math
from datetime import datetime
from math import sqrt

import numpy as np
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
    ProcessColumns


def print_scores(y, y_pred):
    print("*** Model's Accuracy ***")
    PER = abs(y - y_pred) / y
    print(f"PER value: {PER}")
    percent = np.mean(PER * 100)
    print("Percent: ", percent)

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
        print(score)

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



def main():

    feature_selector = ProcessAndFullSystem()
    splitter = RegularTrainTestSplit(feature_selector)
    x_train, y_train = splitter.create_train_set()
    x_test, y_test = splitter.create_test_set()
    params = {"classifier": [RandomForestRegressor()],
                              'classifier__n_estimators': [100, 500, 1000],
                              'classifier__max_features': ['sqrt'],
                              'classifier__max_depth': [7, 15, 10],
                               'classifier__min_samples_split': [5, 8, 12]}



    pipe = Pipeline([('classifier', RandomForestRegressor())])
    grid_mae = GridSearchCV(pipe, params, verbose=3, refit=True, cv=x_train.shape[0], scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_mae.fit(x_train, y_train.values.ravel())

    pipe = Pipeline([('classifier', RandomForestRegressor())])
    grid_rmse = GridSearchCV(pipe, params, verbose=3, refit=True, cv=x_train.shape[0], scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_rmse.fit(x_train, y_train.values.ravel())

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
    y_pred_rmse = estimator_rmse.predict(x_test)
    print_scores(y_test, y_pred_rmse)
    #model = RandomForestRegressor(max_depth=70, max_features='auto', n_estimators=500)
    #model.fit(x_train, y_train.values.ravel())
    """pred_mae = estimator_mae.predict(x_test)
    print_scores(y_test.values.ravel(), pred_mae)
    print(estimator_rmse)
    pred_rmse = estimator_rmse.predict(x_test)
    print_scores(y_test.values.ravel(), pred_rmse)"""


main()
