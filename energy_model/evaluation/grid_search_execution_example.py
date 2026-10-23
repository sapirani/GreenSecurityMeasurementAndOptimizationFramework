import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, \
    RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import ExtraTreeRegressor

from energy_model.configs.columns import ProcessColumns
from energy_model.evaluation.model_evaluator import ModelEvaluator
from energy_model.pipelines.pipeline_utils import extract_x_y

PROCESS_SYSTEM_DF_PATH = r"C:\Users\Administrator\Desktop\GreenSecurityAll\documents\gt\with batch 10 minutes\basic - system based - process of interest\all_durations\system_process_df_new.csv"

HistGradientBoostingRegressorModel = {"classifier": [HistGradientBoostingRegressor()],
                                      "classifier__quantile": [0.5],
                                      "classifier__max_iter": [400],
                                      "classifier__l2_regularization": [0.3],
                                      'classifier__max_depth': [8]}
GradientBoostingRegressorModel = {"classifier": [GradientBoostingRegressor()],
                                  'classifier__max_depth': [80],
                                  'classifier__max_features': [3],
                                  'classifier__min_samples_leaf': [5],
                                  'classifier__min_samples_split': [8],
                                  'classifier__n_estimators': [500]}

RandomForestRegressorModel = {"classifier": [RandomForestRegressor()],
                              'classifier__n_estimators': [1000],
                              'classifier__max_features': ['sqrt'],
                              'classifier__max_depth': [60]}

if __name__ == '__main__':
    df = pd.read_csv(PROCESS_SYSTEM_DF_PATH)

    x, y = extract_x_y(df, target_column=ProcessColumns.ENERGY_USAGE_PROCESS_COL)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    pipe = Pipeline([("classifier", LinearRegression())])
    kf_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, RandomForestRegressorModel, verbose=3, refit=True, cv=kf_cv, scoring="neg_root_mean_squared_error",
                        n_jobs=-1)

    grid.fit(x_train, y_train)
    m = grid.best_estimator_

    y_pred = m.predict(x_test)
    ev = ModelEvaluator()
    res = ev.evaluate(y_test, y_pred)
    ev.print_results(res)
