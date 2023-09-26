from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from SapirModel.CreateDataset import *

# constants
TRAIN_SET_PATH = r""
TEST_SET_PATH = r""

# model related

classifier = {MLPRegressor(random_state=1, max_iter=500)}

algorithms = {'MLPRegressor': MLPRegressor()}

params_MLP_model = {"classifier": [MLPRegressor()],
                    "classifier__hidden_layer_sizes": [(1,), (50,)],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "solver": ["lbfgs", "sgd", "adam"],
                    "alpha": [0.00005, 0.0005]}


params = [params_MLP_model]

def read_dataset(path):
    df = pd.read_csv(path)
    # TODO: should change df? define cols?
    return df


def create_train_set():
    print("======== Creating Train Dataset ========")
    train_df = initialize_dataset(True)
    train_df = read_directories(train_df, TRAIN_MEASUREMENTS_DIR_PATH, is_train=True)
    print(train_df)

    #full_train_set = read_dataset(TRAIN_SET_PATH)

    # TODO: should remove features??
    return train_df


def create_test_set():
    print("======== Creating Test Dataset ========")
    test_df = initialize_dataset(False)
    test_df = read_directories(test_df, TEST_MEASUREMENTS_DIR_PATH, is_train=False)
    print(test_df)

    #full_test_set = read_dataset(TEST_SET_PATH)

    # TODO: should remove features??
    return test_df


def print_and_save_grid_search_results(final_results):
    res_list = []
    for index, row in final_results.iterrows():
        """best_allowed_index = np.argmax(np.array([row[f"mean_test_{i}_allowed"] for i in ALLOWED_MALICIOUS_PREDICTIONS]))
        best_allowed_for_model = ALLOWED_MALICIOUS_PREDICTIONS[best_allowed_index]
        model_score = row[f"mean_test"]"""
        res_list.append((row[f"mean_test"], row["params"]))


    for index, (score, model) in enumerate(sorted(res_list, key=lambda x: x[0], reverse=True)[:10]):
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


def main():
    train_set = create_train_set()
    test_set = create_test_set()

    X_train = train_set.iloc[:, :-1]
    y_train = train_set.iloc[:, -1:]

    X_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1:]

    pipe = Pipeline([('classifier', MLPRegressor())])
    grid = GridSearchCV(pipe, params, verbose=3, refit=True, cv=5, error_score=0, n_jobs=-1)
    grid.fit(X_train, y_train)

    final_results = pd.DataFrame(grid.cv_results_)

    print_and_save_grid_search_results(final_results)

    grid.best_estimator_.score(X_test, y_test)

    print("Best Parameters: ")
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)

    # print regression report
    print(print_scores(y_test, grid_predictions))


    #basic_df = create_basic_df()

    # prepared df_x contains following features: user index, segment index, extracted features by vectorizers
    #prepared_df = extract_features(basic_df)

    #prepared_labeled_df, prepared_unlabeled_df = separate_not_labeled_rows(prepared_df)

    #sampler = RandomOverSampler(sampling_strategy='minority')
    #sampler = RandomOverSampler(sampling_strategy=1/3)
    #x_resample, y_resample = sampler.fit_resample(prepared_labeled_df.drop([SEGMENT_LABEL_COL], axis=1), prepared_labeled_df[SEGMENT_LABEL_COL])

    #splitter = cv_splitter(x_resample)
    #splitter = cv_splitter(prepared_labeled_df)

    #first_train_indices = next(splitter)[0]
    #print(len(first_train_indices))
    #x_resample.iloc[first_train_indices, :].to_csv("amen.csv")
    # splitter = cv_splitter(prepared_labeled_df)

    #X = prepared_labeled_df.drop([SEGMENT_LABEL_COL], axis=1)
    #X = prepared_labeled_df.drop([SEGMENT_LABEL_COL, SEGMENT_NUMBER_COL], axis=1)
    #y = np.array(prepared_labeled_df[SEGMENT_LABEL_COL])

    #pd.concat([x_resample, pd.Series(y_resample)], axis=1).to_csv("Resampled x.csv")
    #x_resample.to_csv("Resampled x.csv")
    """model = Model()
    model.fit(X, y)"""
    #print(model.predict_proba(X.iloc[150:300, :]))

    # just placeholder


main()
