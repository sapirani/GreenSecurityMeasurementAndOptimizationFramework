from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline




def print_scores(y, y_pred):
    print("*** Model's Accuracy ***")

    APE = 100 * (abs(y - y_pred) / y)
    print(f"APE value: {APE}")

    MSE = mean_squared_error(y_pred, y)
    print(f"MSE value: {MSE}")


def main():
    # after receiving the best model, train it and predict about the test.
    x_train, y_train = []
    x_test, y_test = []

    model = MLPRegressor()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    print_scores(y_test, pred)


main()
