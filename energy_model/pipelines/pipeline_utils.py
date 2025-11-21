import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    y_train = pd.Series(y_train.squeeze()).reset_index(drop=True)
    y_test = pd.Series(y_test.squeeze()).reset_index(drop=True)

    return X_train, X_test, y_train, y_test