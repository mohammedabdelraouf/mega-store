import pandas as pd
import ast


def pre_processing(x):
    x = date_split(x, {'Order Date', 'Ship Date'})
    dict_df = pd.DataFrame([ast.literal_eval(i) for i in x['CategoryTree'].values])
    x = x.join(dict_df)
    x.drop(columns={'CategoryTree'}, axis=1, inplace=True)
    return x


def numerical_Categorical(X=pd.DataFrame()):
    x_numeric = X.select_dtypes(include=['float64', 'int64'])
    X_categorical = X.select_dtypes(include=['object'])
    return x_numeric, X_categorical


def date_split(data, cols):
    result = pd.DataFrame(data)
    for col in cols:
        Z = pd.to_datetime(data[col].values)
        result[col + ' year'] = Z.year
        result[col + ' month'] = Z.month
        result[col + ' day'] = Z.day
        result.drop(columns=col, axis=1, inplace=True)
    return result

