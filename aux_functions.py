import pandas as pd
import numpy as np
import scipy.stats as st
import warnings
import itertools

def fix_dtypes(df):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df

def remove_non_features(df, columns_to_remove):
    """
    Remove columns from df if they exists
    """
    remove_mask = df.columns.isin(columns_to_remove)
    return df.drop(columns = df.columns[remove_mask])

def breakdown_datetime(df):
    df["year"] = df["datetime"].apply(lambda x: x.year).astype(int)
    df["month"] = df["datetime"].apply(lambda x: x.month).astype(int)
    df["day"] = df["datetime"].apply(lambda x: x.day).astype(int)
    df["hour"] = df["datetime"].apply(lambda x: x.hour).astype(int)
    df["weekday"] = df["datetime"].apply(lambda x: x.weekday()).astype(int)
    return df

def one_hot_encode_categoricals(df, categorical_columns, drop = False):
    for col in set(categorical_columns.keys()):
        if col in df.columns:
            df_temp = pd.get_dummies(df[col])
            df_temp.columns = [col + " = " + str(c) for c in df_temp.columns]
            df = df.join(df_temp)
    if drop:
        df = remove_non_features(df, categorical_columns.keys())
    df = adding_must_have_cols(df, categorical_columns)
    return df

def adding_must_have_cols(df, categorical_columns):
    must_have_cols = [
        [f"{col} = {x}" for x in values] for col, values in categorical_columns.items()
    ]
    must_have_cols = list(itertools.chain.from_iterable(must_have_cols))
    for col in must_have_cols:
        if col in df.columns:
            continue
        else:
            df[col] = 0
    return df

def create_cyclic_features(df):

    df["cyc_s_month"] = np.sin(df["month"] * 2 * np.pi / 12)
    df["cyc_s_month_day"] = np.sin(df["day"] * 2 * np.pi / 31)
    df["cyc_s_hour"] = np.sin(df["hour"] * 2 * np.pi / 24)
    df["cyc_s_weekday"] = np.sin(df["weekday"] * 2 * np.pi / 7)

    df["cyc_c_month"] = np.cos(df["month"] * 2 * np.pi / 12)
    df["cyc_c_month_day"] = np.cos(df["day"] * 2 * np.pi / 31)
    df["cyc_c_hour"] = np.cos(df["hour"] * 2 * np.pi / 24)
    df["cyc_c_weekday"] = np.cos(df["weekday"] * 2 * np.pi / 7)

    return df