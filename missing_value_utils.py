import pandas as pd

def desc_missing_value(df_dataset):
    mvr_by_col = df_dataset.isnull().mean(axis = 1)
    mvr_by_row = df_dataset.isnull().mean(axis = 0)
    return mvr_by_row, mvr_by_col

def missing_value_rate(sr_feature):
    """rate of missing value
    args:
        sr_feature : pandas.Series, feature
    return:
        float, rate of missing value
    """
    return sr_feature.isnull().mean()

def drop_feature_by_mvr(df_dataset, threshold = 0.9, inplace = True):
    dict_mvr = {col : missing_value_rate(sr) for col, sr in df_dataset.iteritems()}
    lst_drop = [col for col, mvr in dict_mvr.items() if mvr > threshold]
    return df_dataset.drop(lst_drop, axis=1, inplace = inplace) or df_dataset

def drop_samples_by_mvr(df_dataset, threshold = 0.9, inplace = True):
    mvr_by_row = df_dataset.isnull().mean(axis = 0)
    idx_drop = mvr_by_row.index[mvr_by_row > threshold]
    return df_dataset.drop(idx_drop, axis = 0, inplace = inplace) or df_dataset

def fillna_sr(sr_feature, method = "mean", inplace = True):
    if method == "mean":
        val = sr_feature.mean()
    elif method == "median":
        val = sr_feature.median()
    elif method == "mode":
        val = sr_feature.mode()
    else:
        val = 0
    return sr_feature.fillna(val, inplace = inplace) or sr_feature

def fillna(df_dataset, cols = None, method = "mean", inplace = True):
    if cols is None:
        cols = df_dataset.columns
    if not inplace:
        df_dataset = df_dataset.copy()
    for col in cols:
        fillna_sr(df_dataset[col], method = method, inplace = True)
    return df_dataset