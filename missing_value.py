import pandas as pd

def missing_value_rate(sr_feature):
    """
    args:
        sr_feature : pandas.Series, feature
    return:
        float, rate of missing value.
    """
    return sr_feature.isnull().mean()

def drop_feature_by_mvr(df_dataset, threshold = 0.9, inplace = True):
    dict_mvr = {col : missing_value_rate(sr) for col, sr in df_dataset.iteritems()}
    lst_drop = [col for col, mvr in dict_mvr.items() if mvr > threshold]
    return df_dataset.drop(lst_drop, axis=0, inplace = inplace)

