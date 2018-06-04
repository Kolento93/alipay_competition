import os
import numpy as np
import pandas as pd

def read_data_test():
    rawdata_path = os.path.join("temp", "atec_anti_fraud_train_labeled.csv")
    cache_path = os.path.join("temp", "cache.pkl")
    if not os.path.exists(cache_path):
        pd.read_csv(rawdata_path).to_pickle(cache_path)
    dataset_test = pd.read_pickle(cache_path)
    dataset_test.loc[dataset_test.index[-100:], 'label'] = np.nan
    return dataset_test

def load_dataset_defualt(dataset_id):
    return None

def load_dataset_from_df(dataset, feature_names, target_name):
    return lambda dataset_id : (dataset[feature_names].values, dataset[target_name].values)