import os
import numpy as np
import pandas as pd

def read_data_test():
    rawdata_path = os.path.join("temp", "atec_anti_fraud_train_labeled.csv")
    cache_path = os.path.join("temp", "cache.pkl")
    if not os.path.exists(cache_path):
        pd.read_csv(rawdata_path).to_pickle(cache_path)
    dataset_test = pd.read_pickle(cache_path)
    return dataset_test

def load_dataset_defualt(dataset_id):
    return None

def load_dataset_from_df(X_train, X_test):
    return lambda dataset_id : (X_train, X_test)