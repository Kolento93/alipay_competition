import os
import pandas as pd

def read_data_test():
    rawdata_path = os.path.join("temp", "atec_anti_fraud_train_labeled.csv")
    cache_path = os.path.join("temp", "cache.pkl")
    if not os.path.exists(cache_path):
        pd.read_csv(rawdata_path).to_pickle(cache_path)
    return pd.read_pickle(cache_path)