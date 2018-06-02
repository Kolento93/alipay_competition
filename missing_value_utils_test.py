import os
import pandas as pd
from missing_value_utils import drop_feature_by_mvr

def read_testdata():
    rawdata_path = os.path.join("temp", "atec_anti_fraud_train_labeled.csv")
    cache_path = os.path.join("temp", "cache.pkl")
    if not os.path.exists(cache_path):
        pd.read_csv(rawdata_path).to_pickle(cache_path)
    return pd.read_pickle(cache_path)
    
def drop_feature_by_mvr_test():
    df_dataset = read_testdata()
    old_len = len(df_dataset.columns)
    ## test drop inplace
    df_dataset = drop_feature_by_mvr(df_dataset, threshold= 0.1)
    new_len = len(df_dataset.columns)
    assert new_len < old_len, "old_len : %d, new_len : %d" % (old_len, new_len)

if __name__ == "__main__":
    drop_feature_by_mvr_test()