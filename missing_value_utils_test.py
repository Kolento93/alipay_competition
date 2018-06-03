import os
import pandas as pd
from missing_value_utils import drop_feature_by_mvr
from data_utils import read_data_test

def drop_feature_by_mvr_test():
    df_dataset = read_data_test()
    old_len = len(df_dataset.columns)
    ## test drop inplace
    df_dataset = drop_feature_by_mvr(df_dataset, threshold= 0.1)
    new_len = len(df_dataset.columns)
    assert new_len < old_len, "old_len : %d, new_len : %d" % (old_len, new_len)

if __name__ == "__main__":
    drop_feature_by_mvr_test()