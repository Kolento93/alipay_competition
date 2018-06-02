import pandas as pd
import os
import sys
import conf
sys.path.append('/Users/haorangu/Documents/data_competition/alipay_competition')
from missing_value_utils import drop_feature_by_mvr
from encode import get_ready_data

if __name__ == '__main__':
    path = conf.file_path
    df = pd.read_csv(path)
    df = drop_feature_by_mvr(df)
    crush_cols = ['id', 'label', 'date']
    df = get_ready_data(df, crush_cols)