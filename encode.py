############################## 
# 变量编码模块
# 离散和连续变量区分
# 离散-->one hot 连续-->缺失值填补
# AUTHOR:haoran.gu
##############################
import pandas as pd 

def divide_features(df, crush_cols, threshold=5):
    continous_cols = []
    category_cols = []

    feature_cols = set(df.columns) - set(crush_cols)
    
    print("This dataset has {} columns \n has {} feature columns".format(len(df.cloumns), len(feature_cols)))
    
    for col in feature_cols:
        feat_unique_cnt = len(df[col].unique())
        if feat_unique_cnt <= threshold:
            category_cols.append(col)
        else:
            continous_cols.append(col)
    print("continous feature count : {} \n category feature count : {}".format(len(continous_cols), len(category_cols)))

    return continous_cols, category_cols

def proprecess_features(df, continous_cols, category_cols, fillna_type = 'zero'):
    # fillna type : 'zero' ...

    df = pd.get_dummies(data=df, columns=category_cols, dummy_na=True)
    print("One-hot category features from {} to {}".format(len(category_cols), \
                                len(df.columns) - len(category_cols) - len(continous_cols)))
                                
    if fillna_type == 'zero':
        df = df.fillna(0)
    
    return df

def get_ready_data(df, crush_cols):
    continous_cols, category_cols = divide_features(df, crush_cols)
    df = proprecess_features(df, continous_cols, category_cols)
    return df

if __name__ == '__main__':
    pass

