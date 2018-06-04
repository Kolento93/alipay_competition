from data_utils import read_data_test, load_dataset_from_df
import ml_utils

def model_test():
    df_dataset = read_data_test()
    X, y = df_dataset.iloc[:,3:], df_dataset["label"]
    data_loader = load_dataset_from_df(X_train = X, X_test = X) # X_train, X_test
    model = ml_utils.Model("xgb-c_01", load_dataset = data_loader)
    y_pred_train, y_pred_valid, y_pred_test = model.fit_and_predict("dataset-test", target = y, idx_valid = range(20))

if __name__ == "__main__":
    model_test()
