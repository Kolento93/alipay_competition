'''
model: tuple, (<model_id>, <dataset_id>, <parameters>)
'''
from data_utils import load_dataset_defualt
import xgboost as xgb
import numpy as np

MODEL_FACTORY = {
    "xgb-c" : xgb.XGBClassifier
}

def get_model_defualt(model_id):
    '''parser model_id and new model
    args:
        model_id: string, <model_type> + <model_name> eg: xgb-c_01
    '''
    model_type, model_name = model_id.split("_")
    try:
        return MODEL_FACTORY[model_type]()
    except:
        pass
    
class Model(object):
    def __init__(self, model_id, params = None, get_model = get_model_defualt, \
        load_dataset = load_dataset_defualt, random_state = 0):
        '''init model object
            args:
                model_id : str, <model_type> + <model_name>, eg : xgb-c_01;
                params : dict, parameters of model;
                get_model: callabel, with one argument (model_id) and return a model;
                load_dataset: callabel, with one argument (dataset_id) and return a dataset (tuple : (X,y)).
        '''
        self.model_id = model_id
        self.params = params or {}
        self.get_model = get_model
        self.load_dataset = load_dataset
        self.model = self.get_model(self.model_id)
        self.set_params(self.params)

    def set_params(self, params):
        self.model.set_params(**params)
    
    def fit_by_data(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X_test):
        if X_test.size == 0:
            return np.array([])
        else:
            return self.model.predict_proba(X_test)[:,0]
        
    def fit_and_predict(self, dataset_id, idx_valid = None, is_unlabeled = np.isnan):
        X, y = self.load_dataset(dataset_id) # array
        idx_test = np.where(np.isnan(y))[0]
        if idx_valid is None:
            idx_valid = np.array([])
        idx_train = np.arange(len(y), dtype = np.int)
        idx_train = np.setdiff1d(idx_train, idx_test)
        idx_train = np.setdiff1d(idx_train, idx_valid)
        # train
        self.fit_by_data(X[idx_train, :], y[idx_train]) 
        # predict
        y_pred_train = self.predict(X[idx_train, :]) # trainset
        y_pred_valid = self.predict(X[idx_valid, :]) # validset
        y_pred_test = self.predict(X[idx_test, :]) # testset
        return y_pred_train, y_pred_valid, y_pred_test

        
        