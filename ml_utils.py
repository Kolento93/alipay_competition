'''
model: tuple, (<model_id>, <dataset_id>, <parameters>)
'''

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from data_utils import load_dataset_defualt

import logging
logger = logging.getLogger(__name__)

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

def stringify(model, dataset_id):
    return "%s::%s" % (model.model_id, dataset_id)

def find_params_by_gridsearch(model, dataset_id, target, param_grid, scoring, cv = 5, random_state = 0):
    dataset_train, _ = model.load_dataset(dataset_id)
    clf = GridSearchCV(model, param_grid, cv=5, n_jobs=6, scoring=scoring)
    X = dataset_train.loc[target.index].values
    y = target.values
    clf.fit(X, y)
    logger.info("found params (%s > %.4f): %s",	stringify(model, dataset_id), clf.best_score_, clf.best_params_)
    return clf.best_params_

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
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        if X is not None:
            return self.model.predict_proba(X)[:,0]
        
    def fit_and_predict(self, dataset_id, target, idx_valid = None):
        '''
            args:
                dataset_id: str, id of feature set;
                target: pandas.Series, target;
                idx_valid: list or np.array, index of valid set.
        '''
        X_train, X_test = self.load_dataset(dataset_id)
        if idx_valid is not None:
            idx_train = np.setdiff1d(range(len(X_train)), idx_valid)
            X_valid, y_valid = X_train.iloc[idx_valid], target.values[idx_valid]
            X_train, y_train = X_train.iloc[idx_train], target.values[idx_train]
        else:
            X_valid = None
        # train
        self.fit(X_train, y_train) 
        # predict
        y_pred_train = self.predict(X_train) # trainset
        y_pred_valid = self.predict(X_valid) # validset
        y_pred_test = self.predict(X_test) # testset
        return y_pred_train, y_pred_valid, y_pred_test

        
        