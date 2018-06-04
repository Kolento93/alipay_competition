###################################
# 模型评价指标模块
# 对模型性能进行评价
#
# AUTHOR:haoran.gu
###################################
import numpy as np
from sklearn.metrics import roc_auc_score


class Metrics(object):

    # 输出模型的precision,recall,auc

    def __init__(self, y_true, y_pred, prob=False, threshold=0.5):
        '''
        Parameters
        ----------
        y_true : 1d array-like

        y_pred : 1d array-like

        prob : define y_true is 0,1 or probability;default is False 

        threshold : if prob == True : you need to set a threshold to get precision and recall
                    default threshold is 0.5    
        '''
        self.y_true = y_true
        self.y_pred = y_pred
        self.prob = prob
        self.threshold = threshold
        
        assert len(y_true) == len(y_pred), "y_true and y_pred don't have same length"
        assert (1 in y_true) & (0 in y_true) & (len(np.unique(y_true)) == 2), "y_true has value error"
        if prob:
            assert sum((y_pred <= 1.0) & (y_pred >= 0.0)) == len(y_pred), "y_pred has value > 1.0 or < 0.0"
        else:
            assert (1 in y_pred) & (0 in y_pred) & (len(np.unique(y_pred)) == 2), "y_true has value error"
            
        self.cm = self._confusion_mat()

    def _confusion_mat(self):
        '''
            get confusion matrix 
            cm = np.array([[tp,fp],
                           [fn,tn]])       

            tp: y_true == 1 & y_pred == 1
            fn: y_true == 1 & y_pred == 0
            fp: y_true == 0 & y_pred == 1
            tn: y_true == 0 & y_pred == 0

        '''
        y_pred = self.y_pred
        y_true = self.y_true
        cm = np.zeros(shape=(2, 2))
        if self.prob:
            y_pred = (y_pred >= self.threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 1) & (y_true == 0))
        fp = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1] = tp, fp, fn, tn

        return cm

    def model_precision(self):
        # calculate precision
        cm = self.cm
        tp = cm[0, 0]
        fp = cm[1, 0]

        return tp / (tp + fp)

    def model_recall(self):
        cm = self.cm
        tp = cm[0, 0]
        fn = cm[0, 1]

        return tp / (tp + fn)

    def model_auc(self):
        return roc_auc_score(y_true=self.y_true,
                             y_score=self.y_pred)


if __name__ == '__main__':
    y_true = np.array([1,0,1,0])
    y_pred = np.array([0.8,0.6,0.7,0.1])

    model_metrics = Metrics(y_true, y_pred, prob=True)
    print(model_metrics.cm)
    print(model_metrics.model_auc())
    print(model_metrics.model_precision())
    print(model_metrics.model_recall())