import numpy as np
import pandas as pd

class LinDiscAnalysis:
    def __init__(self):
        self.w = None
        self.binary = True

    def fit(self, X, y):
        if y.value_counts().shape[0] == 2:
            self.binary = True
            return self.__binLDA(X, y)
        else:
            self.binary = False
            return self.__multiLDA(X, y)
        
    def __binLDA(self, X, y):
        inv_x_cov = np.linalg.inv(X.cov())
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        p_0 = len(X_0) / len(X)
        p_1 = len(X_1) / len(X)
        mean_0 = X_0.mean()
        mean_1 = X_1.mean()
        w_0 = np.log(p_1 / p_0) - 0.5 * np.dot(np.dot(mean_1, inv_x_cov), mean_1) + 0.5 * np.dot(np.dot(mean_0, inv_x_cov), mean_0)
        w = np.dot((mean_1 - mean_0), inv_x_cov)
        self.w = np.append(w_0, w)

    def __multiLDA(self, X, y):
        inv_x_cov = np.linalg.inv(X.cov())
        p = {}
        mean = {}
        for i in y.unique():
            p[i] = len(X[y == i]) / len(X)
            mean[i] = X[y == i].mean()
        w = {}
        for i in y.unique():
            w[i] = np.append(np.log(p[i]) - 0.5 * np.dot(np.dot(mean[i], inv_x_cov), mean[i]), np.dot(mean[i], inv_x_cov))
        self.w = w

    def predict(self, X, prob = False):
        if self.w is None:
            raise ValueError('Model is not fitted yet.')
        if self.binary:
            return self.__pred_binLDA(X, prob)
        else:
            return self.__pred_multiLDA(X, prob)

    def __pred_binLDA(self, X, prob = False):
        w_0 = self.w[0]
        w = self.w[1:]
        pred_prob = np.dot(X, w) + w_0
        if prob:
            return pred_prob
        else:
            pred = [1 if i > 0 else 0 for i in pred_prob]
            return pred
        
    def __pred_multiLDA(self, X, prob = False):
        pred_prob = {}
        for i in self.w.keys():
            w_0 = self.w[i][0]
            w = self.w[i][1:]
            pred_prob[i] = np.dot(X, w) + w_0
        if prob:
            return pred_prob
        else:
            pred = []
            for i in range(len(X)):
                max_prob = -np.inf
                pred_class = None
                for j in pred_prob.keys():
                    if pred_prob[j][i] > max_prob:
                        max_prob = pred_prob[j][i]
                        pred_class = j
                pred.append(pred_class)
            return pred
        
    def get_params(self):
        return self.w
    
    def set_params(self, w):
        self.w = w