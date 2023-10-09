import numpy as np
import pandas as pd

class QuadDiscAnalysis:
    def __init__(self):
        self.cov_mat = None
        self.mean = None
        self.probabilities = None
        self.classes = None

    def fit(self, X, y):
        self.probabilities = {}
        self.cov_mat = {}
        self.mean = {}
        self.classes = y.unique()
        cov_det = []
        for c in self.classes:
            self.probabilities[c] = y.value_counts()[c] / len(y)
            self.cov_mat[c] = X[y == c].cov()
            cov_det.append(np.linalg.det(X[y == c].cov()))
            self.mean[c] = X[y == c].mean()
        if np.any(np.array(cov_det) == 0):
            raise ValueError('Covariance matrix is singular.')

    def predict(self, X, prob = False):
        if self.cov_mat is None or self.mean is None or self.probabilities is None or self.classes is None:
            raise ValueError('Model is not fitted yet.')
        pred_prob = pd.DataFrame(columns=self.classes)
        for i in range(len(X)):
            for c in self.classes:
                pred_prob.loc[i, c] = np.log(self.probabilities[c]) - 0.5 * np.log(np.linalg.det(self.cov_mat[c])) - 0.5 * (X.iloc[i] - self.mean[c]).dot(np.linalg.inv(self.cov_mat[c])).dot((X.iloc[i] - self.mean[c]).T)
        if prob:
            return pred_prob
        else:
            pred = [pred_prob.iloc[i].idxmax() for i in range(len(pred_prob))]
            return pred

    def get_params(self):
        return {'classes': self.classes, 'cov_mat': self.cov_mat, 'mean': self.mean, 'probabilities': self.probabilities}
    
    def set_params(self, params):
        self.classes = params['classes']
        self.cov_mat = params['cov_mat']
        self.mean = params['mean']
        self.probabilities = params['probabilities']