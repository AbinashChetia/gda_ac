import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def splitTrainTest(x, y, train_ratio=0.8):
    '''
    Split data into training and testing sets.
    '''
    df_x = x.copy()
    df_y = y.copy()
    df_y = df_y.rename('y')
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    train_size = int(len(df) * train_ratio)
    train_x = df.iloc[:train_size, :-1].reset_index(drop=True)
    train_y = df.iloc[:train_size, -1].reset_index(drop=True)
    test_x = df.iloc[train_size:, :-1].reset_index(drop=True)
    test_y = df.iloc[train_size:, -1].reset_index(drop=True)
    return train_x, train_y, test_x, test_y

def get_performance_measure(y, pred):
    if np.unique(y).shape[0] == 2:
        tp, tn, fp, fn = 0, 0, 0, 0
        classes = np.unique(y)
        p_class = classes.max()
        n_class = classes.min()
        for i in range(len(y)):
            if y[i] == p_class and pred[i] == p_class:
                tp += 1
            elif y[i] == n_class and pred[i] == n_class:
                tn += 1
            elif y[i] == n_class and pred[i] == p_class:
                fp += 1
            elif y[i] == p_class and pred[i] == n_class:
                fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1 = 2 * precision * recall / (precision + recall)
        return {'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'spec': spec,
                'f1': f1}
    elif np.unique(y).shape[0] > 2:
        acc = 0
        for i in range(len(y)):
            if y[i] == pred[i]:
                acc += 1
        acc /= len(y)
        return {'acc': acc}

def split_kfold(x, y, k=5):
    '''
    Split data into training and testing sets for k-fold cross validation.
    '''
    df_x = x.copy()
    df_y = y.copy()
    df_y = df_y.rename('y')
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    fold_size = int(len(df) / k)
    data_folds = []
    for i in range(k):
        if i != k - 1:
            data_folds.append(df.iloc[i * fold_size: (i + 1) * fold_size, :].reset_index(drop=True))
        else:
            data_folds.append(df.iloc[i * fold_size:, :].reset_index(drop=True))
    return data_folds
    

def normMinMax(df, mode='train', train_min=None, train_max=None):
    '''
    Perform min-max normalization on data.
    '''
    data = df.copy()
    if mode == 'train':
        train_max = {}
        train_min = {}
        for col in data.columns:
            train_max[col] = data[col].max()
            train_min[col] = data[col].min()
            data[col] = (data[col] - train_min[col]) / (train_max[col] - train_min[col])
        return data, train_min, train_max
    
    elif mode == 'test':
        if train_min is None or train_max is None:
            raise Exception('Pass train_min and/or train_max.')
        for col in data.columns:
            data[col] = (data[col] - train_min[col]) / (train_max[col] - train_min[col])
        return data
    
def disp_conf_mat(perf_m):
    gd1_cfm = [[perf_m['tn'], perf_m['fn']], [perf_m['fp'], perf_m['tp']]]
    _, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(gd1_cfm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(len(gd1_cfm)):
        for j in range(len(gd1_cfm[i])):
            ax.text(x=j, y=i,s=gd1_cfm[i][j], va='center', ha='center', size='xx-large')
    plt.xlabel('Actuals', fontsize=10)
    plt.ylabel('Predictions', fontsize=10)
    plt.title('Confusion Matrix', fontsize=10)
    plt.show()

def plot_dec_bound(X, y, w, gda_mode=None):
    c_map = {0: 'b', 1: 'r', 2: 'g', 3: 'm', 4: 'c', 5: 'y', 6: 'k'}
    if gda_mode == None:
        raise Exception('Pass gda_mode.')
    elif gda_mode == 'lda':
        if X.shape[1] == 2 and np.unique(y).shape[0] == 2:
            db = [(-w[0] - w[1] * i) / w[2] for i in X[0]]
            plt.figure(figsize=(5, 5))
            plt.scatter(X[y == 0][0], X[y == 0][1], c='b', marker='x', label='Negative class')
            plt.scatter(X[y == 1][0], X[y == 1][1], c='r', marker='x', label='Positive class')
            plt.plot(X[0], db, c='black', label='Decision boundary')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.legend()
            plt.title('Decision Boundary')
            plt.show()
        # elif X.shape[1] == 3 and np.unique(y).shape[0] == 2:
        #     db = [(-w[0] - w[1] * i - w[2] * j) / w[3] for i, j in zip(X[0], X[1])]
        #     ax = plt.axes(projection='3d')
        #     ax.plot_trisurf(X.iloc[:, 0], X.iloc[:, 1], db, alpha=0.7)
        #     ax.scatter3D(X[y == 0].iloc[:, 0], X[y == 0].iloc[:, 1], X[y == 0].iloc[:, 2], c='b', marker='x', label='Negative class')
        #     ax.scatter3D(X[y == 1].iloc[:, 0], X[y == 1].iloc[:, 1], X[y == 1].iloc[:, 2], c='r', marker='x', label='Positive class')
        #     ax.set_xlabel('X1')
        #     ax.set_ylabel('X2')
        #     ax.set_zlabel('X3')
        #     plt.title('Decision Boundary')
        #     plt.show()
        elif X.shape[1] == 3 and np.unique(y).shape[0] > 2:
            plt.figure(figsize=(16, 9))
            ax = plt.axes(projection='3d')
            db_done = []
            for i in np.unique(y):
                ax.scatter3D(X[y == i][0], X[y == i][1], X[y == i].iloc[:, 2], c=y[y == i].apply(lambda x: c_map[x]), marker='x', label=f'Class {i}')
                for j in np.unique(y):
                    if j != i and (i, j) not in db_done and (j, i) not in db_done:
                        w_db = w[i] - w[j]
                        db = []
                        for m in range(X.shape[0]):
                            db.append((-w_db[0] - w_db[1] * X.iloc[m, 0] - w_db[2] * X.iloc[m, 1]) / w_db[3])
                        ax.plot_trisurf(X[0], X[1], db, alpha=0.5, label=f'Decision boundary between classes {i} & {j}')
                        db_done.append((i, j))
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('X3')
            plt.legend()
            plt.title('Decision Boundaries')
            plt.show()
        else:
            print('Facility to plot decision boundary for data with more than 3 features has not been added yet!')
    elif gda_mode == 'qda':
        pass

def plot_roc(y, pred_prob, thresh):
    tpr = []
    fpr = []
    for t in thresh:
        pred = [1 if i >= t else 0 for i in pred_prob]
        cf_info = get_performance_measure(y, pred)
        tp = cf_info['tp']
        fp = cf_info['fp']
        tn = cf_info['tn']
        fn = cf_info['fn']
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()