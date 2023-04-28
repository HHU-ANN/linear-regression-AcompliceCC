# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
     X,y=read_data()
    m = X.shape[0]  # 数据量
    n = X.shape[1]  # 特征量
    iternum=400
    alpha=0.01
    lamda=0.001
    costs = np.ones(iternum)
    for i in range(iternum):
        for j in range(n):
            weight[j] = weight[j] + np.sum((y - np.matmul(X, weight)) * X[:, j].reshape(-1, 1)) * (alpha / m) - 2 * lamda * weight[j]
    return weight @ data
    
def lasso(data):
    return ridge(data)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
