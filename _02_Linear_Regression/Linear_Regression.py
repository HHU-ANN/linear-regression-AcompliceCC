# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y=read_data()
    # z=np.matmul(X.T,X)+np.eye(X.shape[1])*0.2
    # weight=np.matmul(np.linalg.inv(z),np.matmul(X.T,y))
    m = X.shape[0]  # 数据量
    n = X.shape[1]  # 特征量
    w = np.zeros(n)
    for i in range(500):
                # 计算梯度
        grad = (X.T @ (X @ w - y)) + 0.1 * w
        # 更新权重
        w -= 0.01 * grad
  return w @ data
    
def lasso(data):
    return ridge(data)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
