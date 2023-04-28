# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y=read_data()
    m,n=X.shape
    w=np.zeros(n)
    max_iterations=100
    for i in range(max_iterations):
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
