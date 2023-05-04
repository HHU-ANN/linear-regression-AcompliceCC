# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y=read_data()
    #X=(X-np.min(X))/(np.max(X)-np.min(X))
    z=np.matmul(X.T,X)+np.eye(X.shape[1])*(-0.1)
    weight=np.matmul(np.linalg.inv(z),np.matmul(X.T,y))
   # m, n = X.shape
   # weight = np.zeros(n)
  #  max_iterations = 1000000
    #for i in range(max_iterations):
        # 计算梯度
      #  grad = (np.matmul(X.T, (np.matmul(X, weight) - y))) + 1e-12 * np.sign(weight)
       # weight = weight - 1e-12 * grad
        #if np.linalg.norm(grad) < 0.0001:
          #  break
    return weight @ data
    
def lasso(data):
     X,y=read_data()
     m,n=X.shape
     w=np.zeros(n)
     alpha=0.01
     beta=1e-12
     mse = np.sum(((X @ w )- y.T) @ ((X @ w) - y.T).T)/(np.shape(X)[0])
     l1 = alpha * ((np.sum(np.abs(w))))
     lassoloss = mse + l1
     max_iterations=1000000
     for i in range(max_iterations):
         mse = np.sum(((X @ w )- y.T) @ ((X @ w) - y.T).T)/(np.shape(X)[0])
         l1 = alpha * ((np.sum(np.abs(w))))
         lassoloss = mse + l1
         dw = X.T @ ((X @ w) - y.T) + alpha * np.sign(w) 
         loss_old = lassoloss
         w = w - beta * dw
         if (np.abs(0.0001 - loss_old) < 0.0001):
            print('提前停止！')
            break
         if (0.0001 >= lassoloss):
             lassoloss=0.0001  
             best = w
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
