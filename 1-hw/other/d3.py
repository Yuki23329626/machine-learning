
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 1 # mean and standard deviation
noise = np.random.normal(mu, sigma, 20) #Create Gaussian Noise
# x = np.linspace(-3, 3, 20)
# y = 2*x+noise
# baseLine = 2*x+0

# # #Plot Gaussian Noise
# # plt.subplot(1,1,1);
# plt.xlim(-3,3)
# plt.ylim(-15,30)
# # plt.plot(x, y, 'ro', x, baseLine)

# # plt.show()

# # create dummy data for training
# x_values = [x]
# x_train = np.array(x_values, dtype=np.float32)
# x_train = x_train.reshape(-1, 1)

# y_values = [y]
# y_train = np.array(y_values, dtype=np.float32)
# y_train = y_train.reshape(-1, 1)

# np.save('my_array_x', x_train)
# np.save('my_array_y', y_train)

import random
selected_points = np.arange(20)
random.shuffle(selected_points)

# plt.plot(x_train, y_train, 'o', x_train, baseLine)
# plt.show() # 印出測試用資料

#y=ax+b
# rng=np.random.RandomState(1)
# #numpy.random.randn(d0, d1, …, dn)是從常態分配中返回一個或多個值 
x=np.linspace(-3, 3, 20)
#numpy.random.rand(d0, d1, …, dn)的數值會產生在(0,1)之間
y=2*x+noise
plt.scatter(x,y,s=20, c='r')
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20,rotation=0)
plt.savefig('data_point')

sample_x = np.array([])
sample_y = np.array([])
for i in range(15):
    sample_x = np.append(sample_x, x[selected_points[i]])
    sample_y = np.append(sample_y, y[selected_points[i]])


def linear_regression(x,y):
  x = np.concatenate((np.ones((x.shape[0],1)),x[:,np.newaxis]),axis=1)
  y = y[:,np.newaxis]
  w_lin = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
  return w_lin

w_lin = linear_regression(x, y)

lr_x = [-3, 3]
lr_y = w_lin[1]*lr_x + w_lin[0] # w_lin[1]是斜率，w_lin[0]為截距

plt.plot(lr_x, lr_y)
plt.show()

def ridge_regression(x,y):
  X_SHAPE_0 = x.shape[0]
  x = np.concatenate((np.ones((X_SHAPE_0,1)),x[:,np.newaxis]),axis=1)
  y = y[:,np.newaxis]
  w_lin = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)+np.identity(X_SHAPE_0)),x.T),y)
  return w_lin

##Plot Gaussian Distribution
#plt.subplot(1,2,2);
#count, bins, ignored = plt.hist(s, 300, normed=True)
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * 
#         np.exp( - (bins - mu)**2 / (2 * sigma**2) ), 
#         linewidth=2, color='r')
#plt.show()
