
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 1 # mean and standard deviation
s = np.random.normal(mu, sigma, 20) #Create Gaussian Noise
x = np.linspace(-3, 3, 20)
y = 2*x+s
baseLine = 2*x+0

# #Plot Gaussian Noise
# plt.subplot(1,1,1);
plt.xlim(-3,3)
plt.ylim(-15,30)
# plt.plot(x, y, 'ro', x, baseLine)

# plt.show()

# create dummy data for training
x_values = [x]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [y]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

import random
selected_points = np.arange(20)
random.shuffle(selected_points)

plt.plot(x_train, y_train, x_train, baseLine)
plt.show()

##Plot Gaussian Distribution
#plt.subplot(1,2,2);
#count, bins, ignored = plt.hist(s, 300, normed=True)
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * 
#         np.exp( - (bins - mu)**2 / (2 * sigma**2) ), 
#         linewidth=2, color='r')
#plt.show()
