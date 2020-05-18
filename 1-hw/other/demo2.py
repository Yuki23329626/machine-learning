# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 19:12:25 2018
@author: Bob
"""
import numpy as np
import matplotlib.pyplot as plt


mu, sigma = 0, 3 # mean and standard deviation
s = np.random.normal(mu, sigma, 10000) #Create Gaussian Noise

#Plot Gaussian Noise
plt.subplot(1,2,1);
plt.xlim(0,1000)
plt.ylim(-20,20)
plt.plot(np.linspace(1, 10000, num = 10000),s )

#Plot Gaussian Distribution
plt.subplot(1,2,2);
count, bins, ignored = plt.hist(s, 300, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * 
         np.exp( - (bins - mu)**2 / (2 * sigma**2) ), 
         linewidth=2, color='r')
plt.show()
