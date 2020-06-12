import numpy as np
import matplotlib.pyplot as plt

img_path = "data/000001.png"

img = plt.imread(img_path)
mu = 0
sigma = 1
normal_distribution_noise = np.random.normal(mu, sigma, img.shape[:-1]) #  np.zeros((224, 224), np.float32)

print(img)
print(normal_distribution_noise)