import numpy as np
import matplotlib.pyplot as plt

img_path = "data/000001.png"

img = plt.imread(img_path)
mu = 0
sigma = 1
# gaussian = np.random.normal(mean, sigma, (233, 178)) #  np.zeros((224, 224), np.float32)

print(img)
print(img[,,0].shape)