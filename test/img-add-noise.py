import numpy as np
import matplotlib.pyplot as plt

img_path = "data/000001.png"

img = plt.imread(img_path)
mu = -10
sigma = 1
noise = np.random.normal(mu, sigma, img.shape) #  np.zeros((224, 224), np.float32)
denominator = np.max(noise) - np.min(noise)

# print(img)
# print(np.max(normal_distribution_noise))
# print(np.min(normal_distribution_noise))

noise_percent = noise/denominator

fake_img = np.clip(img + noise_percent, 0, 1)

row, col = 1, 2

plt.subplot(row, col, 1)
plt.imshow(img)

plt.subplot(row, col, 2)
plt.imshow(fake_img)

plt.show()