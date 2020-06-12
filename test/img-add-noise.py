import numpy as np
import matplotlib.pyplot as plt

img_path = "data/000001.png"

img = plt.imread(img_path)
mu = 0
sigma = 1
noise = np.random.normal(mu, sigma, img.shape[:-1]) #  np.zeros((224, 224), np.float32)
denominator = np.max(noise) - np.min(noise)

# print(img)
# print(np.max(normal_distribution_noise))
# print(np.min(normal_distribution_noise))

noise_percent = noise/denominator

fake_img = np.clip(img + noise_percent, 0, 1)

plt.imshow(img, "origin", 1, 2, 1)
plt.imshow(img, "normal distribution", 1, 2, 2)