import numpy as np
import matplotlib.pyplot as plt

img_path = "data/000001.png"

img = plt.imread(img_path)
mean = 0
var = 10
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (218, 178)) #  np.zeros((224, 224), np.float32)

noisy_image = np.zeros(img.shape, np.float32)

if len(img.shape) == 2:
    noisy_image = img + gaussian
else:
    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian

plt.normalize(noisy_image, noisy_image, 0, 255, plt.NORM_MINMAX, dtype=-1)
noisy_image = noisy_image.astype(np.uint8)

plt.show("img", img)
plt.show("gaussian", gaussian)
plt.show("noisy", noisy_image)

plt.waitKey(0)