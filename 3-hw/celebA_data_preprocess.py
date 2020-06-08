import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

# root path depends on your computer
root = '/home/nxshen/machine-learning/3-hw/data/'
save_root = '/home/nxshen/machine-learning/3-hw/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'resized_celebA'):
    os.mkdir(save_root + 'resized_celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = imresize(img, (resize_size, resize_size))
    plt.imsave(fname=save_root + 'resized_celebA/' + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)