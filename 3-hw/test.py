import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

# mu = 0
# sigma = 1
# fixed_z_ = torch.normal(mu, sigma, (5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise

# print ("fixed_z_:", fixed_z_)
# print ("fixed_z_.shape", fixed_z_.shape)

# 2020.06.22
def find_dir(path):
    # 函數功能: 遞迴顯示指定路徑下的所有檔案及資料夾名稱
    for fd in os.listdir(path):
        full_path=os.path.join(path,fd)
        if os.path.isdir(full_path):
            print('資料夾:',full_path)
            find_dir(full_path)
        else:
            print('檔案:',full_path)

find_dir('../')