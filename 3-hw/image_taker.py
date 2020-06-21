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
import shutil

img_list = []
dirPath = "lfw/"
dir_list = os.listdir(dirPath)
for i in range(len(dir_list)):
  if os.path.isdir(dirPath + dir_list[i]):
    img_list = os.listdir(dirPath + dir_list[i])
    for j in range(len(img_list)):
      print(dirPath + dir_list[i] + '/' + img_list[j])
      shutil.move(dirPath + dir_list[i] + '/' + img_list[j], dirPath)
