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

dirPath = "lfw/"
img_list = os.listdir(dirPath)
for i in img_list:
  if os.isdir(dirPath + img_list[i]):
    shutil.copyfile(dirPath + img_list[i]+ '/*', dirPath)
