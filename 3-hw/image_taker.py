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

dirPath = "lfw/"
results = next(os.walk(dirPath))
print( results[0] + results[1] + results[2] + '.jpg')
