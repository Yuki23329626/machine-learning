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

fixed_z_ = torch.normal((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise

print ("fixed_z_:", fixed_z_)
print ("fixed_z_.shape", fixed_z_.shape)