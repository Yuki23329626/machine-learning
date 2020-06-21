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

def show_train_hist(testing_loss, validation_loss,  show = False, save = False, path = 'hw3-2.png'):
    x = range(len(testing_loss))

    plt.plot(x, testing_loss, label='training_loss')
    plt.plot(x, validation_loss, label='validation_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

testing_loss = [1.0191,0.9318,0.8424,0.8250,0.8091,0.8100,0.7920,0.7886,0.7833,0.7707,0.7670,0.7724,0.7621,0.7525,0.7505,0.7511,0.7348,0.7334,0.7406,0.7243,0.7278,0.7200,0.7246,0.7286,0.7185,0.7144,0.7221,0.7169,0.7109,0.7106,0.7122,0.7229,0.7193,0.7047,0.7251,0.7172,0.7179,0.7228,0.7189,0.7119,0.7168,0.7110,0.7133,0.7197,0.7252,0.7128,0.7157,0.7201,0.7188,0.7353]
validation_loss = [0.9555,0.8721,0.8273,0.8364,0.8202,0.8154,0.8126,0.8301,0.7989,0.8139,0.8094,0.7941,0.8128,0.8011,0.8026,0.8076,0.7968,0.8086,0.8070,0.8131,0.8058,0.8109,0.8101,0.8187,0.8131,0.8266,0.8242,0.8228,0.8452,0.8112,0.8302,0.8259,0.8203,0.8256,0.8191,0.8230,0.8422,0.8431,0.8207,0.8102,0.8625,0.8301,0.8761,0.8336,0.8213,0.8277,0.8284,0.8242,0.8410,0.8487]

show_train_hist(testing_loss, validation_loss, show=True, save=True)