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

#torch.cuda.set_device(1)

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = fixed_z_.cuda()

def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=True, noise=fixed_z_):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = z_.cuda()

    G.eval()
    if isFix:
        test_images = G(noise)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

noise_hw3_3_1 = torch.normal(0, 1, (5 * 5, 100)).view(-1, 100, 1, 1)    # normal distribution noise
noise_hw3_3_2 = torch.normal(-10, 1, (5 * 5, 100)).view(-1, 100, 1, 1)    # normal distribution noise
noise_hw3_3_3 = torch.empty(5 * 5, 100).uniform_(0, 1).view(-1, 100, 1, 1)    # uniform distribution noise

noise_hw3_3_1 = noise_hw3_3_1.cuda()
noise_hw3_3_2 = noise_hw3_3_2.cuda()
noise_hw3_3_3 = noise_hw3_3_3.cuda()

G = generator(128)
G.cuda()

# 讀訓練好的模型參數
G.load_state_dict(torch.load('CelebA_DCGAN_results/generator_param.pkl'))

result_path = 'hw3-3/'
result_img1 = result_path + 'N(0,1)' + '.png'
result_img2 = result_path + 'N(-10,1)' + '.png'
result_img3 = result_path + 'U(0,1)' + '.png'

if not os.path.isdir(result_path):
    os.mkdir(result_path)

show_result(200+1, save=True, path=result_img1, isFix=True, noise=noise_hw3_3_1)
show_result(200+1, save=True, path=result_img2, isFix=True, noise=noise_hw3_3_2)
show_result(200+1, save=True, path=result_img3, isFix=True, noise=noise_hw3_3_3)