from model      import cnn_model
from config     import cfg
from datasets   import make_test_loader

import torch, os
import numpy as np

model = cnn_model()

weight_path = cfg.MODEL.OUTPUT_PATH
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = cfg.DEVICE.GPU

weight = torch.load(weight_path)
model.load_state_dict(weight)

if use_cuda:
    torch.cuda.set_device(gpu_id)
    model.cuda()

test_loader = make_test_loader(cfg)

model.eval()

test_loss = 0.
correct = 0
count = 0

import pandas as pd
df = pd.read_csv('./datasets/test_example.csv')
filename = df["image_id"].values
label_np = df['label'].values

filename = [i+"" for i in filename]
file_test = filename[:]

label_np = [i+"" for i in filename]
label_test = label_np[:]

z = list(zip(file_test, label_test))
for i in range(len(z)):
    print("z[i][0]: ", z[i][0], "z[i][1]: ", z[i][1])

import csv
# 開啟輸出的 CSV 檔案

label_set = ['A', 'B', 'C']

with open('./datasets/test_result.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    writer.writerow(['image_id', 'label'])
    with torch.no_grad():
        for data, target in test_loader:
            print("count: ", count)
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            test_loss += loss.item() * data.size(0)
            correct += (output.max(1)[1] == target).sum()
            print("output.max(1)[1]: ", output.max(1)[1].cpu().numpy()[0], "target: ", target.cpu().numpy())
            writer.writerow([ z[count][0], label_set[output.max(1)[1].cpu().numpy()[0]]])#label_set[output.max(1)[1].cpu().numpy()[0]]
            count = count+1
        
    # test_loss /= len(test_loader.dataset)
    # accuracy = 100. * correct / len(test_loader.dataset)

    # print('Test Loss: {:.6f}, Test Accuracy: {:.2f}% ({}/{})'.format(test_loss, accuracy, correct, len(test_loader.dataset)))
