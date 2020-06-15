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
count = 1

import pandas as pd
df = pd.read_csv('./datasets/dev.csv')
filename = df["image_id"].values
label_np = df['label'].values

filename = [i+"" for i in filename]
filename = [os.path.join("./datasets/C1-P1_Dev/1",i) for i in filename]
file_test = filename[:]
print(file_test)

# with torch.no_grad():
#     for data, target in test_loader:
#         print("count: ", count)
#         if use_cuda:
#             data, target = data.cuda(), target.cuda()

#         output = model(data)
#         loss = torch.nn.functional.cross_entropy(output, target)
#         test_loss += loss.item() * data.size(0)
#         correct += (output.max(1)[1] == target).sum()
#         print(output.max(1)[1], target)
#         count = count+1
        
#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)

#     print('Test Loss: {:.6f}, Test Accuracy: {:.2f}% ({}/{})'.format(test_loss, accuracy, correct, len(test_loader.dataset)))
