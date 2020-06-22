from model import cnn_model
from config import cfg
from datasets import make_train_loader

import torch, os
import numpy as np
import matplotlib.pyplot as plt

import logging

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='log.2-hw.train',
                    filemode='w')

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger().addHandler(console)

# Now, we can log to the root logger, or any other logger. First the root...
# logging.info('just for test')

# Now, define a couple of other loggers which might represent areas in your
# application:

logger1 = logging.getLogger('2-hw.train.py')

def show_train_hist(training_loss, validation_loss,  show = False, save = False, path = 'hw3-2.png'):
    x = range(len(training_loss))

    plt.plot(x, training_loss, label='training_loss')
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

model = cnn_model()

valid_size  = cfg.DATA.VALIDATION_SIZE
epochs      = cfg.MODEL.EPOCH
lr          = cfg.MODEL.LR
weight_path = cfg.MODEL.OUTPUT_PATH
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = cfg.DEVICE.GPU


if use_cuda:
    torch.cuda.set_device(gpu_id)
    model = model.cuda()

train_loader, valid_loader = make_train_loader(cfg)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

training_loss = []
validation_loss = []

for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0.
    valid_loss = 0.

    for data, target in train_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()
    for data, target in valid_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
            
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        valid_loss += loss.item() * data.size(0)

    train_loss /= int(np.floor(len(train_loader.dataset) * (1 - valid_size)))
    valid_loss /= int(np.floor(len(valid_loader.dataset) * valid_size))
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, train_loss, valid_loss))
    logging.info("Training Loss: %s", train_loss)
    logging.info("Validation Loss: %s", valid_loss)
    training_loss = np.append(training_loss, train_loss)
    validation_loss = np.append(validation_loss, valid_loss)

show_train_hist(training_loss, validation_loss, save=True)

output_dir = "/".join(weight_path.split("/")[:-1])
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

torch.save(model.state_dict(), weight_path)

