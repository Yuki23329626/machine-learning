from yacs.config import CfgNode as CN
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PATH = CN()
_C.MODEL = CN()
_C.DEVICE = CN()
_C.DATA = CN()

_C.DEVICE.GPU = 0 # <gpu_id>
_C.DEVICE.CUDA = True # use gpu or not

_C.PATH.TRAIN_SET = "./datasets/C1-P1_Train"
_C.PATH.DEV_SET =  "./datasets/C1-P1_Dev"
_C.PATH.TEST_SET = "./datasets/C1-P1_Test"
_C.PATH.TRAIN_CSV =  "./datasets/train.csv"
_C.PATH.DEV_CSV =  "./datasets/dev.csv"
_C.PATH.TEST_CSV =  "./datasets/test.csv"

_C.MODEL.OUTPUT_PATH = "./weights/model.pth" # <weight_output_path>
_C.MODEL.LR = 1e-3 # <learning_rate>
_C.MODEL.EPOCH = 50 # <train_epochs>

# -----------------------------------------------
# normalization parameters(suggestion)

_C.DATA.PIXEL_MEAN = [0.485, 0.456, 0.406] 
_C.DATA.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------

_C.DATA.RESIZE = [224, 224] # picture size after resizing
_C.DATA.NUM_WORKERS = 8 # use how many processors
_C.DATA.TRAIN_BATCH_SIZE = 32 # <train_batch_size>
_C.DATA.TEST_BATCH_SIZE = 16 # <test_batch_size>
_C.DATA.VALIDATION_SIZE = 0.2

_C.merge_from_file(os.path.join(BASE_PATH, "config.yml"))
