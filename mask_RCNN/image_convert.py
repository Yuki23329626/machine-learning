# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime

# 本 code 是用來偵測物件(芒果)，並且去除背景後只輸出芒果(score最高的那一個)部分的圖，圖片大小會改變

# Root directory of the project
ROOT_DIR = os.getcwd()
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# 使用 train 好的 weight，來進行圖片物件偵測
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"shapes20200615T1714/mask_rcnn_shapes_0010.h5")
# 如果沒有 weight 的話，就用 coco trained weight
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")
 
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
 
class ShapesConfig(Config):
    # 設定 config
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
 
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
 
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
 
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'mango']
# Load a random image from the images folder

# 要進行轉換的原始資料 path
PATH_ORIGINAL_DATA = "../2-hw/datasets/C1-P1_Dev/1/"
imglist = os.listdir(PATH_ORIGINAL_DATA)
count = len(imglist)
# print("count", count)
# 輸出資料 path
PATH_OUTPUT_DATA = "C1-P1_Dev/1/"

# logger1.info('Just for testing')

from mrcnn.visualize import logging

for i in range(count):
  logging.info('index: %s', i)#print("count: ", i)
  logging.info('image name: %s', imglist[i])#print("image: ", imglist[i])
  fileName = imglist[i].split(".")[0]
  # print("filestr: ", filestr)

  image = skimage.io.imread(PATH_ORIGINAL_DATA + fileName + ".jpg")
 
  a=datetime.now()
  # Run detection
  results = model.detect([image], verbose=1)
  b=datetime.now()
  # 花費時間
  logging.info('time spent: %s', (b-a).seconds)# print("time spent: ",(b-a).seconds)
  r = results[0]

  # 圖片偵測芒果，並轉換成芒果大圖儲存
  visualize.convert_images(image, r['rois'], r['masks'], r['class_ids'], class_names, fileName, PATH_OUTPUT_DATA, r['scores'], show_bbox=False, show_mask=True)
