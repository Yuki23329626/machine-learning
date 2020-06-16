import argparse
import base64
import json
import os
import os.path as osp

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils
import yaml

# labelme 將 json 檔案轉換之後儲存的目的地 path
# 最後會將檔案分到三個資料夾
dataset_root_path = "/home/nxshen/machine-learning/mask_RCNN/samples/trinmy/myinfo/"
PATH_ORIGINAL_DATA = dataset_root_path + "json/"
img_floder = dataset_root_path + "pic/"
mask_floder = dataset_root_path + "cv2_mask/"
yaml_folder = dataset_root_path + "labelme_json/"
imglist = os.listdir(PATH_ORIGINAL_DATA)
count = len(imglist)
# print("count", count)

for i in range(count):
    print("count: ", i)

    logger.warning('This script is aimed to demonstrate how to convert the '
                   'JSON file to a single image dataset.')
    logger.warning("It won't handle multiple JSON files to generate a "
                   "real-use dataset.")

    json_file = imglist[i]
    
    data = json.load(open(PATH_ORIGINAL_DATA + json_file))
    imageData = data.get('imageData')

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {'_background_': 0}

    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    lbl, _ = utils.shapes_to_label(
        img.shape, data['shapes'], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb'
    )
    
    PIL.Image.fromarray(img).save(img_floder + json_file.split(".")[0] + ".png", "PNG")

    if not os.path.exists(os.path.dirname(yaml_folder + json_file.split(".")[0] + "_json/label_viz.png")):
        try:
            os.makedirs(os.path.dirname(yaml_folder + json_file.split(".")[0] + "_json/label_viz.png"))
        except OSError as exc: # Guard against race condition
            print("OSerror")

    PIL.Image.fromarray(img).save(yaml_folder + json_file.split(".")[0] + "_json/img.png", "PNG")
    utils.lblsave(mask_floder + json_file.split(".")[0] + ".png", lbl)

    PIL.Image.fromarray(lbl_viz).save(yaml_folder + json_file.split(".")[0] + "_json/label_viz.png")

    with open(osp.join(yaml_folder + json_file.split(".")[0] + "_json/", 'label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')
    logger.warning('info.yaml is being replaced by label_names.txt')
    info = dict(label_names=label_names)
    with open(osp.join(yaml_folder + json_file.split(".")[0] + "_json/", 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f, default_flow_style=False)

    logger.info('Saved to: {}'.format(PATH_ORIGINAL_DATA))
