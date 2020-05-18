# https://www.itread01.com/content/1544541602.html
# https://stackoverflow.com/questions/20443846/python-pil-nameerror-global-name-image-is-not-defined

import pandas as pd	##1
import numpy as np
df = pd.read_csv('./datasets/train.csv')
print(df.info())
print(df.head())

# 看看一共多少個breed,把每種breed名稱和一個數字編號對應起來
from pandas import Series, DataFrame

label_np = df['label'].values
#label_np = Series.as_matrix(label)
print(type(label_np) )
print(label_np.shape)   #(5600,)

#看一下一共多少不同種類
label_set = set(label_np)
print(len(label_set))   #3

#構建一個編號與名稱對應的字典，以後輸出的數字要變成名字的時候用：
label_3_list = list(label_set)
dic = {}
for i in range(3):
    dic[label_3_list[i]] = i

# 處理id那一列，分割成兩段：
file = df["image_id"].values
#file =  Series.as_matrix(df["id"])
print(file.shape)

import os
file = [i+"" for i in file]
file = [os.path.join("./datasets/C1-P1_Train/2",i) for i in file ]
file_train = file[:4480]
file_valid = file[4480:]
print(file_train)

np.save( "file_train.npy" ,file_train )
np.save( "file_valid.npy" ,file_valid )

#處理breed那一列，分成兩段：
print(label_np.shape)
number = []
for i in range(5600):
    number.append(dic[label_np[i]])
number = np.array(number) 
number_train = number[:4480]
number_valid = number[4480:]
np.save( "number_train.npy" ,number_train )
np.save( "number_valid.npy" ,number_valid )

##
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

from PIL import Image

def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor

#當然出來的時候已經全都變成了tensor
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定義好 image 的路徑
        self.images = file_train
        self.target = number_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

class validset(Dataset):
    def __init__(self, loader=default_loader):
        #定義好 image 的路徑
        self.images = file_train
        self.target = number_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

class testset(Dataset):
    def __init__(self, loader=default_loader):
        #定義好 image 的路徑
        self.images = file_train
        self.target = number_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

##

train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=4,shuffle=True)
print(type(trainloader))
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images), type(labels))
print(images.size(), labels.size())

