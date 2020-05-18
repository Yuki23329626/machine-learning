# PyTorch手把手自定義Dataloader讀取數據 https://zhuanlan.zhihu.com/p/35698470
# Python PIL NameError global name Image is not defined https://stackoverflow.com/questions/20443846/python-pil-nameerror-global-name-image-is-not-defined

from .transforms              import build_transform
from torchvision              import datasets
from torch.utils.data         import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import os

np.random.seed(11)

'''
# 四.使用：實例化一個dataset，然後用Daraloader包起來
train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=4,shuffle=True)

print(type(trainloader))
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images), type(labels))
print(images.size(), labels.size())
'''

def make_train_loader(cfg):
    
    num_workers = cfg.DATA.NUM_WORKERS
    batch_size  = cfg.DATA.TRAIN_BATCH_SIZE
    valid_size  = cfg.DATA.VALIDATION_SIZE
    train_path  = cfg.PATH.TRAIN_SET
    
    transforms = build_transform(cfg)

    #-----------------------------------讀取csv檔-----------------------------------
    import pandas as pd
    import numpy as np
    # 讀取csv檔
    df = pd.read_csv('./datasets/train.csv')
    # 用pandas把表格信息讀出來
    #print(df.info())
    #print(df.head())

    # 二.預處理
    # 我們要做的事情是：
    # 1)得到一個長list1 : 裡面是每張圖片的路徑
    # 2)另外一個長list2: 裡面是每張圖片對應的標籤(整數)，順序要和list1對應。
    # 3)把這兩個list切分出來一部分作為驗證集

    # 1)查看共有多少label,把每種label名稱和數字編號對應起來
    from pandas import DataFrame
    label_np = df['label'].values
    #print(type(label_np))
    #print(label_np.shape)   # (5600,)

    # 查看共有多少不同種類
    label_set = set(label_np)
    #print(len(label_set))   # 3
    #print(label_set)	# {'C', 'A', 'B'}

    # 構建一個編號與名稱對應的字典，以後輸出的數字要變成名字的時候用：
    label_3_list = list(label_set)
    dic = {}
    for i in range(3):
        dic[label_3_list[i]] = i

    # 2)處理csv中image_id那一列，分割成兩段：裡面就是圖片的路徑
    file = df["image_id"].values
    #print(file.shape)

    import os
    file = [i+"" for i in file]
    file = [os.path.join("./datasets/C1-P1_Train/2",i) for i in file]
    file_train = file[:4480]
    file_valid = file[4480:]
    #print(len(file_train))	# 5600
    #print(file_train)	# 圖片路徑

    # 分別儲存train及valid之對映好的檔案
    np.save("file_train.npy", file_train)
    np.save("file_valid.npy", file_valid)

    # 3)處理label那一列，將其分成train(4/5, 80)及valid(1/5, 20)兩段：
    # 分別儲存train及valid之label對映number
    #print(label_np.shape)
    number = []
    for i in range(5600):
        number.append(dic[label_np[i]])
    number = np.array(number) 
    number_train = number[:4480]
    number_valid = number[4480:]
    np.save("number_train.npy", number_train)
    np.save("number_valid.npy", number_valid)

    # 三.Dataloader
    # 我們已經有圖片路徑的list、target編號的list，將他們填到Dataset類裡即可
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

    # 因為他們出來的時候都已經全都變成tensor，所以我們只需要接著做以下三個函數定義

    #自定義Dataset只需要最下面一個class,繼承自Dataset類。

    #以下三個私有函數綜合起來看其實就是你告訴它你所有數據的長度，它每次給你返回一個shuffle過的index，以這個方式遍歷數據集，通過__getitem__(self, index)返回一組你要的(input,target)
    class trainset(Dataset):
        def __init__(self, loader=default_loader):
	    # 這裡一般要初始化一個loader,一個images_path的列表，一個target的列表
            self.images = file_train
            self.target = number_train
            self.loader = loader

        def __getitem__(self, index):
	    # 這裡是在給你一個index的時候，返回一個圖片的tensor和target的tensor
	    # 使用loader方法，經過歸一化，剪裁，類型轉化，從圖像變成tensor
            fn = self.images[index]
            img = self.loader(fn)
            target = self.target[index]
            return img,target

        def __len__(self):
            # return所有數據的個數
            return len(self.images)

    class validset(Dataset):
        def __init__(self, loader=default_loader):
	    #  這裡一般要初始化一個loader,一個images_path的列表，一個target的列表
            self.images = file_valid
            self.target = number_valid
            self.loader = loader

        def __getitem__(self, index):
            # 這裡是在給你一個index的時候，返回一個圖片的tensor和target的tensor
	    # 使用loader方法，經過歸一化，剪裁，類型轉化，從圖像變成tensor
            fn = self.images[index]
            img = self.loader(fn)
            target = self.target[index]
            return img,target

        def __len__(self):
            # return所有數據的個數
            return len(self.images)
    #------------------------------------------------------------------------------
    
    #trainset = datasets.ImageFolder(train_path, transform=transforms)
    
    num_train = len(trainset())
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(trainset(), batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    valid_loader = DataLoader(validset(), batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler)
    
    return train_loader, valid_loader


def make_test_loader(cfg):

    num_workers = cfg.DATA.NUM_WORKERS
    batch_size  = cfg.DATA.TEST_BATCH_SIZE
    test_path   = cfg.PATH.TEST_SET

    transforms = build_transform(cfg)

    #-----------------------------------讀取csv檔-----------------------------------
    import pandas as pd
    import numpy as np
    # 讀取csv檔
    df = pd.read_csv('./datasets/dev.csv')
    # 用pandas把表格信息讀出來
    #print(df.info())
    #print(df.head())

    # 二.預處理
    # 我們要做的事情是：
    # 1)得到一個長list1 : 裡面是每張圖片的路徑
    # 2)另外一個長list2: 裡面是每張圖片對應的標籤（整數），順序要和list1對應。
    # 3)把這兩個list切分出來一部分作為驗證集

    # 1)查看共有多少label,把每種label名稱和數字編號對應起來
    from pandas import DataFrame
    label_np = df['label'].values
    #print(type(label_np))
    #print(label_np.shape)   # (5600,)

    # 查看共有多少不同種類
    label_set = set(label_np)
    #print(len(label_set))   # 3
    #print(label_set)	# {'C', 'A', 'B'}

    #構建一個編號與名稱對應的字典，以後輸出的數字要變成名字的時候用：
    label_3_list = list(label_set)
    dic = {}
    for i in range(3):
        dic[label_3_list[i]] = i

    # 2)處理csv中image_id那一列，分割成兩段：裡面就是圖片的路徑
    file = df["image_id"].values
    #print(file.shape)

    import os
    file = [i+"" for i in file]
    file = [os.path.join("./datasets/C1-P1_Dev/1",i) for i in file]
    file_test = file[:]
    #print(len(file_test))	# 800
    #print(file_test)	# 圖片路徑

    # 儲存test之對映好的檔案
    np.save("file_test.npy", file_test)

    # 3)處理label那一列
    # 分別儲存test之label對映number
    #print(label_np.shape)
    number = []
    for i in range(800):
        number.append(dic[label_np[i]])
    number = np.array(number) 
    number_test = number[:]
    np.save("number_test.npy", number_test)

    # 三.Dataloader
    # 我們已經有圖片路徑的list、target編號的list，將他們填到Dataset類裡即可
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

    # 因為他們出來的時候都已經全都變成tensor，所以我們只需要接著做以下三個函數定義

    #自定義Dataset只需要最下面一個class,繼承自Dataset類。

    #以下三個私有函數綜合起來看其實就是你告訴它你所有數據的長度，它每次給你返回一個shuffle過的index，以這個方式遍歷數據集，通過__getitem__(self, index)返回一組你要的(input,target)

    class testset(Dataset):
        def __init__(self, loader=default_loader):
            # 這裡一般要初始化一個loader,一個images_path的列表，一個target的列表
            self.images = file_test
            self.target = number_test
            self.loader = loader

        def __getitem__(self, index):
            # 這裡是在給你一個index的時候，返回一個圖片的tensor和target的tensor
	    # 使用loader方法，經過歸一化，剪裁，類型轉化，從圖像變成tensor
            fn = self.images[index]
            img = self.loader(fn)
            target = self.target[index]
            return img,target

        def __len__(self):
            # return所有數據的個數
            return len(self.images)

    #------------------------------------------------------------------------------

    #testset = datasets.ImageFolder(test_path, transform=transforms)

    test_loader = DataLoader(testset(), batch_size=batch_size, num_workers=num_workers)

    return test_loader

