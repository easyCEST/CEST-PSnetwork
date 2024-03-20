import random
import scipy.io as scio
import numpy as np
from torchvision.transforms.functional import to_tensor
import torch
import torchvision.transforms as transforms
import os


def get_files(path):
    file_list = []
    file_name_list = []
    for filepath,dirnames,filenames in os.walk(path):
        for filename in filenames:
            file_list.append(os.path.join(filepath, filename))
            file_name_list.append(str(filename))
    return file_list, file_name_list


class DataProvider3d:
    def __init__(self, file_list, filename_list, img_type, label_type, batch_size=1):
        self.train_root_path = "./data/train"
        self.file_list = file_list
        self.filename_list = filename_list
        self.img_type = img_type
        self.label_type = label_type
        self.batch_size = batch_size
        self.data = []
        for k in range(len(self.file_list)):
            self.data.append(scio.loadmat(self.file_list[k]))
        
    def __getitem__(self, index):
        index = index // self.batch_size
        data = self.data[index]
        name = self.filename_list[index]
        img_x = data[self.img_type]
        img_x = np.array(img_x)
        img_y = data[self.label_type]
        img_y = np.array(img_y)
        if img_x.shape[2] == 33:
            img_x = img_x[:, :, 1:33]
        if img_y.shape[2] == 33:
            img_y = img_y[:, :, 1:33]

        img = img_x[np.newaxis, :, :, :]
        label = img_y[np.newaxis, :, :, :]
        img = np.array(img)
        label = np.array(label)
        img = torch.from_numpy(img)
        img = torch.as_tensor(img, dtype=torch.float32)
        label = torch.from_numpy(label)
        label = torch.as_tensor(label, dtype=torch.float32)
        return img, label, name

    def __len__(self):
        return len(self.file_list * self.batch_size)
