import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import scipy.io as scio
import numpy as np
from DataProvider3d import *
from Loss import *
import os
from UVnet import *
import argparse

parser = argparse.ArgumentParser(description='PyTorch deepSNAP Training')
parser.add_argument('--img_type', dest='img_type', default="Alias_img", type=str)
parser.add_argument('--label_type', dest='label_type', default="FS_img", type=str)
parser.add_argument('--gpu', default="1", dest='gpu', type=str)
parser.add_argument('--base_channel', default=32, dest='base_channel', type=int)
parser.add_argument('--checkpoint', default="./checkpoint_64.pth.tar", dest='checkpoint', type=str)


def predict():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_file_list, test_filename_list = get_files("./data/test/")
    data_test = DataProvider3d(test_file_list, test_filename_list, args.img_type, args.label_type)
    dataload = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=4)

    loss = MSELoss()
    model = ResUnet3d(1, 1, args.base_channel).to("cuda")
    model2 = UVUnet3d(1, 1, args.base_channel).to("cuda")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model2.load_state_dict(checkpoint['state_dict2'])
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model2.parameters())))
    model.eval()
    model2.eval()
    with torch.no_grad():
        loss_test = 0
        loss_test2 = 0
        flag = 0
        for inputs, label, name in dataload:
            print(name[0])
            flag += 1
            inputs =inputs.to(device)
            result = model(inputs).to(device)
            result2 = model2(result).to(device)
            loss_test += loss(result.cpu(), label.cpu()).item()
            loss_test2 += loss(result2.cpu(), label.cpu()).item()
            result = np.squeeze(result.cpu().numpy())
            result2 = np.squeeze(result2.cpu().numpy())
            label = np.array(label)
            scio.savemat("./result/" + name[0], mdict={'prediction_CNN': result, 'prediction': result2, 'label': np.squeeze(label), 'input': np.squeeze(inputs.cpu().numpy())})
        print(loss_test/flag, loss_test2/flag)


if __name__ == '__main__':
    predict()
