import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from Loss import *
import torch.backends.cudnn as cudnn
from DataProvider3d import *
import argparse
import os
from UVnet import *
import random


parser = argparse.ArgumentParser(description='PyTorch deepSNAP Training')
parser.add_argument('--lr', dest='lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=100, dest='num_epochs', type=int)
parser.add_argument('--gpu', default="1", dest='gpu', type=str)
parser.add_argument('--print_interval', default=100, dest='print_interval', type=int)
parser.add_argument('--base_channel', default=32, dest='base_channel', type=int)
parser.add_argument('--save_path', default='./checkpoint.pth.tar', dest='save_path', type=str)
parser.add_argument('--img_type', dest='img_type', default="img_rand", type=str)
parser.add_argument('--train_type', dest='train_type', default="1", type=str)
parser.add_argument('--label_type', dest='label_type', default="img_full", type=str)


def train():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model1 = ResUnet3d(1, 1, args.base_channel).to("cuda")
    model2 = UVUnet3d(1, 1, args.base_channel).to("cuda")
    loss_arr = []
    loss_arr_val = []
    loss_arr2 = []
    loss_arr_val2 = []
    if args.train_type == "2":
        checkpoint = torch.load(args.save_path)
        model1.load_state_dict(checkpoint['state_dict'])
        loss_arr = checkpoint['loss_arr']
        loss_arr_val = checkpoint['loss_arr_val']

    train_file_list, file_name_list_train = get_files("./data/train/")
    val_file_list, file_name_list_val = get_files("./data/val/")

    Loss = MSELoss()
    optimizer = optim.Adam(model1.parameters(), lr=args.lr)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 150], gamma=0.1)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[30, 150], gamma=0.1)

    data_train = DataProvider3d(train_file_list, file_name_list_train, args.img_type, args.label_type)
    data_val = DataProvider3d(val_file_list, file_name_list_val, args.img_type, args.label_type)

    dataload_train = DataLoader(data_train, batch_size=1, shuffle=True, num_workers=8)
    dataload_val = DataLoader(data_val, batch_size=1, shuffle=True, num_workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    for epoch in range(args.num_epochs):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        print('-' * 10)
        dt_size = len(dataload_train.dataset)
        epoch_loss = 0
        epoch_loss2 = 0
        step = 0
        for inputs, labels, file_name in dataload_train:
            step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            if args.train_type == "1":
                outputs1 = model1(inputs).to(device)
                outputs2 = 0
                loss = Loss(outputs1, labels)
                loss.backward()
                optimizer.step()
            if args.train_type == "2":
                outputs1 = model1(inputs).to(device)
                outputs2 = model2(outputs1.detach()).to(device)
                loss = Loss(outputs2, labels)
                loss.backward()
                optimizer2.step()
            if args.train_type == "3":
                outputs1 = model1(inputs).to(device)
                outputs2 = model2(outputs1).to(device)
                loss = Loss(outputs1, labels) + Loss(outputs2, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += Loss(outputs1, labels).item()
            epoch_loss2 += Loss(outputs2, labels).item()
            if step % args.print_interval == 0:
                print("epoch %d: %d/%d,train loss:%f, %f" % (
                    epoch + 1, step, (dt_size - 1) // dataload_train.batch_size + 1, Loss(outputs1, labels).item(),
                    Loss(outputs2, labels).item()))
        scheduler.step()
        scheduler2.step()
        loss_val, loss_val2 = validate(model1, model2, Loss, dataload_val, device)
        if args.train_type == "1":
            loss_arr.append(epoch_loss / step)
            loss_arr_val.append(loss_val)
        if args.train_type == "2":
            loss_arr2.append(epoch_loss2 / step)
            loss_arr_val2.append(loss_val2)
        print("epoch %d loss on train set:%f ,%f" % (epoch + 1, epoch_loss / step, epoch_loss2 / step))
        print("epoch %d loss on val set:%f ,%f" % (epoch + 1, loss_val, loss_val2))

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model1.state_dict(),
            'state_dict2': model2.state_dict(),
            'loss_arr': loss_arr,
            'loss_arr_val': loss_arr_val,
            'loss_arr2': loss_arr2,
            'loss_arr_val2': loss_arr_val2
        }, args.save_path)
    return model1


def validate(model, model2, Loss, dataload_val, device):
    with torch.no_grad():
        loss1 = 0
        loss2 = 0
        dt_size = len(dataload_val.dataset)
        for x, y, name in dataload_val:
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs).to("cuda")
            outputs2 = model2(outputs).to("cuda")
            loss1 += Loss(outputs, labels).item()
            loss2 += Loss(outputs2, labels).item()
    return loss1 / (dt_size - 1), loss2 / (dt_size - 1)


if __name__ == '__main__':
    train()

