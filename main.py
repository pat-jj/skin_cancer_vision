import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# %matplotlib inline
from tqdm import tqdm
import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader

from utils import SkinDataset, compute_img_mean_std
from train import train, validate
from read_data import get_data

from train import train, validate

device = torch.device("cpu")

# '''

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--use_cuda', type=bool, default=True, help='')
    parser.add_argument('--samples', type=bool, default=False, help='')
    parser.add_argument('--num_epochs', type=int, default=20, help='')
    parser.add_argument('--train', default=True, type=bool, help='')

    opt = parser.parse_args()


    if opt.use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    base_dir = os.path.join('.', 'dataverse_files')
    image_path = glob(os.path.join(base_dir, '*.jpg'))
    id_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in image_path}

    print(base_dir)
    print(id_path_dict)
    df_train, df_val = get_data(base_dir, id_path_dict)

    # norm_mean,norm_std = compute_img_mean_std(image_path)
    norm_mean,norm_std=[0.7630374,0.5456423,0.5700383],[0.14092843,0.15261276,0.16997045]

    # resnet101
    input_size = 224

    # resnet18
    # input_size = 64
    train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(20),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)])

    val_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    # model = models.resnet18(pretrained=True)
    # model = models.resnext101_32x8d(pretrained=True)
    # model = EfficientNet.from_pretrained('efficientnet-b4')
    model = models.resnet50(pretrained=True)

    # resnet101
    model.fc = nn.Linear(in_features=2048, out_features=7)

    # resnet18
    # model.fc = nn.Linear(in_features=512, out_features=7)
    model.to(device)

    training_set = SkinDataset(df_train, transform=train_transform)
    train_loader = DataLoader(training_set, batch_size=16, shuffle=True, num_workers=1)

    validation_set = SkinDataset(df_val, transform=train_transform)
    val_loader = DataLoader(validation_set, batch_size=16, shuffle=False, num_workers=1)


    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    print("using ", device)
    if opt.train:
        epoch_num = opt.num_epochs
        best_val_acc = 0
        total_loss_val, total_acc_val = [],[]
        for epoch in tqdm(range(1, epoch_num+1)):
            loss_train, acc_train, total_loss_train, total_acc_train = train(train_loader, model, criterion, optimizer, epoch, device)
            loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch, device)
            total_loss_val.append(loss_val)
            total_acc_val.append(acc_val)
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                print('-'*50)
                print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
                print('-'*50)
                torch.save(model, './model.pkl')

    x = range(0, len(total_loss_val))

    plt.plot(x, total_loss_val, '.-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss_fig.png')

    plt.close()

    plt.plot(x, total_acc_val, '.-')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig('acc_fig.png')