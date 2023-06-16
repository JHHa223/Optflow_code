from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import sys
import torch
from PIL import Image, ImageFilter
import torchvision
from torchvision import transforms
import glob

class Loader(Dataset):

    def __init__(self, dummy=0, path='', transform=None):
        self.dummy = dummy
        if dummy == 0:
            self.path = glob.glob('/data0/jhha223/OptFlow/srf/train_set_tvl1/*.npy')
        else:
            self.path = glob.glob('/data0/jhha223/OptFlow/srf/val_set_tvl1/*.npy')

        self.transform = transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(size=(400,400)),
            transforms.RandomCrop(size=(256,256)),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):

        img = np.load(self.path[idx])
        img = img.astype(np.uint8)
        img = self.transform(img)
        img = img.cpu().detach().numpy().reshape(4,256,256)

        return img

if __name__ == '__main__':
    loader = Loader()

