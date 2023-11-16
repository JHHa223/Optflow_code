from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import sys
import torch
from PIL import Image, ImageFilter
import torchvision
from torchvision import transforms
import glob

# data structure: np.stack(img0,img1,img_pred,img_gt), size=(nx,ny,4)
# img0 -> radar image at t-10 with size=(nx,ny)
# img1 -> radar image at t with size=(nx,ny)
# img_pred -> predicted image at t+10 through optical flow with size=(nx,ny)
# img_gt -> ground truth at t+10 with size=(nx,ny)

class Loader(Dataset):

    def __init__(self, dummy=0, path='', transform=None):
        self.dummy = dummy
        if dummy == 0:
            self.path = glob.glob('/data0/jhha223/OptFlow/srf/train_set_tvl1/*.npy') 
        else:
            self.path = glob.glob('/data0/jhha223/OptFlow/srf/val_set_tvl1/*.npy')

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):

        img = np.load(self.path[idx])
        img = img.reshape(4,1024,1024) # for example, nx=256, ny=256

        return img

if __name__ == '__main__':
    loader = Loader()

