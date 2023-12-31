import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(ConvLReLU, self).__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
                                  nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(DoubleConv, self).__init__()
        padding = (kernel_size - 1)/2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=int(padding)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=int(padding)),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, conv=None):
        super(inconv, self).__init__()
        if conv:
            self.conv = conv
        else:
            self.conv = DoubleConv(in_ch, out_ch, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, conv=None):
        super(down, self).__init__()
        self.pool = nn.AvgPool2d(2)

        if conv is not None:
            self.conv = conv
        else:
            self.conv = DoubleConv(in_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, bilinear=False):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
