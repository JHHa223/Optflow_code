import torch.nn.functional as F
from UNET_Parts import *

class UNetOptflow(nn.Module):
    def __init__(self, n_channels=3, output_channels=1):
        super(UNetMedium, self).__init__()
        self.inc = inconv(n_channels, 64, kernel_size=7)
        self.down0 = down(64, 128, kernel_size=5)
        self.down1 = down(128, 256, kernel_size=5)
        self.down2 = down(256, 512)
        self.down3 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)
        self.outc = outconv(32, output_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.outc(x)
        return x

