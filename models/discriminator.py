import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

class UNetConv2(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetConv2, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.model(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x, skip_input):
        x = self.model(x)
        out = torch.cat((x, skip_input), 1)
        return out

class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch=1, num_feat=64):
        super(UNetDiscriminatorSN, self).__init__()
        self.conv0 = nn.Sequential(
            spectral_norm(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = UNetConv2(num_feat, num_feat * 2)
        self.conv2 = UNetConv2(num_feat * 2, num_feat * 4)
        self.conv3 = UNetConv2(num_feat * 4, num_feat * 8)
        self.conv4 = UNetConv2(num_feat * 8, num_feat * 8)

        self.up1 = UNetUpBlock(num_feat * 8, num_feat * 8) 
        self.up2 = UNetUpBlock(num_feat * 16, num_feat * 4) 
        self.up3 = UNetUpBlock(num_feat * 8, num_feat * 2) 
        self.up4 = UNetUpBlock(num_feat * 4, num_feat)      

        self.final_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(num_feat, 1, 3, 1, 1, bias=False)) 
        )

    def forward(self, x):
        x0 = self.conv0(x); x1 = self.conv1(x0); x2 = self.conv2(x1); x3 = self.conv3(x2); x4 = self.conv4(x3)
        d1 = self.up1(x4, x3); d2 = self.up2(d1, x2); d3 = self.up3(d2, x1); d4 = self.up4(d3, x0)
        return self.final_conv(d4)
