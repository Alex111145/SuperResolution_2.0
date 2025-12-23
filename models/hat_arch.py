import math
import torch
import torch.nn as nn
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

class HAT(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=1, embed_dim=180, 
                 depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6),
                 window_size=7, compress_ratio=3, squeeze_factor=30,
                 conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=2.,
                 upscale=4, img_range=1., upsampler='pixelshuffle', resi_connection='1conv'):
        super(HAT, self).__init__()
        self.mean = torch.zeros(1, in_chans, 1, 1)
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
            m = []
            for _ in range(int(math.log(upscale, 2))):
                m.append(nn.Conv2d(64, 4 * 64, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
            self.upsample = nn.Sequential(*m)
            self.conv_last = nn.Conv2d(64, in_chans, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        if self.upsampler == 'pixelshuffle':
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        return x / self.img_range + self.mean
