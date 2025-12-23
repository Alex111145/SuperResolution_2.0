import sys
import os
from pathlib import Path

import torchvision.transforms.functional as TF_functional
sys.modules['torchvision.transforms.functional_tensor'] = TF_functional

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
HAT_ARCH_PATH = CURRENT_DIR / "hat_arch"

if HAT_ARCH_PATH.exists():
    sys.path.insert(0, str(HAT_ARCH_PATH))
    print(f"✓ Path HAT aggiunto: {HAT_ARCH_PATH}")
else:
    raise FileNotFoundError(f"Cartella HAT non trovata: {HAT_ARCH_PATH}")

try:
    from hat_arch import HAT
    print("✓ HAT importato correttamente")
except ImportError as e:
    raise ImportError(f"Impossibile importare HAT: {e}")


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._init_weights()
        
    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDBBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(RRDBBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class HybridHATRealESRGAN(nn.Module):
    def __init__(
        self,
        img_size=128,
        in_chans=1,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        upscale=4,
        num_rrdb=23,
        num_feat=64,
        num_grow_ch=32
    ):
        super(HybridHATRealESRGAN, self).__init__()
        
        self.upscale = upscale
        self.img_size = img_size
        
        self.hat = HAT(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            upscale=2,
            upsampler='pixelshuffle',
            img_range=1.0,
            resi_connection='1conv'
        )
        
        self.conv_adapt = nn.Conv2d(in_chans, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.rrdb_trunk = nn.Sequential(
            *[RRDBBlock(num_feat, num_grow_ch) for _ in range(num_rrdb)]
        )
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        self.conv_up = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in [self.conv_adapt, self.conv_body, self.conv_up,
                  self.conv_hr, self.conv_last]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        hat_out = self.hat(x)
        
        feat = self.lrelu(self.conv_adapt(hat_out))
        trunk_feat = feat
        
        body_feat = self.rrdb_trunk(feat)
        body_feat = self.conv_body(body_feat)
        feat = trunk_feat + body_feat
        
        feat = self.lrelu(self.conv_up(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        
        return out
    
    def load_pretrained_hat(self, hat_path):
        try:
            hat_state = torch.load(hat_path, map_location='cpu')
            if 'model_state_dict' in hat_state:
                hat_state = hat_state['model_state_dict']
            
            hat_state_cleaned = {k.replace('module.', ''): v for k, v in hat_state.items()}
            self.hat.load_state_dict(hat_state_cleaned, strict=False)
            print(f"✓ HAT pre-trained caricato da {hat_path}")
        except Exception as e:
            print(f"⚠️  Errore caricamento HAT pre-trained: {e}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("TEST MODELLO IBRIDO HAT + REAL-ESRGAN")
    print("=" * 70)
    
    model = HybridHATRealESRGAN(
        img_size=128,
        in_chans=1,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        upscale=4,
        num_rrdb=23
    ).to(device)
    
    print("\n📊 Testing forward pass...")
    x = torch.randn(1, 1, 128, 128).to(device)
    with torch.no_grad():
        y = model(x)
    
    print(f"\n✓ Test superato!")
    print(f"  • Input:  {x.shape}")
    print(f"  • Output: {y.shape}")
    print(f"  • Params: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)
