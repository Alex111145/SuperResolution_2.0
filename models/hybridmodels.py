"""
Modello Ibrido HAT + Real-ESRGAN
Combina Hybrid Attention Transformer per struttura globale
con blocchi RRDB di Real-ESRGAN per dettagli fini
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hat_arch import HAT

class RRDBBlock(nn.Module):
    """Residual-in-Residual Dense Block da Real-ESRGAN"""
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

class ResidualDenseBlock(nn.Module):
    """Dense Block con connessioni residuali"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class HybridHATRealESRGAN(nn.Module):
    """
    Architettura Ibrida che combina:
    - HAT: cattura dipendenze globali e struttura generale
    - Real-ESRGAN (RRDB): aggiunge dettagli fini e texture
    """
    def __init__(
        self,
        img_size=128,
        in_chans=1,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=7,
        upscale=4,
        num_rrdb=23,  # Real-ESRGAN usa 23 blocchi RRDB
        num_feat=64,
        num_grow_ch=32
    ):
        super(HybridHATRealESRGAN, self).__init__()
        
        self.upscale = upscale
        
        # === PARTE 1: HAT per feature globali ===
        self.hat = HAT(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            upscale=1,  # No upscaling, solo feature extraction
            upsampler='pixelshuffle'
        )
        
        # Adattamento canali: HAT output -> RRDB input
        self.conv_adapt = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)
        
        # === PARTE 2: Blocchi RRDB per dettagli fini ===
        self.rrdb_blocks = nn.Sequential(
            *[RRDBBlock(num_feat, num_grow_ch) for _ in range(num_rrdb)]
        )
        
        # Connessione residuale globale
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # === PARTE 3: Upsampling finale ===
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        # 1. Feature extraction con HAT (cattura struttura globale)
        hat_feat = self.hat.forward_features(x)  # Usa solo feature extraction di HAT
        
        # 2. Adatta dimensioni per RRDB
        feat = self.conv_adapt(hat_feat)
        body_feat = self.rrdb_blocks(feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat  # Connessione residuale globale
        
        # 3. Upsampling con pixel shuffle (4x = 2x * 2x)
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        # 4. Ricostruzione finale
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        
        return out

    def load_pretrained_hat(self, hat_path):
        """Carica pesi pre-trained di HAT (opzionale per transfer learning)"""
        hat_state = torch.load(hat_path, map_location='cpu')
        self.hat.load_state_dict(hat_state, strict=False)
        print(f"✓ HAT pre-trained caricato da {hat_path}")
