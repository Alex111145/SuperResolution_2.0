"""
Modello Ibrido HAT + Real-ESRGAN per Super-Resolution Astronomica
Combina HAT (struttura globale) con blocchi RRDB (dettagli fini)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# === IMPORT HAT DA CARTELLA SEPARATA ===
CURRENT_DIR = Path(__file__).resolve().parent
HAT_PATH = CURRENT_DIR / "HAT"

if HAT_PATH.exists():
    sys.path.insert(0, str(HAT_PATH))
    try:
        from hat.archs.hat_arch import HAT
    except ImportError:
        try:
            from archs.hat_arch import HAT
        except ImportError:
            raise ImportError(f"HAT non trovato in {HAT_PATH}")
else:
    raise FileNotFoundError(f"Cartella HAT non trovata: {HAT_PATH}")


# === BLOCCHI RRDB DA REAL-ESRGAN ===

class ResidualDenseBlock(nn.Module):
    """Dense Block con connessioni residuali (5 conv layers)"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Inizializzazione pesi
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
        return x5 * 0.2 + x  # Residual scaling


class RRDBBlock(nn.Module):
    """Residual-in-Residual Dense Block (cuore di Real-ESRGAN)"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(RRDBBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x  # Residual scaling


# === MODELLO IBRIDO PRINCIPALE ===

class HybridHATRealESRGAN(nn.Module):
    """
    Architettura Ibrida per Super-Resolution Astronomica
    
    Pipeline:
    1. HAT → Feature extraction globali (struttura, contesto)
    2. Adattamento dimensioni → Passaggio HAT → RRDB
    3. RRDB (23 blocchi) → Raffinamento texture e dettagli fini
    4. Upsampling 4x → Ricostruzione finale HR
    
    Args:
        img_size: Dimensione patch input (default: 128)
        in_chans: Canali input (1 per grayscale astronomico)
        embed_dim: Dimensione embedding HAT (default: 180)
        depths: Profondità layer HAT per stage
        num_heads: Numero attention heads per stage
        window_size: Dimensione finestra per HAT attention
        upscale: Fattore upscaling (default: 4)
        num_rrdb: Numero blocchi RRDB (default: 23, come Real-ESRGAN)
        num_feat: Feature channels per RRDB (default: 64)
        num_grow_ch: Growth channels in Dense Blocks (default: 32)
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
        num_rrdb=23,
        num_feat=64,
        num_grow_ch=32
    ):
        super(HybridHATRealESRGAN, self).__init__()
        
        self.upscale = upscale
        self.img_size = img_size
        
        # === STAGE 1: HAT (Global Structure) ===
        # HAT estrae features globali con Hybrid Attention Transformer
        self.hat = HAT(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            upscale=1,  # NO upscaling qui, solo feature extraction
            upsampler='',  # Disabilitiamo l'upsampler di HAT
            img_range=1.0,
            resi_connection='1conv'
        )
        
        # Adattatore dimensioni: HAT output (embed_dim) → RRDB input (num_feat)
        self.conv_adapt = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # === STAGE 2: RRDB Trunk (Local Texture Refinement) ===
        # 23 blocchi RRDB per raffinare dettagli ad alta frequenza
        self.rrdb_trunk = nn.Sequential(
            *[RRDBBlock(num_feat, num_grow_ch) for _ in range(num_rrdb)]
        )
        
        # Convoluzione dopo RRDB trunk (come Real-ESRGAN)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # === STAGE 3: Upsampling Module (4x = 2x → 2x) ===
        # Prima fase upsampling 2x
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # Seconda fase upsampling 2x
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # === STAGE 4: HR Reconstruction ===
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
        
        # Inizializzazione pesi
        self._init_weights()
        
    def _init_weights(self):
        """Inizializza pesi per stabilità training"""
        for m in [self.conv_adapt, self.conv_body, self.conv_up1, 
                  self.conv_up2, self.conv_hr, self.conv_last]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        """
        Forward pass del modello ibrido
        
        Args:
            x: Input LR tensor [B, 1, H, W]
            
        Returns:
            sr: Output SR tensor [B, 1, H*4, W*4]
        """
        # === STAGE 1: HAT Feature Extraction ===
        # HAT cattura struttura globale e dipendenze long-range
        hat_feat = self.hat.forward_features(x)  # [B, embed_dim, H, W]
        
        # === STAGE 2: Adattamento e RRDB Processing ===
        # Adatta dimensioni per RRDB
        feat = self.lrelu(self.conv_adapt(hat_feat))  # [B, num_feat, H, W]
        
        # Salva feature per skip connection globale
        trunk_feat = feat
        
        # Passa attraverso trunk RRDB (23 blocchi)
        body_feat = self.rrdb_trunk(feat)
        body_feat = self.conv_body(body_feat)
        
        # Skip connection globale (come Real-ESRGAN)
        feat = trunk_feat + body_feat
        
        # === STAGE 3: Progressive Upsampling (4x = 2x → 2x) ===
        # Prima fase 2x
        feat = self.lrelu(self.conv_up1(F.interpolate(
            feat, scale_factor=2, mode='nearest'
        )))
        
        # Seconda fase 2x
        feat = self.lrelu(self.conv_up2(F.interpolate(
            feat, scale_factor=2, mode='nearest'
        )))
        
        # === STAGE 4: HR Reconstruction ===
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        
        return out
    
    def load_pretrained_hat(self, hat_path):
        """
        Carica pesi pre-trained di HAT (opzionale per transfer learning)
        
        Args:
            hat_path: Path al checkpoint HAT
        """
        try:
            hat_state = torch.load(hat_path, map_location='cpu')
            # Rimuovi prefisso "module." se presente (da DDP)
            if 'model_state_dict' in hat_state:
                hat_state = hat_state['model_state_dict']
            
            hat_state_cleaned = {}
            for k, v in hat_state.items():
                k_cleaned = k.replace('module.', '')
                hat_state_cleaned[k_cleaned] = v
            
            self.hat.load_state_dict(hat_state_cleaned, strict=False)
            print(f"✓ HAT pre-trained caricato da {hat_path}")
        except Exception as e:
            print(f"⚠️  Errore caricamento HAT pre-trained: {e}")
            raise


# === TEST MODELLO ===
if __name__ == "__main__":
    # Test architettura
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HybridHATRealESRGAN(
        img_size=128,
        in_chans=1,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=7,
        upscale=4,
        num_rrdb=23
    ).to(device)
    
    # Test forward pass
    x = torch.randn(1, 1, 128, 128).to(device)
    with torch.no_grad():
        y = model(x)
    
    print(f"✓ Test superato!")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Parametri totali: {sum(p.numel() for p in model.parameters()):,}")
