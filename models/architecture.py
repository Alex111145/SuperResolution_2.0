import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# --- MODIFICA IMPORT: Fix Path per 'utils' ---
# Questo script è in 'models/', env_setup è in 'utils/'. Dobbiamo salire alla root.
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_setup import import_external_archs
except ImportError:
    print("Attenzione: Impossibile importare utils.env_setup. Il codice potrebbe fallire.")
    class DummyArch: pass
    def import_external_archs(): return DummyArch, DummyArch

RRDBNet, HAT_Arch = import_external_archs()

class AntiCheckerboardLayer(nn.Module):
    """
    Applica un filtro Gaussiano (o Box-like) a basso passaggio per ridurre 
    gli artefatti a scacchiera (checkerboard artifacts), tipici della 
    upsampling con strati di deconvolution/pixel-shuffle.
    """
    def __init__(self, mode='balanced'):
        super().__init__()
        
        if mode == 'strong':
            k, p, s = 7, 3, 1600.0
            bk = [[1,6,15,20,15,6,1],[6,36,90,120,90,36,6],[15,90,225,300,225,90,15],
                  [20,120,300,400,300,120,20],[15,90,225,300,225,90,15],[6,36,90,120,90,36,6],[1,6,15,20,15,6,1]]
        elif mode == 'balanced':
            k, p, s = 5, 2, 256.0
            bk = [[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]
        elif mode == 'light':
            k, p, s = 3, 1, 16.0
            bk = [[1,2,1],[2,4,2],[1,2,1]]
        else: 
            k, p, s = 1, 0, 1.0
            bk = [[1]]

        kernel = torch.tensor(bk, dtype=torch.float32) / s
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('weight', kernel)
        self.padding = p

    def forward(self, x):
        """Esegue una convoluzione per ogni canale separatamente (groups=x.shape[1])."""
        if self.padding == 0:
             return x 
        
        return F.conv2d(x, 
                        self.weight.expand(x.shape[1], -1, -1, -1),
                        padding=self.padding, 
                        groups=x.shape[1]) 

class TrainHybridModel(nn.Module):
    """
    Modello Ibrido per Super Risoluzione Astronomica: 
    RRDBNet per estrazione feature robuste + HAT per raffinamento texture.
    """
    def __init__(self, output_size=512, smoothing='balanced', device='cpu'):
        super().__init__()
        self.output_size = output_size
        
        if RRDBNet is None: 
            raise ImportError("RRDBNet non è stato importato correttamente. Controlla utils/env_setup.")
        
        self.stage1 = RRDBNet(
            num_in_ch=1, num_out_ch=1, 
            num_feat=64,      
            num_block=23,     
            num_grow_ch=32, 
            scale=2 
        )
        
        self.has_stage2 = False
        self.stage2 = nn.Identity() 

        if HAT_Arch:
            try:
                self.stage2 = HAT_Arch(
                    img_size=128,     
                    patch_size=1, in_chans=1, 
                    embed_dim=96, 
                    depths=[6, 6, 6, 6], 
                    num_heads=[6, 6, 6, 6], 
                    window_size=8, 
                    compress_ratio=3, squeeze_factor=30, conv_scale=0.01, 
                    overlap_ratio=0.5, mlp_ratio=2., qkv_bias=True, 
                    upscale=2,
                    img_range=1., upsampler='pixelshuffle', resi_connection='1conv'
                )
                self.has_stage2 = True
                print("[TRAIN ARCH] HAT Inizializzato (x2 Upsampling).")
            except Exception as e:
                print(f"HAT Error during initialization: {e}")

        if smoothing != 'none':
            self.s1 = AntiCheckerboardLayer(smoothing)
            self.s2 = AntiCheckerboardLayer(smoothing) 
            self.sf = AntiCheckerboardLayer('light')
        else:
            self.s1 = self.s2 = self.sf = nn.Identity()

    def forward(self, x):
        """
        Input: Immagine LR [B, 1, 128, 128]
        Output: Immagine SR [B, 1, 512, 512]
        """
        
        x = self.stage1(x)
        x = self.s1(x) 
        
        if self.has_stage2: 
            x = self.stage2(x)
            x = self.s2(x) 

        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, 
                              size=(self.output_size, self.output_size), 
                              mode='bicubic', 
                              align_corners=False)
                              
        return self.sf(x)
